"""SFT with QLoRA: one example per player move, loss on the assistant move only.

Backends:
  unsloth  4-bit base + LoRA via Unsloth on a CUDA GPU (Kaggle T4 fp16 / L4 bf16). Used for real runs.
  hf-cpu   plain transformers + peft in fp32 on CPU. Local smoke tests only (tiny / 0.5B Qwen).

The trainer is HF ``Trainer`` on pre-tokenized examples. Labels are -100 everywhere except the
assistant move and its ``<|im_end|>``. Checkpoints (adapter + optimizer/scheduler/RNG state) are
pushed to the run's Hub repo as ``last-checkpoint/`` on every save. ``--resume`` pulls that folder
back and continues. The final adapter is pushed to the repo root. ``--merge`` (or
``merge_adapter``) merges the adapter into the 16-bit base for vLLM eval.

Secrets: ``HF_TOKEN`` and ``WANDB_API_KEY`` are read from the environment only (``--offline`` skips both).

Run: python -m src.train_sft --config configs/sft_pilot.yaml --lr 2e-4 --seed 0
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import yaml

from src.evaluate import dtype_for_capability
from src.prompts import PROMPT_VERSION

SPLIT_TAGS = {"sft_5k": "5k", "sft_15k": "15k", "sft": "50k"}
REQUIRED_SECRETS = ("HF_TOKEN", "WANDB_API_KEY")


# --- naming / bookkeeping -----------------------------------------------------


def lr_tag(lr: float) -> str:
    """2e-4 → '2e-4', 1e-4 → '1e-4', 5e-5 → '5e-5', 1.5e-4 → '1.5e-4'."""
    mant, exp = f"{lr:e}".split("e")
    mant = mant.rstrip("0").rstrip(".")
    return f"{mant}e{int(exp)}"


def run_name(split: str, lr: float) -> str:
    """Name shared by the output dir, Hub repo suffix, W&B run and results/ dir."""
    return f"sft-{SPLIT_TAGS.get(split, split)}-lr{lr_tag(lr)}"


def expected_steps(n_examples: int, per_device_batch: int, grad_accum: int, epochs: int = 1) -> int:
    """Optimizer steps HF Trainer takes: a trailing partial accumulation still makes a step."""
    batches = math.ceil(n_examples / per_device_batch)
    return epochs * math.ceil(batches / grad_accum)


def require_secrets(names=REQUIRED_SECRETS, env=None) -> None:
    env = os.environ if env is None else env
    missing = [n for n in names if not env.get(n)]
    if missing:
        raise RuntimeError(
            f"Missing environment variable(s): {', '.join(missing)}. "
            "Set them from Kaggle/Colab secrets (or export from .env locally), or pass --offline."
        )


def resolve_wandb_entity(env=None, api_factory=None) -> str:
    """W&B entity (user or team) for this run: $WANDB_ENTITY, else the account's default entity.

    Accounts without a default entity make the Trainer's wandb.init fail *after* the model has loaded
    ("entity not specified, and viewer has no default entity"), so this runs first and fails fast.
    """
    env = os.environ if env is None else env
    if env.get("WANDB_ENTITY"):
        return env["WANDB_ENTITY"]
    if api_factory is None:
        import wandb

        api_factory = wandb.Api
    entity = getattr(api_factory(), "default_entity", None)
    if not entity:
        raise RuntimeError(
            "W&B account has no default entity. Set WANDB_ENTITY to your W&B username or team "
            "(Kaggle: add a WANDB_ENTITY secret), or set a default team in wandb.ai → User settings."
        )
    return entity


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return None


def _versions() -> dict:
    out = {"python": platform.python_version()}
    for pkg in ["torch", "transformers", "peft", "trl", "unsloth", "bitsandbytes", "accelerate",
                "datasets", "huggingface_hub", "wandb"]:
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            pass
    return out


# --- data -----------------------------------------------------------------------


def load_examples(path: str | os.PathLike, limit: int | None = None) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def tokenize_example(messages: list[dict], tokenizer, max_seq_length: int) -> dict:
    """input_ids / attention_mask / labels for one chat example; loss only on the assistant turn.

    The prompt is the system + user turns rendered with ``add_generation_prompt=True`` (exactly what
    eval feeds the model). The completion is the assistant move + ``<|im_end|>``. The newline the chat
    template puts after ``<|im_end|>`` is dropped, so the sequence ends at the stop token.
    """
    assert messages[-1]["role"] == "assistant", "last message must be the assistant target"
    prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(messages, tokenize=False)
    eos = "<|im_end|>"
    completion = messages[-1]["content"] + eos
    if not full.startswith(prompt + completion):
        raise ValueError("Chat template: assistant turn is not a clean suffix of the generation prompt")
    p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    c_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
    ids = p_ids + c_ids
    if len(ids) > max_seq_length:
        raise ValueError(f"Example has {len(ids)} tokens > max_seq_length {max_seq_length}")
    if ids != tokenizer(prompt + completion, add_special_tokens=False)["input_ids"]:
        raise ValueError("Tokenizing prompt and completion separately changed the token boundary")
    return {
        "input_ids": ids,
        "attention_mask": [1] * len(ids),
        "labels": [-100] * len(p_ids) + c_ids,
    }


class PaddingCollator:
    """Right-pad a batch: input_ids with pad id, attention_mask with 0, labels with -100."""

    def __init__(self, pad_token_id: int):
        self.pad = pad_token_id

    def __call__(self, features: list[dict]):
        import torch

        n = max(len(f["input_ids"]) for f in features)

        def pad(key, value):
            return torch.tensor([f[key] + [value] * (n - len(f[key])) for f in features])

        return {
            "input_ids": pad("input_ids", self.pad),
            "attention_mask": pad("attention_mask", 0),
            "labels": pad("labels", -100),
        }


# --- trainer arguments ------------------------------------------------------------


def training_kwargs(cfg: dict, lr: float, seed: int, dtype: str | None, backend: str) -> dict:
    """TrainingArguments kwargs from config, adapted to the installed transformers version.

    transformers 5.x renamed ``group_by_length`` → ``train_sampling_strategy='group_by_length'`` and
    folded ``warmup_ratio`` into a float ``warmup_steps``. The Kaggle training env may have either API.
    """
    from transformers import TrainingArguments

    params = inspect.signature(TrainingArguments).parameters
    t = cfg["train"]
    kw = dict(
        num_train_epochs=t["epochs"],
        per_device_train_batch_size=t["per_device_batch_size"],
        gradient_accumulation_steps=t["grad_accum"],
        learning_rate=lr,
        lr_scheduler_type=t["lr_scheduler"],
        optim=t["optim"] if backend == "unsloth" else "adamw_torch",
        weight_decay=t["weight_decay"],
        max_grad_norm=t["max_grad_norm"],
        adam_beta1=t["adam_beta1"],
        adam_beta2=t["adam_beta2"],
        adam_epsilon=t["adam_epsilon"],
        logging_steps=t["logging_steps"],
        save_strategy="steps",
        save_steps=t["save_steps"],
        save_total_limit=1,
        seed=seed,
        data_seed=seed,
        fp16=dtype == "float16",
        bf16=dtype == "bfloat16",
        remove_unused_columns=False,
        dataloader_drop_last=False,
    )
    if "warmup_ratio" in params:
        kw["warmup_ratio"] = t["warmup_ratio"]
    else:
        kw["warmup_steps"] = float(t["warmup_ratio"])  # < 1 → ratio of total steps
    if t["group_by_length"]:
        if "train_sampling_strategy" in params:
            kw["train_sampling_strategy"] = "group_by_length"
        else:
            kw["group_by_length"] = True
    if backend == "hf-cpu" and "use_cpu" in params:
        kw["use_cpu"] = True
    return kw


# --- model ------------------------------------------------------------------------


def load_model(cfg: dict, model_name: str, backend: str, seed: int):
    """(model with LoRA attached, tokenizer, dtype string)."""
    lc = cfg["lora"]
    if backend == "unsloth":
        import unsloth  # noqa: F401  (must be imported before transformers/peft to patch them)
        import torch
        from unsloth import FastLanguageModel

        dtype = dtype_for_capability(torch.cuda.get_device_capability())
        kwargs = dict(
            model_name=model_name,
            max_seq_length=cfg["model"]["max_seq_length"],
            dtype=getattr(torch, dtype),
            load_in_4bit=cfg["model"]["load_in_4bit"],
        )
        if "use_exact_model_name" in inspect.signature(FastLanguageModel.from_pretrained).parameters:
            kwargs["use_exact_model_name"] = True  # quantize Qwen/... on the fly, don't swap repos
        model, tokenizer = FastLanguageModel.from_pretrained(**kwargs)
        model = FastLanguageModel.get_peft_model(
            model,
            r=lc["r"],
            lora_alpha=lc["alpha"],
            lora_dropout=lc["dropout"],
            bias=lc["bias"],
            target_modules=lc["target_modules"],
            use_gradient_checkpointing="unsloth",
            random_state=seed,
        )
        return model, tokenizer, dtype

    if backend == "hf-cpu":
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.manual_seed(seed)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
        lora = LoraConfig(
            r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"], bias=lc["bias"],
            target_modules=lc["target_modules"], task_type="CAUSAL_LM",
        )
        return get_peft_model(model, lora), tokenizer, "float32"

    raise ValueError(f"Unknown backend {backend!r}")


def merge_adapter(adapter_dir: str, base_model: str, out_dir: str, dtype: str) -> None:
    """Merge a LoRA adapter into the 16-bit base and save a plain HF model (for vLLM).

    Always merges into ``base_model`` (the 16-bit Qwen repo), never the 4-bit repo the adapter
    config may point at. The base model's tokenizer and generation_config are saved unchanged
    (eval sets every decoding param explicitly, so the shipped sampling defaults are never used).
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = AutoModelForCausalLM.from_pretrained(base_model, dtype=getattr(torch, dtype)).to(device)
    merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
    merged.save_pretrained(out_dir, safe_serialization=True)
    AutoTokenizer.from_pretrained(base_model).save_pretrained(out_dir)
    print(f"Merged {adapter_dir} into {base_model} ({dtype}) → {out_dir}")


# --- main -------------------------------------------------------------------------


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="SFT (QLoRA) on one SFT split.")
    ap.add_argument("--config", default="configs/sft_pilot.yaml")
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--split", default=None, help="Override the config split (e.g. sft_5k)")
    ap.add_argument("--backend", choices=["unsloth", "hf-cpu"], default="unsloth")
    ap.add_argument("--model", default=None, help="Override the base model (local smoke tests)")
    ap.add_argument("--limit", type=int, default=None, help="Use only the first N examples (smoke tests)")
    ap.add_argument("--max-steps", type=int, default=None, help="Stop after N steps (smoke tests)")
    ap.add_argument("--offline", action="store_true", help="No Hub push, no W&B (local tests)")
    ap.add_argument("--resume", action="store_true", help="Resume from the Hub repo's last-checkpoint/")
    ap.add_argument("--merge", default=None, metavar="DIR", help="After training, merge the adapter into DIR")
    ap.add_argument("--output-dir", default=None, help="Override outputs/{run_name}")
    return ap.parse_args(argv)


def main(argv=None) -> dict:
    args = parse_args(argv)
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    split = args.split or cfg["split"]
    model_name = args.model or cfg["model"]["base"]
    name = run_name(split, args.lr)
    out_dir = Path(args.output_dir or Path(cfg["output_dir"]) / name)
    out_dir.mkdir(parents=True, exist_ok=True)

    hub_id = None
    if not args.offline:
        require_secrets()
        from huggingface_hub import whoami

        ns = cfg["hub"]["namespace"] or whoami()["name"]
        hub_id = f"{ns}/{cfg['hub']['repo_prefix']}{name}"
        os.environ.setdefault("WANDB_PROJECT", cfg["wandb"]["project"])
        os.environ["WANDB_ENTITY"] = resolve_wandb_entity()
        # Fix the W&B run id up front so eval metrics can be attached to this run later (src.wandb_log).
        os.environ.setdefault("WANDB_RUN_ID", f"{name}-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}")

    t0 = time.time()
    model, tokenizer, dtype = load_model(cfg, model_name, args.backend, args.seed)
    load_seconds = time.time() - t0

    from transformers import Trainer, TrainingArguments

    rows = load_examples(Path(cfg["data_dir"]) / f"{split}.jsonl", args.limit)
    max_len = cfg["model"]["max_seq_length"]
    train_ds = [tokenize_example(r["messages"], tokenizer, max_len) for r in rows]

    kw = training_kwargs(cfg, args.lr, args.seed, dtype, args.backend)
    kw.update(
        output_dir=str(out_dir),
        run_name=name,
        report_to="none" if args.offline else "wandb",
        push_to_hub=not args.offline,
        hub_model_id=hub_id,
        hub_strategy="checkpoint",
        hub_private_repo=cfg["hub"]["private"],
    )
    if args.max_steps:
        kw["max_steps"] = args.max_steps
    trainer = Trainer(
        model=model,
        args=TrainingArguments(**kw),
        train_dataset=train_ds,
        data_collator=PaddingCollator(tokenizer.pad_token_id),
    )

    resume_from = None
    if args.resume:
        from huggingface_hub import snapshot_download

        snapshot_download(hub_id, allow_patterns=["last-checkpoint/*"], local_dir=str(out_dir))
        resume_from = str(out_dir / "last-checkpoint")
        print(f"Resuming from {hub_id} → {resume_from}")

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass
    started = datetime.now(timezone.utc)
    t1 = time.time()
    result = trainer.train(resume_from_checkpoint=resume_from)
    train_seconds = time.time() - t1

    trainer.save_model(str(out_dir))  # adapter only (PEFT)
    n_steps = trainer.state.global_step
    n_tokens = sum(len(x["input_ids"]) for x in train_ds)
    n_target_tokens = sum(sum(t != -100 for t in x["labels"]) for x in train_ds)
    info = {
        "run_name": name,
        "split": split,
        "base_model": model_name,
        "loaded_model": getattr(model.config, "_name_or_path", None),
        "backend": args.backend,
        "dtype": dtype,
        "gpu": _gpu_name(),
        "lr": args.lr,
        "seed": args.seed,
        "prompt_version": PROMPT_VERSION,
        "n_examples": len(train_ds),
        "n_tokens": n_tokens,
        "n_target_tokens": int(n_target_tokens),
        "max_seq_len_seen": max(len(x["input_ids"]) for x in train_ds),
        "expected_steps": expected_steps(len(train_ds), cfg["train"]["per_device_batch_size"],
                                         cfg["train"]["grad_accum"], cfg["train"]["epochs"]),
        "n_steps": n_steps,
        "max_steps_override": args.max_steps,
        "resumed": args.resume,
        "final_train_loss": result.training_loss,
        "load_seconds": load_seconds,
        "train_seconds": train_seconds,
        "seconds_per_step": train_seconds / max(n_steps, 1),
        "tokens_per_second": n_tokens / train_seconds if not args.max_steps else None,
        "peak_vram_gb": _peak_vram_gb(),
        "lora": cfg["lora"],
        "train": cfg["train"],
        "training_args": {k: v for k, v in kw.items() if isinstance(v, (int, float, str, bool, type(None)))},
        "hub_repo": hub_id,
        "wandb_run_id": None if args.offline else os.environ["WANDB_RUN_ID"],
        "wandb_project": None if args.offline else os.environ["WANDB_PROJECT"],
        "wandb_entity": None if args.offline else os.environ["WANDB_ENTITY"],
        "commit": _git_commit(),
        "versions": _versions(),
        "started_utc": started.isoformat(),
        "log_history": trainer.state.log_history,
    }
    (out_dir / "train_info.json").write_text(json.dumps(info, indent=2) + "\n")

    if not args.offline:
        import wandb

        if wandb.run is not None:
            wandb.run.summary.update({k: info[k] for k in [
                "n_examples", "n_steps", "train_seconds", "seconds_per_step", "tokens_per_second",
                "peak_vram_gb", "dtype", "gpu", "commit"]})
        trainer.push_to_hub(commit_message=f"{name}: final adapter ({n_steps} steps)")
        from huggingface_hub import upload_file

        upload_file(path_or_fileobj=str(out_dir / "train_info.json"), path_in_repo="train_info.json",
                    repo_id=hub_id, commit_message="train_info.json")

    summary = {k: info[k] for k in ["run_name", "n_examples", "n_steps", "final_train_loss",
                                    "train_seconds", "seconds_per_step", "peak_vram_gb", "dtype", "gpu"]}
    print(json.dumps(summary, indent=2))

    if args.merge:
        import gc

        del trainer, model
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except ImportError:
            pass
        t2 = time.time()
        merge_adapter(str(out_dir), model_name, args.merge, dtype)  # 16-bit base, eval dtype
        info["merge_seconds"] = time.time() - t2
        (out_dir / "train_info.json").write_text(json.dumps(info, indent=2) + "\n")
    return info


def _gpu_name() -> str:
    try:
        import torch

        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _peak_vram_gb() -> dict | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return {
            "allocated": torch.cuda.max_memory_allocated() / 2**30,
            "reserved": torch.cuda.max_memory_reserved() / 2**30,
            "total": torch.cuda.get_device_properties(0).total_memory / 2**30,
        }
    except ImportError:
        return None


if __name__ == "__main__":
    main()
