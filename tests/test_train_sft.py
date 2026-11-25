import json
import math

import pytest
import yaml

from src.project_compute import project
from src.train_sft import (
    PaddingCollator,
    expected_steps,
    lr_tag,
    parse_args,
    require_secrets,
    run_name,
    tokenize_example,
    training_kwargs,
)

CFG = yaml.safe_load(open("configs/sft_pilot.yaml"))
MESSAGES = [
    {"role": "system", "content": "You are a chess engine. Reply with a single move in UCI notation."},
    {"role": "user", "content": "Position (FEN): 6k1/6pp/8/8/8/8/5PPP/RR4K1 w - - 0 1\nYou are playing White."},
    {"role": "assistant", "content": "a1a8"},
]
PROMO = MESSAGES[:2] + [{"role": "assistant", "content": "e7e8q"}]


def _qwen_tokenizer():
    pytest.importorskip("transformers")
    from huggingface_hub import try_to_load_from_cache
    from transformers import AutoTokenizer

    for repo in ["Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct"]:
        if isinstance(try_to_load_from_cache(repo, "tokenizer_config.json"), str):
            return AutoTokenizer.from_pretrained(repo)
    pytest.skip("No Qwen2.5-Instruct tokenizer in the local HF cache")


# --- frozen values ------------------------------------------------------------------


def test_config_matches_frozen_plan_values():
    assert CFG["model"]["base"] == "Qwen/Qwen2.5-3B-Instruct"
    assert CFG["lora"]["r"] == 16 and CFG["lora"]["alpha"] == 32
    assert set(CFG["lora"]["target_modules"]) == {
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    t = CFG["train"]
    assert t["per_device_batch_size"] * t["grad_accum"] == 32
    assert (t["epochs"], t["lr_scheduler"], t["warmup_ratio"], t["optim"], t["weight_decay"]) == (
        1, "cosine", 0.03, "adamw_8bit", 0.0)
    assert CFG["model"]["max_seq_length"] == 256 and t["group_by_length"]
    assert CFG["sweep_lrs"] == [1e-4, 2e-4, 5e-4]
    assert CFG["hub"]["private"] is True


# --- naming / steps / secrets ----------------------------------------------------


@pytest.mark.parametrize("lr,tag", [(2e-4, "2e-4"), (1e-4, "1e-4"), (5e-4, "5e-4"), (5e-5, "5e-5"), (1.5e-4, "1.5e-4")])
def test_lr_tag(lr, tag):
    assert lr_tag(lr) == tag


def test_run_name():
    assert run_name("sft_5k", 2e-4) == "sft-5k-lr2e-4"
    assert run_name("sft", 1e-4) == "sft-50k-lr1e-4"


def test_expected_steps_counts_trailing_partial_accumulation():
    # sft_5k: 12,356 examples → 773 micro-batches of 16 → 387 optimizer steps (last one has 1 batch)
    assert expected_steps(12356, 16, 2) == 387
    assert expected_steps(37026, 16, 2) == 1158
    assert expected_steps(123301, 16, 2) == 3854
    assert expected_steps(64, 16, 2) == 2
    assert expected_steps(32, 16, 2, epochs=3) == 3


def test_expected_steps_matches_hf_trainer(tmp_path):
    """The real Trainer agrees with expected_steps on a ragged dataset (5 examples, batch 2, accum 2)."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("accelerate")
    from transformers import Trainer, TrainingArguments

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(1))

        def forward(self, input_ids=None, attention_mask=None, labels=None):
            return {"loss": (self.w - 1).pow(2).sum()}

    ds = [{"input_ids": [1, 2], "attention_mask": [1, 1], "labels": [-100, 2]}] * 5
    kw = training_kwargs(CFG, 1e-3, 0, "float32", "hf-cpu")
    kw.update(output_dir=str(tmp_path), per_device_train_batch_size=2, gradient_accumulation_steps=2,
              report_to="none", save_strategy="no")
    tr = Trainer(model=Tiny(), args=TrainingArguments(**kw), train_dataset=ds,
                 data_collator=PaddingCollator(0))
    tr.train()
    assert tr.state.global_step == expected_steps(5, 2, 2) == 2


def test_require_secrets():
    require_secrets(env={"HF_TOKEN": "x", "WANDB_API_KEY": "y"})
    with pytest.raises(RuntimeError, match="WANDB_API_KEY"):
        require_secrets(env={"HF_TOKEN": "x"})
    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        require_secrets(env={"HF_TOKEN": "", "WANDB_API_KEY": "y"})


def test_parse_args():
    a = parse_args(["--lr", "5e-4", "--seed", "0"])
    assert a.lr == 5e-4 and a.seed == 0 and a.backend == "unsloth" and not a.offline
    with pytest.raises(SystemExit):
        parse_args(["--lr", "2e-4"])  # seed is required


# --- trainer kwargs --------------------------------------------------------------------


def test_training_kwargs_t4_fp16():
    pytest.importorskip("transformers")
    kw = training_kwargs(CFG, 2e-4, 0, "float16", "unsloth")
    assert kw["fp16"] and not kw["bf16"]
    assert kw["learning_rate"] == 2e-4 and kw["optim"] == "adamw_8bit"
    assert kw["per_device_train_batch_size"] == 16 and kw["gradient_accumulation_steps"] == 2
    assert kw["lr_scheduler_type"] == "cosine" and kw["weight_decay"] == 0.0
    assert kw["seed"] == kw["data_seed"] == 0
    assert kw.get("warmup_ratio", kw.get("warmup_steps")) == 0.03
    assert kw.get("train_sampling_strategy") == "group_by_length" or kw.get("group_by_length") is True


def test_training_kwargs_bf16_and_cpu():
    pytest.importorskip("transformers")
    assert training_kwargs(CFG, 1e-4, 1, "bfloat16", "unsloth")["bf16"]
    cpu = training_kwargs(CFG, 1e-4, 1, "float32", "hf-cpu")
    assert not cpu["fp16"] and not cpu["bf16"] and cpu["optim"] == "adamw_torch"


def test_training_kwargs_accepted_by_installed_transformers(tmp_path):
    pytest.importorskip("transformers")
    from transformers import TrainingArguments

    kw = training_kwargs(CFG, 2e-4, 0, "float32", "hf-cpu")
    args = TrainingArguments(output_dir=str(tmp_path), **kw)
    assert args.get_warmup_steps(387) == math.ceil(387 * 0.03) == 12


# --- tokenization / loss mask ------------------------------------------------------------


@pytest.mark.parametrize("messages,move", [(MESSAGES, "a1a8"), (PROMO, "e7e8q")])
def test_loss_only_on_assistant_move(messages, move):
    tok = _qwen_tokenizer()
    ex = tokenize_example(messages, tok, 256)
    ids, labels = ex["input_ids"], ex["labels"]
    assert len(ids) == len(labels) == len(ex["attention_mask"])
    kept = [t for t in labels if t != -100]
    assert tok.decode(kept) == move + "<|im_end|>"
    # The supervised tokens are a contiguous suffix; everything before is masked
    first = next(i for i, t in enumerate(labels) if t != -100)
    assert all(t == -100 for t in labels[:first]) and labels[first:] == ids[first:]
    # Masked part is exactly what eval feeds the model
    prompt = tok.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    assert tok.decode(ids[:first]) == prompt
    assert ids[-1] == tok.convert_tokens_to_ids("<|im_end|>")


def test_real_sft_example_tokenizes():
    tok = _qwen_tokenizer()
    try:
        line = open("data/processed/sft_5k.jsonl").readline()
    except FileNotFoundError:
        pytest.skip("data/processed/sft_5k.jsonl not built")
    row = json.loads(line)
    ex = tokenize_example(row["messages"], tok, 256)
    assert tok.decode([t for t in ex["labels"] if t != -100]) == row["target"] + "<|im_end|>"


def test_too_long_example_raises():
    tok = _qwen_tokenizer()
    with pytest.raises(ValueError, match="max_seq_length"):
        tokenize_example(MESSAGES, tok, 20)


def test_collator_pads_right():
    pytest.importorskip("torch")
    batch = PaddingCollator(pad_token_id=7)([
        {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [-100, 2, 3]},
        {"input_ids": [4], "attention_mask": [1], "labels": [4]},
    ])
    assert batch["input_ids"].tolist() == [[1, 2, 3], [4, 7, 7]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]
    assert batch["labels"].tolist() == [[-100, 2, 3], [4, -100, -100]]


# --- compute projection ----------------------------------------------------------------


def test_projection_arithmetic():
    info = {"split": "sft_5k", "n_examples": 12356, "seconds_per_step": 3.6, "load_seconds": 60.0}
    timings = {"setup_seconds": 300.0, "merge_seconds": 120.0, "eval_seconds": 180.0}
    rows = {r["split"]: r for r in project(info, timings, CFG)}
    assert rows["sft_5k"]["steps"] == 387 and rows["sft"]["steps"] == 3854
    assert rows["sft"]["train_hours"] == pytest.approx((60 + 3.6 * 3854) / 3600)
    assert rows["sft"]["session_hours"] == pytest.approx((300 + 60 + 3.6 * 3854 + 120 + 180) / 3600)
    assert rows["sft"]["fits_session"]
    slow = {r["split"]: r for r in project({**info, "seconds_per_step": 12.0}, timings, CFG)}
    assert not slow["sft"]["fits_session"]  # 3854 × 12 s ≈ 12.8 h > 12 h


def test_flat_eval_metrics():
    from src.wandb_log import flat_eval_metrics

    m = {"first_move_acc": 0.5, "full_solution_acc": 0.25, "illegal_rate": 0.1, "unparseable_rate": 0.0,
         "first_move_illegal_rate": 0.1, "first_move_unparseable_rate": 0.0, "n_puzzles": 4,
         "n_model_moves": 6, "by_band": {"0": {"n": 2, "first_move_acc": 1.0, "full_solution_acc": 0.5}}}
    out = flat_eval_metrics(m, "dev")
    assert out["dev/first_move_acc"] == 0.5 and out["dev/band0/full_solution_acc"] == 0.5
    assert all(k.startswith("dev/") for k in out)


def test_length_grouping_sampler_is_used(tmp_path):
    """group_by_length must actually take effect with our plain list-of-dicts dataset."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("accelerate")
    from transformers import Trainer, TrainingArguments
    from transformers.trainer_pt_utils import LengthGroupedSampler

    ds = [{"input_ids": [1] * n, "attention_mask": [1] * n, "labels": [1] * n} for n in range(1, 9)]
    kw = training_kwargs(CFG, 1e-3, 0, "float32", "hf-cpu")
    kw.update(output_dir=str(tmp_path), report_to="none", save_strategy="no")
    tr = Trainer(model=torch.nn.Linear(1, 1), args=TrainingArguments(**kw), train_dataset=ds,
                 data_collator=PaddingCollator(0))
    assert isinstance(tr._get_train_sampler(), LengthGroupedSampler)


def test_resolve_wandb_entity():
    from src.train_sft import resolve_wandb_entity

    class Api:
        def __init__(self, entity):
            self.default_entity = entity

    assert resolve_wandb_entity(env={"WANDB_ENTITY": "team-x"}, api_factory=lambda: Api("me")) == "team-x"
    assert resolve_wandb_entity(env={}, api_factory=lambda: Api("me")) == "me"
    with pytest.raises(RuntimeError, match="WANDB_ENTITY"):
        resolve_wandb_entity(env={}, api_factory=lambda: Api(None))


def test_start_wandb_run_passes_explicit_id_and_target():
    from src.train_sft import start_wandb_run

    calls = {}

    class FakeWandb:
        @staticmethod
        def init(**kw):
            calls.update(kw)
            return type("Run", (), {"id": kw["id"]})()

    run = start_wandb_run("sft-5k-lr2e-4", {"lr": 2e-4},
                          env={"WANDB_ENTITY": "team", "WANDB_PROJECT": "tiny-gambits"}, wandb_module=FakeWandb)
    assert calls["entity"] == "team" and calls["project"] == "tiny-gambits" and calls["name"] == "sft-5k-lr2e-4"
    assert calls["id"].startswith("sft-5k-lr2e-4-") and run.id == calls["id"]
    assert calls["config"] == {"lr": 2e-4}
