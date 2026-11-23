"""Eval harness: interactive puzzle protocol, batched greedy generation, metrics, saved results.

Every active puzzle advances one player move per batched generate call, so the
number of rounds equals the longest puzzle. Per-puzzle results (including every
model move and its position) are saved so later analysis (bootstrap CIs,
McNemar, Stockfish evaluation of moves) needs no re-generation.

Run: python -m src.evaluate --config configs/eval.yaml --model <path|hub id> \
        --backend vllm --split dev --run-name <name> --seed 0 [--limit N]
The test split additionally requires --final.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import platform
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Protocol

import pandas as pd
import yaml

from src.chess_utils import PuzzleResult, PuzzleState
from src.data import load_config, load_split
from src.prompts import PROMPT_VERSION, build_messages

Messages = list[dict[str, str]]


# --- backends ---------------------------------------------------------------


class Backend(Protocol):
    def generate(self, batch: list[Messages]) -> list[str]: ...


def dtype_for_capability(capability: tuple[int, int]) -> str:
    """Native bf16 needs compute capability >= 8.0 (Ampere+: L4, A100); older GPUs (T4 = 7.5) use fp16."""
    return "bfloat16" if tuple(capability) >= (8, 0) else "float16"


def _cuda_dtype() -> str:
    # Not torch.cuda.is_bf16_supported(): it also counts *emulated* bf16 and returns True on a T4,
    # which vLLM then rejects.
    import torch

    return dtype_for_capability(torch.cuda.get_device_capability())


class VLLMBackend:
    """Greedy generation with vLLM (Kaggle/Colab GPUs)."""

    def __init__(self, model: str, max_tokens: int, seed: int, dtype: str | None = None):
        from vllm import LLM, SamplingParams

        self.dtype = dtype or _cuda_dtype()
        self.llm = LLM(model=model, dtype=self.dtype, seed=seed)
        self.tokenizer = self.llm.get_tokenizer()
        self.params = SamplingParams(temperature=0.0, max_tokens=max_tokens, seed=seed)

    def generate(self, batch: list[Messages]) -> list[str]:
        prompts = [
            self.tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in batch
        ]
        outs = self.llm.generate(prompts, self.params, use_tqdm=False)
        return [o.outputs[0].text for o in outs]


class HFBackend:
    """Greedy generation with transformers (local CPU smoke tests; GPU fallback)."""

    def __init__(self, model: str, max_tokens: int, seed: int, dtype: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.manual_seed(seed)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = dtype or (_cuda_dtype() if self.device == "cuda" else "float32")
        self.tokenizer = AutoTokenizer.from_pretrained(model, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(model, dtype=getattr(torch, self.dtype))
        self.model.to(self.device).eval()
        self.max_tokens = max_tokens

    def generate(self, batch: list[Messages]) -> list[str]:
        import torch

        prompts = [
            self.tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in batch
        ]
        enc = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **enc,
                max_new_tokens=self.max_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        new = out[:, enc["input_ids"].shape[1]:]
        return self.tokenizer.batch_decode(new, skip_special_tokens=True)


BACKENDS = {"vllm": VLLMBackend, "hf": HFBackend}


# --- protocol ---------------------------------------------------------------


def evaluate(df: pd.DataFrame, backend: Backend, batch_size: int) -> list[PuzzleResult]:
    """Run the interactive protocol on every puzzle in ``df`` (needs PuzzleId, FEN, Moves)."""
    states = [PuzzleState(f, m, pid) for pid, f, m in zip(df["PuzzleId"], df["FEN"], df["Moves"])]
    while True:
        active = [s for s in states if not s.done]
        if not active:
            break
        for i in range(0, len(active), batch_size):
            chunk = active[i : i + batch_size]
            outputs = backend.generate([build_messages(s.board) for s in chunk])
            if len(outputs) != len(chunk):
                raise RuntimeError(f"Backend returned {len(outputs)} outputs for {len(chunk)} prompts")
            for state, text in zip(chunk, outputs):
                state.step(text)
    return [s.result() for s in states]


# --- metrics ----------------------------------------------------------------


def _acc(frame: pd.DataFrame) -> dict:
    return {
        "n": int(len(frame)),
        "first_move_acc": float(frame["first_move_correct"].mean()) if len(frame) else None,
        "full_solution_acc": float(frame["solved"].mean()) if len(frame) else None,
    }


def compute_metrics(results: list[PuzzleResult], df: pd.DataFrame, themes: list[str]) -> dict:
    """Dependent variables (except the Stockfish one, computed later from saved moves).

    Illegal/unparseable rates are per model move: count / all model moves attempted.
    First-move rates use one move per puzzle (every puzzle gets a first move).
    """
    r = pd.DataFrame(
        {
            "PuzzleId": [x.puzzle_id for x in results],
            "first_move_correct": [x.first_move_correct for x in results],
            "solved": [x.solved for x in results],
            "outcome": [x.outcome for x in results],
        }
    ).merge(df[["PuzzleId", "band", "Themes"]], on="PuzzleId", how="left", validate="one_to_one")

    all_moves = [m for x in results for m in x.moves]
    first_moves = [x.moves[0] for x in results]
    n_moves = len(all_moves)

    def rate(moves, status):
        return sum(m["status"] == status for m in moves) / len(moves) if moves else None

    by_theme = {}
    for t in themes:
        mask = r["Themes"].apply(lambda ts: t in list(ts))
        by_theme[t] = _acc(r[mask])

    return {
        "n_puzzles": int(len(r)),
        "n_model_moves": n_moves,
        **{k: v for k, v in _acc(r).items() if k != "n"},
        "illegal_rate": rate(all_moves, "illegal"),
        "unparseable_rate": rate(all_moves, "unparseable"),
        "first_move_illegal_rate": rate(first_moves, "illegal"),
        "first_move_unparseable_rate": rate(first_moves, "unparseable"),
        "outcomes": {k: int(v) for k, v in r["outcome"].value_counts().sort_index().items()},
        "by_band": {int(b): _acc(g) for b, g in r.groupby("band")},
        "by_theme": by_theme,
    }


# --- saving -----------------------------------------------------------------


def _versions() -> dict:
    out = {"python": platform.python_version()}
    for pkg in ["chess", "pandas", "transformers", "torch", "vllm", "unsloth"]:
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            pass
    return out


def _gpu_name() -> str:
    try:
        import torch

        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def save_results(
    results: list[PuzzleResult], df: pd.DataFrame, metrics: dict, run_info: dict, out_dir: Path
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = df.set_index("PuzzleId")
    with open(out_dir / "puzzles.jsonl", "w") as f:
        for res in results:
            row = meta.loc[res.puzzle_id]
            rec = {
                "PuzzleId": res.puzzle_id,
                "band": int(row["band"]),
                "Rating": int(row["Rating"]),
                "Themes": list(row["Themes"]),
                **dataclasses.asdict(res),
            }
            f.write(json.dumps(rec) + "\n")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (out_dir / "run_info.json").write_text(json.dumps(run_info, indent=2) + "\n")


def load_results(out_dir: Path) -> tuple[pd.DataFrame, dict, dict]:
    puzzles = pd.read_json(out_dir / "puzzles.jsonl", lines=True, dtype={"PuzzleId": str})
    metrics = json.loads((out_dir / "metrics.json").read_text())
    run_info = json.loads((out_dir / "run_info.json").read_text())
    return puzzles, metrics, run_info


# --- CLI --------------------------------------------------------------------


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Evaluate a model on a puzzle split (interactive, greedy).")
    ap.add_argument("--config", default="configs/eval.yaml")
    ap.add_argument("--model", required=True, help="Local path or HF Hub id")
    ap.add_argument("--backend", choices=sorted(BACKENDS), default="vllm")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--limit", type=int, default=None, help="Evaluate only the first N puzzles")
    ap.add_argument("--final", action="store_true", help="Required to evaluate on the test split")
    args = ap.parse_args(argv)
    if args.split == "test" and not args.final:
        ap.error("The test split is for final evaluation only; pass --final to confirm.")
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    data_cfg = load_config(cfg["data_config"])
    df = load_split(args.split, data_cfg)
    if args.limit:
        df = df.head(args.limit)

    backend = BACKENDS[args.backend](args.model, max_tokens=cfg["decoding"]["max_tokens"], seed=args.seed)
    started = datetime.now(timezone.utc)
    results = evaluate(df, backend, cfg["batch_size"])
    metrics = compute_metrics(results, df, cfg["themes"])
    run_info = {
        "run_name": args.run_name,
        "model": args.model,
        "backend": args.backend,
        "dtype": getattr(backend, "dtype", None),
        "gpu": _gpu_name(),
        "split": args.split,
        "limit": args.limit,
        "seed": args.seed,
        "prompt_version": PROMPT_VERSION,
        "decoding": cfg["decoding"],
        "batch_size": cfg["batch_size"],
        "versions": _versions(),
        "started_utc": started.isoformat(),
        "seconds": (datetime.now(timezone.utc) - started).total_seconds(),
    }
    out = Path(cfg["results_dir"]) / args.run_name / args.split
    save_results(results, df, metrics, run_info, out)
    summary = {k: metrics[k] for k in ["n_puzzles", "first_move_acc", "full_solution_acc", "illegal_rate", "unparseable_rate"]}
    print(json.dumps(summary, indent=2))
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
