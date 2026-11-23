"""Build SFT examples: one example per player move of each puzzle in a split.

Examples whose position matches any position the model is prompted at in dev or
test (every player-move position of those puzzles, since eval is interactive)
are dropped to prevent leakage.

Run: ``python -m src.sft_data --config configs/data.yaml --split sft --seed 0``
Writes ``data/processed/{split}.jsonl`` and ``data/processed/{split}.stats.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.chess_utils import player_positions
from src.data import load_config, load_split
from src.prompts import PROMPT_VERSION, sft_messages

EVAL_SPLITS = ("dev", "test")


def _position_key(board) -> str:
    """Same key as ``src.data.fen_key``: first 4 FEN fields, move counters ignored."""
    return " ".join(board.fen().split(" ")[:4])


def eval_position_keys(dfs: list[pd.DataFrame]) -> set[str]:
    """FEN keys (first 4 fields) of every player-move position in the given puzzles."""
    keys = set()
    for df in dfs:
        for fen, moves in zip(df["FEN"], df["Moves"]):
            keys.update(_position_key(b) for b, _ in player_positions(fen, moves))
    return keys


def build_sft_examples(
    df: pd.DataFrame, exclude_keys: set[str], seed: int
) -> tuple[pd.DataFrame, dict]:
    """One row per player move: PuzzleId, ply, band, fen, target, messages.

    ``ply`` is the player-move index within the puzzle (0 = first move).
    Rows whose position key is in ``exclude_keys`` are dropped. Rows are
    shuffled with ``seed``.
    """
    rows, dropped = [], 0
    for pid, fen, moves, band in zip(df["PuzzleId"], df["FEN"], df["Moves"], df["band"]):
        for ply, (board, move) in enumerate(player_positions(fen, moves)):
            if _position_key(board) in exclude_keys:
                dropped += 1
                continue
            rows.append(
                {
                    "PuzzleId": pid,
                    "ply": ply,
                    "band": int(band),
                    "fen": board.fen(),
                    "target": move.uci(),
                    "messages": sft_messages(board, move),
                }
            )
    ex = pd.DataFrame(rows, columns=["PuzzleId", "ply", "band", "fen", "target", "messages"])
    ex = ex.sample(frac=1, random_state=np.random.RandomState(seed)).reset_index(drop=True)
    stats = {
        "prompt_version": PROMPT_VERSION,
        "seed": seed,
        "n_puzzles": int(len(df)),
        "n_examples": int(len(ex)),
        "n_dropped_leakage": dropped,
        "examples_per_band": ex["band"].value_counts().sort_index().to_dict(),
        "examples_per_ply": ex["ply"].value_counts().sort_index().to_dict(),
    }
    return ex, stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Build SFT examples from a saved split.")
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--split", default="sft")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out-dir", default="data/processed")
    args = ap.parse_args()

    cfg = load_config(args.config)
    exclude = eval_position_keys([load_split(s, cfg) for s in EVAL_SPLITS])
    ex, stats = build_sft_examples(load_split(args.split, cfg), exclude, args.seed)
    stats |= {"split": args.split, "n_eval_position_keys": len(exclude)}

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ex.to_json(out / f"{args.split}.jsonl", orient="records", lines=True)
    (out / f"{args.split}.stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
