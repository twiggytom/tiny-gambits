"""Project Kaggle GPU time for the other SFT splits from one measured SFT run.

Training time scales with optimizer steps (all splits have the same per-example token length
distribution). Model load, merge, dev eval and session setup are fixed costs per run.

Inputs:
  --train-info  outputs/{run}/train_info.json written by src.train_sft
  --timings     timings.json written by notebooks/kaggle_sft.ipynb (wall-clock per stage); optional

Run: python -m src.project_compute --config configs/sft.yaml --train-info ... [--timings ...]
"""

from __future__ import annotations

import argparse
import json

import yaml

from src.train_sft import expected_steps


def project(train_info: dict, timings: dict | None, cfg: dict) -> list[dict]:
    """One row per target split: steps, projected train hours, and the run's total hours."""
    t = cfg["train"]
    p = cfg["projection"]
    sec_per_step = train_info["seconds_per_step"]
    timings = timings or {}
    setup = timings.get("setup_seconds", 0.0)       # clone + install + data build (once per session)
    merge = timings.get("merge_seconds", train_info.get("merge_seconds", 0.0))
    evals = timings.get("eval_seconds", 0.0)        # incl. vLLM load
    load = train_info.get("load_seconds", 0.0)

    rows = []
    targets = {train_info["split"]: train_info["n_examples"]}
    targets |= {k: v for k, v in p["examples"].items() if k not in targets}
    for split, n in targets.items():
        steps = expected_steps(n, t["per_device_batch_size"], t["grad_accum"], t["epochs"])
        train_s = load + sec_per_step * steps
        total_s = setup + train_s + merge + evals
        rows.append(
            {
                "split": split,
                "n_examples": n,
                "steps": steps,
                "train_hours": train_s / 3600,
                "session_hours": total_s / 3600,
                "fits_session": total_s / 3600 <= p["kaggle_session_hours"],
                "share_of_week": total_s / 3600 / p["kaggle_weekly_hours"],
            }
        )
    return rows


def format_table(rows: list[dict], session_hours: float = 12, weekly_hours: float = 45) -> str:
    lines = [
        f"| Split | Examples | Steps | Train h | Session h (setup+train+merge+eval) | Fits {session_hours:g} h session | Share of {weekly_hours:g} h week |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['split']} | {r['n_examples']:,} | {r['steps']:,} | {r['train_hours']:.2f} | "
            f"{r['session_hours']:.2f} | {'yes' if r['fits_session'] else 'NO'} | {r['share_of_week']:.0%} |"
        )
    return "\n".join(lines)


def main(argv=None) -> list[dict]:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/sft.yaml")
    ap.add_argument("--train-info", required=True)
    ap.add_argument("--timings", default=None)
    args = ap.parse_args(argv)
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    info = json.load(open(args.train_info))
    timings = json.load(open(args.timings)) if args.timings else None
    rows = project(info, timings, cfg)
    print(f"Measured on {info['gpu']} ({info['dtype']}): {info['seconds_per_step']:.2f} s/step, "
          f"peak VRAM {info.get('peak_vram_gb')}")
    print(format_table(rows, cfg["projection"]["kaggle_session_hours"], cfg["projection"]["kaggle_weekly_hours"]))
    return rows


if __name__ == "__main__":
    main()
