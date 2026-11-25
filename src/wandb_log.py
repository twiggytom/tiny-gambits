"""Attach eval metrics (and stage timings) to the W&B run that trained the model.

Run: python -m src.wandb_log --train-info outputs/{run}/train_info.json \
        --metrics results/{run}/dev/metrics.json [--timings outputs/{run}/timings.json]
Reads WANDB_API_KEY from the environment.
"""

from __future__ import annotations

import argparse
import json
import os


def flat_eval_metrics(metrics: dict, split: str) -> dict:
    """Scalar headline metrics plus per-band accuracy, keyed '{split}/...'."""
    out = {}
    for k in ["first_move_acc", "full_solution_acc", "illegal_rate", "unparseable_rate",
              "first_move_illegal_rate", "first_move_unparseable_rate", "n_puzzles", "n_model_moves"]:
        out[f"{split}/{k}"] = metrics[k]
    for band, m in metrics["by_band"].items():
        out[f"{split}/band{band}/first_move_acc"] = m["first_move_acc"]
        out[f"{split}/band{band}/full_solution_acc"] = m["full_solution_acc"]
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--train-info", required=True)
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--timings", default=None)
    ap.add_argument("--run-id", default=None,
                    help="Correct W&B run id (recorded back into train_info.json); for runs whose recorded id is wrong")
    args = ap.parse_args(argv)
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("WANDB_API_KEY is not set")
    import wandb

    info = json.load(open(args.train_info))
    metrics = json.load(open(args.metrics))
    if args.run_id and args.run_id != info.get("wandb_run_id"):
        info["wandb_run_id_recorded"] = info.get("wandb_run_id")
        info["wandb_run_id"] = args.run_id
        with open(args.train_info, "w") as f:
            json.dump(info, f, indent=2)
            f.write("\n")
        print(f"train_info.json: wandb_run_id corrected {info['wandb_run_id_recorded']} → {args.run_id}")
    if not info.get("wandb_run_id"):
        raise RuntimeError(f"No wandb_run_id in {args.train_info} (trained with --offline?)")
    run = wandb.init(entity=info["wandb_entity"], project=info["wandb_project"], id=info["wandb_run_id"],
                     resume="must")
    summary = flat_eval_metrics(metrics, args.split)
    if args.timings:
        summary.update({f"timing/{k}": v for k, v in json.load(open(args.timings)).items()})
    run.summary.update(summary)
    run.finish()
    print(f"Logged {len(summary)} values to W&B run {info['wandb_run_id']}")


if __name__ == "__main__":
    main()
