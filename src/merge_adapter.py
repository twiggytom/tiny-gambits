"""Merge a trained SFT adapter into the 16-bit base for eval, without retraining.

Used when a session ended after training (the adapter and train_info.json are on the Hub but the
local outputs are gone), and for re-evaluating any pushed adapter. With ``--from-hub`` it downloads
the final adapter + train_info.json (not last-checkpoint/) from ``{namespace}/tiny-gambits-{run}``
into ``outputs/{run}/``. It then merges in the eval dtype (T4 → fp16, L4 → bf16) and writes
``merge_seconds`` into ``outputs/{run}/train_info.json``.

Run: python -m src.merge_adapter --config configs/sft_pilot.yaml --run-name sft-5k-lr2e-4 \
        --out /tmp/merged/sft-5k-lr2e-4 --from-hub
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import yaml

from src.evaluate import dtype_for_capability
from src.train_sft import merge_adapter, require_secrets

HUB_PATTERNS = ["adapter_config.json", "adapter_model.safetensors", "train_info.json"]


def hub_repo_id(cfg: dict, run_name: str, namespace: str) -> str:
    return f"{namespace}/{cfg['hub']['repo_prefix']}{run_name}"


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/sft_pilot.yaml")
    ap.add_argument("--run-name", required=True, help="e.g. sft-5k-lr2e-4")
    ap.add_argument("--out", required=True, help="Directory for the merged model")
    ap.add_argument("--from-hub", action="store_true", help="Download the adapter from the run's Hub repo first")
    ap.add_argument("--dtype", default=None, help="Default: from GPU compute capability (CPU: float32)")
    args = ap.parse_args(argv)
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    adapter_dir = Path(cfg["output_dir"]) / args.run_name

    if args.from_hub:
        require_secrets(("HF_TOKEN",))
        from huggingface_hub import snapshot_download, whoami

        repo = hub_repo_id(cfg, args.run_name, cfg["hub"]["namespace"] or whoami()["name"])
        snapshot_download(repo, allow_patterns=HUB_PATTERNS, local_dir=str(adapter_dir))
        print(f"Downloaded {HUB_PATTERNS} from {repo} → {adapter_dir}")
    for fn in ["adapter_config.json", "adapter_model.safetensors"]:
        if not (adapter_dir / fn).exists():
            raise FileNotFoundError(f"{adapter_dir / fn} missing (pass --from-hub?)")

    dtype = args.dtype
    if dtype is None:
        import torch

        dtype = dtype_for_capability(torch.cuda.get_device_capability()) if torch.cuda.is_available() else "float32"

    info_path = adapter_dir / "train_info.json"
    info = json.loads(info_path.read_text()) if info_path.exists() else {}
    base = info.get("base_model") or cfg["model"]["base"]
    t = time.time()
    merge_adapter(str(adapter_dir), base, args.out, dtype)
    info["merge_seconds"] = time.time() - t
    info["merge_dtype"] = dtype
    info_path.write_text(json.dumps(info, indent=2) + "\n")
    return info


if __name__ == "__main__":
    main()
