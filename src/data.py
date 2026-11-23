"""Data pipeline: load Lichess puzzles → quality filter → dedup → split by game,
stratified by rating band → save PuzzleId lists to ``data/splits/``.

Run: ``python -m src.data --config configs/data.yaml --seed 0``

Downstream code loads splits with :func:`load_split` rather than re-sampling.
"""

from __future__ import annotations

import argparse
import json
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

from src.chess_utils import apply_setup_move

RAW_COLUMNS = [
    "PuzzleId",
    "GameId",
    "FEN",
    "Moves",
    "Rating",
    "RatingDeviation",
    "NbPlays",
    "Popularity",
    "Themes",
]
SPLIT_ORDER = ["test", "dev", "grpo_candidates", "sft"]


# --- config / loading -------------------------------------------------------


def load_config(path: str | os.PathLike) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def raw_paths(cfg: dict, download: bool = True) -> list[Path]:
    """Local paths of the raw parquet shards, downloading them at the pinned revision if missing."""
    ds = cfg["dataset"]
    raw_dir = Path(ds["raw_dir"])
    paths = [raw_dir / f for f in ds["files"]]
    missing = [p for p in paths if not p.exists()]
    if missing and download:
        from huggingface_hub import hf_hub_download

        for f in ds["files"]:
            hf_hub_download(
                ds["repo_id"], f, repo_type="dataset", revision=ds["revision"], local_dir=raw_dir
            )
    elif missing:
        raise FileNotFoundError(f"Missing raw shards: {missing}")
    return paths


def load_raw(cfg: dict, columns: list[str] = RAW_COLUMNS, filters=None) -> pd.DataFrame:
    return pd.read_parquet(raw_paths(cfg), columns=columns, filters=filters)


# --- pure transforms --------------------------------------------------------


def quality_filter(
    df: pd.DataFrame, max_rating_deviation: int, min_nb_plays: int, min_popularity: int
) -> pd.DataFrame:
    keep = (
        (df["RatingDeviation"] < max_rating_deviation)
        & (df["NbPlays"] > min_nb_plays)
        & (df["Popularity"] > min_popularity)
    )
    return df[keep]


def game_id(game_ids: pd.Series) -> pd.Series:
    """``787zsVup/black#48`` or ``F8M8OS71#53`` → ``787zsVup`` / ``F8M8OS71``."""
    return game_ids.str.split(r"[/#]", regex=True).str[0]


def fen_key(fens: pd.Series) -> pd.Series:
    """First four FEN fields (placement, side to move, castling, en passant); ignores move counters."""
    return fens.str.split(" ").str[:4].str.join(" ")


def _prompt_fen(args: tuple[str, str]) -> str | None:
    fen, moves = args
    try:
        board, _ = apply_setup_move(fen, moves)
    except ValueError:
        return None
    return board.fen()


def add_prompt_fen(df: pd.DataFrame, processes: int | None = None) -> tuple[pd.DataFrame, int]:
    """Add ``prompt_fen`` (position after the setup move). Drops rows whose line is
    invalid; returns ``(df, n_dropped)``."""
    items = list(zip(df["FEN"], df["Moves"]))
    if processes == 1 or len(items) < 10_000:
        fens = [_prompt_fen(x) for x in items]
    else:
        with Pool(processes) as pool:
            fens = pool.map(_prompt_fen, items, chunksize=5_000)
    out = df.assign(prompt_fen=fens)
    bad = out["prompt_fen"].isna()
    return out[~bad], int(bad.sum())


def _most_played_first(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["NbPlays", "PuzzleId"], ascending=[False, True], kind="mergesort")


def dedupe_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate raw FENs (first 4 fields), keeping the most-played puzzle."""
    df = _most_played_first(df)
    return df[~fen_key(df["FEN"]).duplicated()]


def dedupe(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicates on the raw FEN key and on the prompt FEN key, keeping the most-played puzzle."""
    df = dedupe_raw(df)
    return df[~fen_key(df["prompt_fen"]).duplicated()]


def assign_band(ratings: pd.Series | np.ndarray, edges: list[int]) -> np.ndarray:
    """Band index 0..len(edges): rating < edges[0] → 0, rating >= edges[-1] → len(edges)."""
    return np.searchsorted(np.asarray(edges), np.asarray(ratings), side="right")


def per_band_targets(total: int, n_bands: int) -> list[int]:
    """Split ``total`` evenly across bands; the remainder goes to the lowest bands."""
    base, rem = divmod(total, n_bands)
    return [base + (1 if b < rem else 0) for b in range(n_bands)]


def make_splits(
    df: pd.DataFrame, sizes: dict[str, int], n_bands: int, seed: int
) -> dict[str, list[str]]:
    """Disjoint, rating-stratified splits drawn at the game level.

    ``df`` needs ``PuzzleId``, ``game`` and ``band``. Each game is assigned the band
    of its lowest PuzzleId; all of a game's puzzles go to the same split. Per band,
    games are shuffled once (seeded) and consumed in ``SPLIT_ORDER``, so an earlier
    split (test first) never depends on the sizes of later ones. A split takes
    games while its per-band puzzle count stays within the target, so counts are
    exact when every game has one puzzle.

    Returns PuzzleIds per split, ordered band by band in shuffled order, so a
    per-band prefix of a split is itself a stratified random subset.
    """
    unknown = set(sizes) - set(SPLIT_ORDER)
    if unknown:
        raise ValueError(f"Unknown splits: {unknown}")
    rng = np.random.default_rng(seed)

    df = df.sort_values("PuzzleId", kind="mergesort")
    games = df.groupby("game", sort=True).agg(band=("band", "first"), ids=("PuzzleId", list))
    out: dict[str, list[list[str]]] = {name: [] for name in SPLIT_ORDER if name in sizes}
    targets = {name: per_band_targets(sizes[name], n_bands) for name in out}

    for b in range(n_bands):
        band_games = games[games["band"] == b]
        order = band_games["ids"].to_numpy()[rng.permutation(len(band_games))]
        pos = 0
        for name in out:
            target, taken, chosen = targets[name][b], 0, []
            while pos < len(order) and taken + len(order[pos]) <= target:
                chosen.extend(order[pos])
                taken += len(order[pos])
                pos += 1
            if taken < target and pos >= len(order):
                raise ValueError(f"Band {b}: not enough puzzles for split {name!r}")
            out[name].append(chosen)
    return {name: [pid for band_ids in per_band for pid in band_ids] for name, per_band in out.items()}


def nested_subsets(
    split_ids: list[str], bands: pd.Series, subsets: dict[str, int], n_bands: int
) -> dict[str, list[str]]:
    """Stratified nested subsets: per-band prefixes of an (already shuffled) split."""
    band_of = bands.loc[split_ids].to_numpy()
    by_band = [[pid for pid, b in zip(split_ids, band_of) if b == k] for k in range(n_bands)]
    out = {}
    for name, total in subsets.items():
        targets = per_band_targets(total, n_bands)
        for k, t in enumerate(targets):
            if t > len(by_band[k]):
                raise ValueError(f"{name}: band {k} has only {len(by_band[k])} puzzles")
        out[name] = [pid for k, t in enumerate(targets) for pid in by_band[k][:t]]
    return out


# --- save / load ------------------------------------------------------------


def save_splits(splits: dict[str, list[str]], out_dir: str | os.PathLike, manifest: dict) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, ids in splits.items():
        (out / f"{name}.txt").write_text("\n".join(ids) + "\n")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def read_split_ids(name: str, out_dir: str | os.PathLike = "data/splits") -> list[str]:
    return (Path(out_dir) / f"{name}.txt").read_text().split()


def load_split(name: str, cfg: dict) -> pd.DataFrame:
    """Rows for a saved split (in saved order), with ``prompt_fen``, ``game`` and ``band``."""
    ids = read_split_ids(name, cfg["out_dir"])
    df = load_raw(cfg, filters=[("PuzzleId", "in", ids)])
    df = df.set_index("PuzzleId").loc[ids].reset_index()
    df, n_bad = add_prompt_fen(df)
    if n_bad:
        raise ValueError(f"{n_bad} puzzles in split {name!r} have invalid lines")
    return df.assign(game=game_id(df["GameId"]), band=assign_band(df["Rating"], cfg["band_edges"]))


# --- pipeline ---------------------------------------------------------------


def build_splits(cfg: dict, seed: int) -> tuple[dict[str, list[str]], dict]:
    counts = {}
    df = load_raw(cfg)
    counts["raw"] = len(df)
    df = quality_filter(df, **cfg["quality_filter"])
    counts["after_quality_filter"] = len(df)
    df = dedupe_raw(df)  # cheap pass first, to cut setup-move work
    counts["after_raw_fen_dedup"] = len(df)
    df, n_bad = add_prompt_fen(df)
    counts["dropped_invalid_line"] = n_bad
    df = dedupe(df)
    counts["after_prompt_fen_dedup"] = len(df)

    edges = cfg["band_edges"]
    n_bands = len(edges) + 1
    df = df.assign(game=game_id(df["GameId"]), band=assign_band(df["Rating"], edges))
    counts["games"] = int(df["game"].nunique())
    counts["pool_per_band"] = df["band"].value_counts().sort_index().tolist()

    splits = make_splits(df, cfg["splits"], n_bands, seed)
    bands = df.set_index("PuzzleId")["band"]
    splits |= nested_subsets(splits["sft"], bands, cfg["sft_subsets"], n_bands)

    manifest = {
        "seed": seed,
        "dataset": cfg["dataset"]["repo_id"],
        "revision": cfg["dataset"]["revision"],
        "config": cfg,
        "counts": counts,
        "splits": {
            name: {
                "n": len(ids),
                "per_band": np.bincount(bands.loc[ids], minlength=n_bands).tolist(),
            }
            for name, ids in splits.items()
        },
    }
    return splits, manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    splits, manifest = build_splits(cfg, args.seed)
    save_splits(splits, cfg["out_dir"], manifest)
    print(json.dumps({"counts": manifest["counts"], "splits": manifest["splits"]}, indent=2))


if __name__ == "__main__":
    main()
