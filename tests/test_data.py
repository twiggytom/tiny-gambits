import numpy as np
import pandas as pd
import pytest

from src.data import (
    add_prompt_fen,
    assign_band,
    dedupe,
    fen_key,
    game_id,
    make_splits,
    nested_subsets,
    per_band_targets,
    quality_filter,
)

EDGES = [1000, 1200, 1400, 1600, 1800, 2000, 2200]
START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
# Knight on e5 instead of g1; Ne5-f3 reaches the same position as Ng1-f3 from START.
KNIGHT_E5 = "rnbqkbnr/pppppppp/8/4N3/8/8/PPPPPPPP/RNBQKB1R w KQkq - 0 1"


# --- quality filter ---------------------------------------------------------


def test_quality_filter_boundaries_are_strict():
    df = pd.DataFrame(
        {
            "PuzzleId": ["ok", "rd_eq", "plays_eq", "pop_eq"],
            "RatingDeviation": [99, 100, 99, 99],
            "NbPlays": [301, 301, 300, 301],
            "Popularity": [1, 1, 1, 0],
        }
    )
    out = quality_filter(df, max_rating_deviation=100, min_nb_plays=300, min_popularity=0)
    assert out["PuzzleId"].tolist() == ["ok"]


# --- game id / fen key ------------------------------------------------------


def test_game_id_parsing():
    s = pd.Series(["787zsVup/black#48", "F8M8OS71#53", "abc/white#1"])
    assert game_id(s).tolist() == ["787zsVup", "F8M8OS71", "abc"]


def test_fen_key_ignores_move_counters():
    a = START
    b = START.replace(" 0 1", " 12 40")
    assert fen_key(pd.Series([a]))[0] == fen_key(pd.Series([b]))[0]
    assert fen_key(pd.Series([a]))[0] == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -"


# --- prompt fen / dedup -----------------------------------------------------


def _puzzles(rows):
    return pd.DataFrame(rows, columns=["PuzzleId", "FEN", "Moves", "NbPlays"])


def test_add_prompt_fen_applies_setup_move_and_drops_invalid():
    df = _puzzles(
        [
            ("a", START, "g1f3 g8f6", 500),
            ("bad", START, "e7e5 g8f6", 500),  # Black move with White to move
        ]
    )
    out, n_bad = add_prompt_fen(df, processes=1)
    assert n_bad == 1
    assert out["PuzzleId"].tolist() == ["a"]
    assert out["prompt_fen"].iloc[0].startswith("rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R b ")


def test_dedupe_raw_fen_keeps_most_played():
    df = _puzzles(
        [
            ("low", START, "g1f3 g8f6", 400),
            ("high", START.replace(" 0 1", " 3 9"), "e2e4 e7e5", 900),  # same raw key
        ]
    )
    df, _ = add_prompt_fen(df, processes=1)
    assert dedupe(df)["PuzzleId"].tolist() == ["high"]


def test_dedupe_prompt_fen_only_duplicate():
    df = _puzzles(
        [
            ("via_g1", START, "g1f3 g8f6", 400),
            ("via_e5", KNIGHT_E5, "e5f3 g8f6", 800),  # different raw FEN, same prompt
            ("other", START, "e2e4 e7e5", 100),  # same raw FEN as via_g1
        ]
    )
    df, _ = add_prompt_fen(df, processes=1)
    assert fen_key(df["FEN"]).nunique() == 2
    assert fen_key(df["prompt_fen"]).nunique() == 2
    out = dedupe(df)
    # Raw-FEN pass: via_g1 (400) beats other (100) on START.
    # Prompt-FEN pass: via_e5 (800) beats via_g1 (400) on the shared prompt.
    assert out["PuzzleId"].tolist() == ["via_e5"]


# --- bands ------------------------------------------------------------------


@pytest.mark.parametrize(
    "rating, band",
    [(399, 0), (999, 0), (1000, 1), (1199, 1), (1200, 2), (1999, 5), (2000, 6), (2199, 6), (2200, 7), (3300, 7)],
)
def test_assign_band(rating, band):
    assert assign_band([rating], EDGES)[0] == band


def test_per_band_targets():
    assert per_band_targets(500, 8) == [63, 63, 63, 63, 62, 62, 62, 62]
    assert sum(per_band_targets(500, 8)) == 500
    assert per_band_targets(2000, 8) == [250] * 8


# --- splits -----------------------------------------------------------------


def _pool(n_per_band=200, n_bands=8, multi_game=None):
    rows = []
    for b in range(n_bands):
        for i in range(n_per_band):
            pid = f"p{b}_{i:04d}"
            rows.append((pid, f"g{b}_{i:04d}", b))
    df = pd.DataFrame(rows, columns=["PuzzleId", "game", "band"])
    if multi_game:
        df.loc[df["PuzzleId"].isin(multi_game), "game"] = "shared_game"
    return df


SIZES = {"test": 80, "dev": 20, "grpo_candidates": 160, "sft": 400}


def test_split_sizes_per_band_and_disjoint():
    df = _pool()
    splits = make_splits(df, SIZES, n_bands=8, seed=0)
    band = df.set_index("PuzzleId")["band"]
    for name, total in SIZES.items():
        assert len(splits[name]) == total
        counts = np.bincount(band.loc[splits[name]], minlength=8).tolist()
        assert counts == per_band_targets(total, 8)
    names = list(splits)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            assert not set(splits[names[i]]) & set(splits[names[j]])


def test_multi_puzzle_game_stays_together():
    shared = ["p3_0000", "p3_0001", "p3_0002"]
    df = _pool(multi_game=shared)
    for seed in range(20):
        splits = make_splits(df, SIZES, n_bands=8, seed=seed)
        homes = [name for name, ids in splits.items() if set(shared) & set(ids)]
        assert len(homes) <= 1
        if homes:
            assert set(shared) <= set(splits[homes[0]])


def test_test_split_independent_of_later_sizes():
    df = _pool()
    a = make_splits(df, SIZES, n_bands=8, seed=0)
    b = make_splits(df, {**SIZES, "sft": 200, "grpo_candidates": 80}, n_bands=8, seed=0)
    assert a["test"] == b["test"]
    assert a["dev"] == b["dev"]


def test_seed_determinism():
    df = _pool()
    assert make_splits(df, SIZES, 8, seed=1) == make_splits(df, SIZES, 8, seed=1)
    assert make_splits(df, SIZES, 8, seed=1)["test"] != make_splits(df, SIZES, 8, seed=2)["test"]


def test_input_row_order_does_not_matter():
    # Include a multi-puzzle game whose puzzles span bands, so ordering within a game matters.
    df = _pool(multi_game=["p2_0005", "p5_0007", "p6_0001"])
    for rs in range(5):
        shuffled = df.sample(frac=1, random_state=rs)
        assert make_splits(df, SIZES, 8, seed=0) == make_splits(shuffled, SIZES, 8, seed=0)


def test_not_enough_puzzles_raises():
    df = _pool(n_per_band=10)
    with pytest.raises(ValueError):
        make_splits(df, SIZES, n_bands=8, seed=0)


def test_nested_subsets():
    df = _pool()
    splits = make_splits(df, SIZES, n_bands=8, seed=0)
    band = df.set_index("PuzzleId")["band"]
    subs = nested_subsets(splits["sft"], band, {"sft_200": 200, "sft_80": 80}, n_bands=8)
    assert set(subs["sft_80"]) < set(subs["sft_200"]) < set(splits["sft"])
    assert np.bincount(band.loc[subs["sft_80"]], minlength=8).tolist() == [10] * 8
