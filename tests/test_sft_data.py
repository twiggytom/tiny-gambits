import chess
import pandas as pd

from src.chess_utils import player_positions
from src.sft_data import build_sft_examples, eval_position_keys

PUZZLE_A = ("4r1k1/3q1ppp/8/8/8/8/4QPPP/4R1K1 b - - 0 1", "d7d8 e2e8 d8e8 e1e8")  # 2 player moves
PUZZLE_C = ("6k1/6pp/8/8/8/8/5PPP/RR4K1 b - - 0 1", "g8h8 a1a8")  # 1 player move


def _df(puzzles):
    return pd.DataFrame(
        [(f"p{i}", fen, moves, i % 8) for i, (fen, moves) in enumerate(puzzles)],
        columns=["PuzzleId", "FEN", "Moves", "band"],
    )


def test_one_example_per_player_move():
    ex, stats = build_sft_examples(_df([PUZZLE_A, PUZZLE_C]), set(), seed=0)
    assert len(ex) == 3 and stats["n_examples"] == 3 and stats["n_puzzles"] == 2
    a = ex[ex.PuzzleId == "p0"].sort_values("ply")
    assert a["target"].tolist() == ["e2e8", "e1e8"]
    assert a["ply"].tolist() == [0, 1]


def test_targets_legal_and_messages_match_fen():
    ex, _ = build_sft_examples(_df([PUZZLE_A, PUZZLE_C]), set(), seed=0)
    for _, row in ex.iterrows():
        board = chess.Board(row.fen)
        assert board.is_legal(chess.Move.from_uci(row.target))
        assert row.fen in row.messages[1]["content"]
        assert row.messages[-1] == {"role": "assistant", "content": row.target}


def test_leaking_position_is_dropped():
    # PUZZLE_A's second (mid-puzzle) position is also an eval prompt position.
    second = player_positions(*PUZZLE_A)[1][0]
    keys = {" ".join(second.fen().split()[:4])}
    ex, stats = build_sft_examples(_df([PUZZLE_A, PUZZLE_C]), keys, seed=0)
    assert stats["n_dropped_leakage"] == 1 and len(ex) == 2
    assert "e1e8" not in ex["target"].tolist()


def test_eval_position_keys_covers_all_player_positions():
    keys = eval_position_keys([_df([PUZZLE_A])])
    expected = {" ".join(b.fen().split()[:4]) for b, _ in player_positions(*PUZZLE_A)}
    assert keys == expected and len(keys) == 2


def test_seed_determinism():
    df = _df([PUZZLE_A, PUZZLE_C] * 5)
    a, _ = build_sft_examples(df, set(), seed=3)
    b, _ = build_sft_examples(df, set(), seed=3)
    c, _ = build_sft_examples(df, set(), seed=4)
    assert a.equals(b)
    assert not a[["PuzzleId", "ply"]].equals(c[["PuzzleId", "ply"]])
