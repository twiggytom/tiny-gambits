import chess
import pytest

from src.chess_utils import (
    MoveStatus,
    apply_setup_move,
    check_move,
    is_correct_move,
    parse_uci,
    run_puzzle,
)

# Hand-built puzzles in Lichess format: FEN is *before* the opponent's setup move.

# Back-rank mate in 2 for White: Qxe8+ Qxe8 Rxe8#. No mate-in-1 at the start.
PUZZLE_A = ("4r1k1/3q1ppp/8/8/8/8/4QPPP/4R1K1 b - - 0 1", "d7d8 e2e8 d8e8 e1e8")
# Recorded line is h3 Qa6 Qxe8#, but Qxe8# is already available on move one.
PUZZLE_B = ("4r1k1/5ppp/8/8/q7/8/4QPPP/4R1K1 b - - 0 1", "a4a5 h2h3 a5a6 e2e8")
# Mate in 1 with two mating moves: Ra8# (solution) and Rb8#.
PUZZLE_C = ("6k1/6pp/8/8/8/8/5PPP/RR4K1 b - - 0 1", "g8h8 a1a8")


def scripted(outputs):
    it = iter(outputs)
    return lambda board: next(it)


# --- apply_setup_move -------------------------------------------------------


def test_setup_move_is_applied():
    fen, moves = PUZZLE_A
    board, solution = apply_setup_move(fen, moves)
    assert board.turn == chess.WHITE  # FEN had Black to move
    assert len(board.move_stack) == 1
    assert board.move_stack[0] == chess.Move.from_uci("d7d8")
    assert board.piece_at(chess.D8) == chess.Piece(chess.QUEEN, chess.BLACK)
    assert board.piece_at(chess.D7) is None
    assert [m.uci() for m in solution] == ["e2e8", "d8e8", "e1e8"]


def test_setup_move_does_not_mutate_or_reuse_fen_state():
    fen, moves = PUZZLE_C
    b1, _ = apply_setup_move(fen, moves)
    b2, _ = apply_setup_move(fen, moves)
    assert b1.fen() == b2.fen()
    assert b1.piece_at(chess.H8) == chess.Piece(chess.KING, chess.BLACK)


def test_illegal_setup_move_raises():
    fen, _ = PUZZLE_A
    with pytest.raises(ValueError):
        apply_setup_move(fen, "e2e8 d7d8")  # White's move, but Black is to move


def test_illegal_solution_move_raises():
    fen, _ = PUZZLE_A
    with pytest.raises(ValueError):
        apply_setup_move(fen, "d7d8 a1a8")


def test_line_without_solution_move_raises():
    fen, _ = PUZZLE_A
    with pytest.raises(ValueError):
        apply_setup_move(fen, "d7d8")


# --- parse_uci --------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("e2e4", "e2e4"),
        (" E2E4\n", "e2e4"),
        ("e7e8q", "e7e8q"),
        ("E7E8Q", "e7e8q"),
        ("\te1g1  ", "e1g1"),
    ],
)
def test_parse_uci_valid(text, expected):
    assert parse_uci(text) == chess.Move.from_uci(expected)


@pytest.mark.parametrize(
    "text",
    ["", "   ", "e2e4 e7e5", "e2e4.", "Move: e2e4", "Nf3", "0000", "e9e4", "e7e8k", "e2-e4", "e2e"],
)
def test_parse_uci_unparseable(text):
    assert parse_uci(text) is None


# --- check_move -------------------------------------------------------------


def test_check_move_legal():
    assert check_move("e2e4", chess.Board()) == (MoveStatus.LEGAL, chess.Move.from_uci("e2e4"))


def test_check_move_blocked_is_illegal():
    status, move = check_move("d1d4", chess.Board())
    assert status is MoveStatus.ILLEGAL
    assert move == chess.Move.from_uci("d1d4")


def test_check_move_no_piece_is_illegal():
    assert check_move("e4e5", chess.Board())[0] is MoveStatus.ILLEGAL


def test_pinned_piece_move_is_illegal():
    # White knight on e2 is pinned to the king on e1 by the rook on e8.
    board = chess.Board("4r1k1/8/8/8/8/8/4N3/4K3 w - - 0 1")
    assert check_move("e2c3", board)[0] is MoveStatus.ILLEGAL


def test_king_into_check_is_illegal():
    board = chess.Board("4r1k1/8/8/8/8/8/8/3K4 w - - 0 1")
    assert check_move("d1e1", board)[0] is MoveStatus.ILLEGAL
    assert check_move("d1c1", board)[0] is MoveStatus.LEGAL


def test_check_move_unparseable():
    assert check_move("hello", chess.Board()) == (MoveStatus.UNPARSEABLE, None)


def test_promotion_requires_suffix():
    board = chess.Board("k7/4P3/8/8/8/8/8/4K3 w - - 0 1")
    assert check_move("e7e8", board)[0] is MoveStatus.ILLEGAL
    assert check_move("e7e8q", board)[0] is MoveStatus.LEGAL
    assert check_move("e7e8n", board)[0] is MoveStatus.LEGAL


def test_castling_legal_and_normalized():
    board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    assert check_move("e1g1", board) == (MoveStatus.LEGAL, chess.Move.from_uci("e1g1"))
    # King-takes-rook notation is normalized to standard UCI.
    assert check_move("e1h1", board) == (MoveStatus.LEGAL, chess.Move.from_uci("e1g1"))


def test_castling_illegal_without_rights():
    board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w - - 0 1")
    assert check_move("e1g1", board)[0] is MoveStatus.ILLEGAL


# --- is_correct_move --------------------------------------------------------


def test_correct_move_exact_solution():
    board, solution = apply_setup_move(*PUZZLE_A)
    assert is_correct_move(board, solution[0], solution[0])


def test_correct_move_alternative_mate():
    board, solution = apply_setup_move(*PUZZLE_C)
    assert is_correct_move(board, chess.Move.from_uci("b1b8"), solution[0])


def test_wrong_move():
    board, solution = apply_setup_move(*PUZZLE_A)
    assert not is_correct_move(board, chess.Move.from_uci("h2h3"), solution[0])


def test_is_correct_move_does_not_mutate_board():
    board, solution = apply_setup_move(*PUZZLE_C)
    fen = board.fen()
    is_correct_move(board, chess.Move.from_uci("b1b8"), solution[0])
    assert board.fen() == fen


# --- run_puzzle -------------------------------------------------------------


def test_run_puzzle_full_solve():
    r = run_puzzle(*PUZZLE_A, get_move=scripted(["e2e8", "e1e8"]), puzzle_id="A")
    assert r.solved and r.first_move_correct
    assert r.outcome == "solved"
    assert r.n_correct_player_moves == 2 and r.n_player_moves == 2
    assert r.model_outputs == ["e2e8", "e1e8"]
    assert r.puzzle_id == "A"


def test_run_puzzle_get_move_sees_correct_positions():
    seen = []

    def get_move(board):
        seen.append(board.fen())
        return ["e2e8", "e1e8"][len(seen) - 1]

    run_puzzle(*PUZZLE_A, get_move=get_move)
    b0, sol = apply_setup_move(*PUZZLE_A)
    assert seen[0] == b0.fen()
    b0.push(sol[0])
    b0.push(sol[1])  # opponent reply from the solution line
    assert seen[1] == b0.fen()


def test_run_puzzle_first_correct_then_wrong():
    # After Qxe8+ Qxe8, White plays a legal non-mating move.
    r = run_puzzle(*PUZZLE_A, get_move=scripted(["e2e8", "h2h3"]))
    assert r.first_move_correct and not r.solved
    assert r.outcome == "wrong"
    assert r.n_correct_player_moves == 1


def test_run_puzzle_wrong_first_move():
    r = run_puzzle(*PUZZLE_A, get_move=scripted(["h2h3"]))
    assert not r.first_move_correct and not r.solved
    assert r.outcome == "wrong"
    assert r.model_outputs == ["h2h3"]


def test_run_puzzle_early_alternative_mate_is_solved():
    # Recorded line starts with h3, but Qxe8# ends the puzzle immediately.
    r = run_puzzle(*PUZZLE_B, get_move=scripted(["e2e8"]))
    assert r.solved and r.first_move_correct
    assert r.n_correct_player_moves == 1 and r.n_player_moves == 2


def test_run_puzzle_alternative_mate_on_final_move():
    r = run_puzzle(*PUZZLE_C, get_move=scripted(["b1b8"]))
    assert r.solved and r.outcome == "solved"


def test_run_puzzle_illegal_output():
    r = run_puzzle(*PUZZLE_A, get_move=scripted(["a1a8"]))
    assert not r.solved and r.outcome == "illegal"


def test_run_puzzle_unparseable_output():
    r = run_puzzle(*PUZZLE_A, get_move=scripted(["Qxe8+"]))
    assert not r.solved and r.outcome == "unparseable"


def test_run_puzzle_illegal_on_second_move():
    r = run_puzzle(*PUZZLE_A, get_move=scripted(["e2e8", "e2e8"]))  # queen no longer on e2
    assert r.first_move_correct and r.outcome == "illegal"


# --- player_positions -------------------------------------------------------


def test_player_positions_multi_move():
    from src.chess_utils import player_positions

    positions = player_positions(*PUZZLE_A)
    assert [m.uci() for _, m in positions] == ["e2e8", "e1e8"]
    b0, b1 = positions[0][0], positions[1][0]
    assert b0.turn == chess.WHITE and b1.turn == chess.WHITE
    assert b0.fen() == apply_setup_move(*PUZZLE_A)[0].fen()
    # Second position: after Qxe8+ Qxe8, the queen on e2 is gone and Black's queen is on e8.
    assert b1.piece_at(chess.E2) is None
    assert b1.piece_at(chess.E8) == chess.Piece(chess.QUEEN, chess.BLACK)
    for board, move in positions:
        assert board.is_legal(move)


def test_player_positions_mate_in_one_and_independent_boards():
    from src.chess_utils import player_positions

    positions = player_positions(*PUZZLE_C)
    assert len(positions) == 1
    board, move = positions[0]
    board.push(move)  # mutating a returned board must not affect a fresh call
    assert player_positions(*PUZZLE_C)[0][0].fen() != board.fen()
