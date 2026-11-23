"""Board logic for chess puzzles: setup move, UCI parsing, legality, solution checking.

All board logic goes through python-chess. Moves are compared as ``chess.Move``
objects, never as strings.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

import chess

_UCI_RE = re.compile(r"[a-h][1-8][a-h][1-8][qrbn]?")


def apply_setup_move(fen: str, moves: str) -> tuple[chess.Board, list[chess.Move]]:
    """Play the opponent's setup move and return the prompt position.

    A Lichess puzzle's ``FEN`` is the position *before* the opponent's move, and
    the first move in ``Moves`` is that setup move. Returns the board after the
    setup move plus the remaining solution moves ``[player, opp, player, ...]``.

    Raises ``ValueError`` if the FEN is invalid, ``moves`` has fewer than two
    moves, or any move in the line is illegal.
    """
    board = chess.Board(fen)
    ucis = moves.split()
    if len(ucis) < 2:
        raise ValueError(f"Puzzle line needs a setup move and a solution move: {moves!r}")

    parsed: list[chess.Move] = []
    probe = board.copy(stack=False)
    for uci in ucis:
        move = chess.Move.from_uci(uci)
        if not probe.is_legal(move):
            raise ValueError(f"Illegal move {uci!r} in puzzle line {moves!r} (FEN {fen!r})")
        probe.push(move)
        parsed.append(move)

    board.push(parsed[0])
    return board, parsed[1:]


def player_positions(fen: str, moves: str) -> list[tuple[chess.Board, chess.Move]]:
    """Every position where the player is to move, paired with its solution move.

    Follows the solution line (opponent replies included), so a puzzle with
    ``k`` player moves yields ``k`` positions. Boards are independent copies.
    """
    board, solution = apply_setup_move(fen, moves)
    out = []
    for i, move in enumerate(solution):
        if i % 2 == 0:
            out.append((board.copy(stack=False), move))
        board.push(move)
    return out


def parse_uci(text: str) -> chess.Move | None:
    """Parse raw model output as a single UCI move (strict-ish).

    Strips surrounding whitespace and lowercases; the entire remaining output
    must be exactly one UCI move (e.g. ``e2e4``, ``e7e8q``). Anything else
    (extra tokens, punctuation, SAN, null move ``0000``) returns ``None``.
    """
    cleaned = text.strip().lower()
    if not _UCI_RE.fullmatch(cleaned):
        return None
    return chess.Move.from_uci(cleaned)


class MoveStatus(Enum):
    LEGAL = "legal"
    ILLEGAL = "illegal"
    UNPARSEABLE = "unparseable"


def check_move(text: str, board: chess.Board) -> tuple[MoveStatus, chess.Move | None]:
    """Parse ``text`` and check legality on ``board``.

    Returns ``(UNPARSEABLE, None)``, ``(ILLEGAL, move)`` or ``(LEGAL, move)``.
    Legal king-takes-rook castling (e.g. ``e1h1``) is normalized to standard
    UCI (``e1g1``) so it compares equal to the Lichess solution move.
    """
    move = parse_uci(text)
    if move is None:
        return MoveStatus.UNPARSEABLE, None
    if not board.is_legal(move):
        return MoveStatus.ILLEGAL, move
    return MoveStatus.LEGAL, board.parse_uci(move.uci())


def is_checkmate_move(board: chess.Board, move: chess.Move) -> bool:
    """True if the (legal) ``move`` checkmates on ``board``. Does not mutate ``board``."""
    after = board.copy(stack=False)
    after.push(move)
    return after.is_checkmate()


def is_correct_move(board: chess.Board, move: chess.Move, solution_move: chess.Move) -> bool:
    """A legal move is correct if it is the solution move or any checkmating move."""
    return move == solution_move or is_checkmate_move(board, move)


@dataclass
class PuzzleResult:
    puzzle_id: str | None
    first_move_correct: bool
    solved: bool
    n_correct_player_moves: int
    n_player_moves: int
    outcome: str  # "solved" | "wrong" | "illegal" | "unparseable"
    model_outputs: list[str] = field(default_factory=list)


def run_puzzle(
    fen: str,
    moves: str,
    get_move: Callable[[chess.Board], str],
    puzzle_id: str | None = None,
) -> PuzzleResult:
    """Run the interactive evaluation protocol on one puzzle.

    The setup move is applied, then for each player turn ``get_move(board)`` is
    called with a copy of the current position and its raw output is checked.
    The puzzle stops on unparseable, illegal or wrong output. Any checkmating
    move ends the puzzle as solved, at any point. Otherwise the opponent's reply
    from the solution line is played and the loop continues.
    """
    board, solution = apply_setup_move(fen, moves)
    n_player_moves = (len(solution) + 1) // 2
    outputs: list[str] = []
    n_correct = 0

    def result(solved: bool, outcome: str) -> PuzzleResult:
        return PuzzleResult(
            puzzle_id=puzzle_id,
            first_move_correct=n_correct >= 1,
            solved=solved,
            n_correct_player_moves=n_correct,
            n_player_moves=n_player_moves,
            outcome=outcome,
            model_outputs=outputs,
        )

    for i in range(0, len(solution), 2):
        text = get_move(board.copy())
        outputs.append(text)
        status, move = check_move(text, board)
        if status is not MoveStatus.LEGAL:
            return result(False, status.value)
        assert move is not None
        if not is_correct_move(board, move, solution[i]):
            return result(False, "wrong")
        n_correct += 1
        if is_checkmate_move(board, move):
            return result(True, "solved")
        board.push(move)
        if i + 1 < len(solution):
            board.push(solution[i + 1])

    return result(True, "solved")
