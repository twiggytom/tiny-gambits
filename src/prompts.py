"""The single fixed prompt template used for SFT, GRPO, and evaluation (all models).

Returns chat-message lists; training/eval scripts apply the tokenizer's chat
template. Changing any text here changes the experiment: bump PROMPT_VERSION
and record it in docs/PLAN.md.
"""

from __future__ import annotations

import chess

PROMPT_VERSION = "v1"

SYSTEM_PROMPT = "You are a chess engine. Reply with a single move in UCI notation."

USER_TEMPLATE = (
    "Position (FEN): {fen}\n"
    "You are playing {side}. Find the best move.\n"
    "Answer with only the move in UCI notation (e.g. e2e4, e7e8q)."
)


def user_prompt(board: chess.Board) -> str:
    side = "White" if board.turn == chess.WHITE else "Black"
    return USER_TEMPLATE.format(fen=board.fen(), side=side)


def build_messages(board: chess.Board) -> list[dict[str, str]]:
    """Prompt messages (system + user) for the position ``board``, player to move."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt(board)},
    ]


def sft_messages(board: chess.Board, move: chess.Move) -> list[dict[str, str]]:
    """Prompt messages plus the assistant target: the bare UCI move."""
    return build_messages(board) + [{"role": "assistant", "content": move.uci()}]
