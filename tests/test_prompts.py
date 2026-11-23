import chess

from src.chess_utils import parse_uci
from src.prompts import SYSTEM_PROMPT, build_messages, sft_messages, user_prompt

START = chess.Board()


def test_user_prompt_golden():
    assert user_prompt(START) == (
        "Position (FEN): rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1\n"
        "You are playing White. Find the best move.\n"
        "Answer with only the move in UCI notation (e.g. e2e4, e7e8q)."
    )


def test_side_to_move_black():
    board = chess.Board()
    board.push_uci("e2e4")
    assert "You are playing Black." in user_prompt(board)
    assert board.fen() in user_prompt(board)


def test_message_roles_and_system():
    msgs = build_messages(START)
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert msgs[0]["content"] == SYSTEM_PROMPT


def test_sft_target_is_bare_uci_and_round_trips():
    for uci in ["e2e4", "g1f3"]:
        move = chess.Move.from_uci(uci)
        msgs = sft_messages(START, move)
        assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
        assert msgs[-1]["content"] == uci
        assert parse_uci(msgs[-1]["content"]) == move
    promo = chess.Move.from_uci("e7e8q")
    assert sft_messages(START, promo)[-1]["content"] == "e7e8q"
