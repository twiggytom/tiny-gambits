import dataclasses
import json

import chess
import pandas as pd
import pytest

from src.chess_utils import PuzzleState, is_correct_move, player_positions, run_puzzle
from src.evaluate import compute_metrics, dtype_for_capability, evaluate, load_results, parse_args, save_results

PUZZLES = {
    # id: (FEN, Moves, band, themes)
    "A": ("4r1k1/3q1ppp/8/8/8/8/4QPPP/4R1K1 b - - 0 1", "d7d8 e2e8 d8e8 e1e8", 0, ["mateIn2", "backRankMate"]),
    "B": ("4r1k1/5ppp/8/8/q7/8/4QPPP/4R1K1 b - - 0 1", "a4a5 h2h3 a5a6 e2e8", 1, ["mateIn2"]),
    "C": ("6k1/6pp/8/8/8/8/5PPP/RR4K1 b - - 0 1", "g8h8 a1a8", 1, ["mateIn1"]),
    # Real Lichess puzzle 00008: 3 player moves.
    "D": ("r6k/pp2r2p/4Rp1Q/3p4/8/1N1P2R1/PqP2bPP/7K b - - 0 24", "f2g3 e6e7 b2b1 b3c1 b1c1 h6c1", 7, ["fork"]),
}


def _df(ids):
    return pd.DataFrame(
        [(i, *PUZZLES[i]) for i in ids], columns=["PuzzleId", "FEN", "Moves", "band", "Themes"]
    ).assign(Rating=1500)


def _key(fen):
    return " ".join(fen.split()[:4])


# Map every player position to (puzzle id, ply, solution move) so a fake backend that only
# sees the prompt can answer according to a per-puzzle policy.
POSITIONS = {
    _key(b.fen()): (pid, ply, m)
    for pid, (fen, moves, _, _) in PUZZLES.items()
    for ply, (b, m) in enumerate(player_positions(fen, moves))
}


def wrong_move(board, solution):
    return next(m for m in board.legal_moves if not is_correct_move(board, m, solution)).uci()


POLICIES = {
    "oracle": lambda board, ply, sol: sol.uci(),
    "wrong_first": lambda board, ply, sol: wrong_move(board, sol),
    "wrong_second": lambda board, ply, sol: sol.uci() if ply == 0 else wrong_move(board, sol),
    "illegal": lambda board, ply, sol: "a3a8",  # no piece on a3 in any test position
    "junk": lambda board, ply, sol: "The best move is Qxe8+",
    "padded": lambda board, ply, sol: f"  {sol.uci().upper()}\n",
}


def answer(board, assignment):
    pid, ply, sol = POSITIONS[_key(board.fen())]
    return POLICIES[assignment[pid]](board, ply, sol)


class FakeBackend:
    """Answers from the FEN in the prompt; records batch sizes and which puzzles were prompted."""

    def __init__(self, assignment):
        self.assignment, self.batches, self.prompted = assignment, [], []

    def generate(self, batch):
        self.batches.append(len(batch))
        outs = []
        for messages in batch:
            assert [m["role"] for m in messages] == ["system", "user"]
            fen = messages[1]["content"].splitlines()[0].removeprefix("Position (FEN): ")
            board = chess.Board(fen)
            self.prompted.append(POSITIONS[_key(fen)][:2])
            outs.append(answer(board, self.assignment))
        return outs


ASSIGNMENTS = [
    {"A": "oracle", "B": "oracle", "C": "oracle", "D": "oracle"},
    {"A": "wrong_second", "B": "wrong_first", "C": "illegal", "D": "junk"},
    {"A": "padded", "B": "junk", "C": "wrong_first", "D": "wrong_second"},
]


# --- batched protocol -------------------------------------------------------


@pytest.mark.parametrize("assignment", ASSIGNMENTS)
@pytest.mark.parametrize("batch_size", [1, 2, 16])
def test_batched_matches_sequential(assignment, batch_size):
    df = _df(list(PUZZLES))
    batched = evaluate(df, FakeBackend(assignment), batch_size=batch_size)
    sequential = [
        run_puzzle(fen, moves, lambda b: answer(b, assignment), puzzle_id=pid)
        for pid, fen, moves in zip(df.PuzzleId, df.FEN, df.Moves)
    ]
    assert batched == sequential


def test_only_active_puzzles_prompted_and_batch_size_respected():
    # A fails at ply 0; D solves all 3 player moves.
    assignment = {"A": "wrong_first", "B": "oracle", "C": "oracle", "D": "oracle"}
    backend = FakeBackend(assignment)
    evaluate(_df(list(PUZZLES)), backend, batch_size=3)
    assert max(backend.batches) <= 3
    # A stops after its wrong first move; B and D follow their full lines; C is mate in 1.
    assert sorted(backend.prompted) == sorted(
        [("A", 0), ("B", 0), ("B", 1), ("C", 0), ("D", 0), ("D", 1), ("D", 2)]
    )
    assert len(backend.batches) == 4  # round 1: 4 prompts -> 3 + 1; round 2: B, D; round 3: D


def test_move_records():
    df = _df(["A"])
    [res] = evaluate(df, FakeBackend({"A": "wrong_second"}), batch_size=4)
    positions = player_positions(*PUZZLES["A"][:2])
    assert [m["ply"] for m in res.moves] == [0, 1]
    assert [m["fen"] for m in res.moves] == [b.fen() for b, _ in positions]
    assert res.moves[0] == {
        "ply": 0, "fen": positions[0][0].fen(), "output": "e2e8", "status": "legal", "uci": "e2e8", "correct": True,
    }
    assert res.moves[1]["status"] == "legal" and res.moves[1]["correct"] is False


def test_move_records_illegal_and_unparseable():
    [ill] = evaluate(_df(["C"]), FakeBackend({"C": "illegal"}), batch_size=1)
    assert ill.moves == [{"ply": 0, "fen": ill.moves[0]["fen"], "output": "a3a8", "status": "illegal", "uci": "a3a8", "correct": False}]
    [junk] = evaluate(_df(["C"]), FakeBackend({"C": "junk"}), batch_size=1)
    assert junk.moves[0]["status"] == "unparseable" and junk.moves[0]["uci"] is None


def test_puzzle_state_guards():
    s = PuzzleState(*PUZZLES["C"][:2])
    with pytest.raises(RuntimeError):
        s.result()
    s.step("a1a8")
    assert s.done
    with pytest.raises(RuntimeError):
        s.step("a1a8")


# --- metrics ----------------------------------------------------------------


def test_compute_metrics_exact():
    # A: correct then wrong (2 moves), B: junk (1), C: illegal (1), D: solved (3 moves). 7 model moves.
    assignment = {"A": "wrong_second", "B": "junk", "C": "illegal", "D": "oracle"}
    df = _df(list(PUZZLES))
    results = evaluate(df, FakeBackend(assignment), batch_size=8)
    m = compute_metrics(results, df, themes=["mateIn2", "mateIn1", "fork", "pin"])

    assert m["n_puzzles"] == 4 and m["n_model_moves"] == 7
    assert m["first_move_acc"] == 0.5  # A, D
    assert m["full_solution_acc"] == 0.25  # D
    assert m["illegal_rate"] == pytest.approx(1 / 7)
    assert m["unparseable_rate"] == pytest.approx(1 / 7)
    assert m["first_move_illegal_rate"] == 0.25
    assert m["first_move_unparseable_rate"] == 0.25
    assert m["outcomes"] == {"illegal": 1, "solved": 1, "unparseable": 1, "wrong": 1}
    assert m["by_band"] == {
        0: {"n": 1, "first_move_acc": 1.0, "full_solution_acc": 0.0},
        1: {"n": 2, "first_move_acc": 0.0, "full_solution_acc": 0.0},
        7: {"n": 1, "first_move_acc": 1.0, "full_solution_acc": 1.0},
    }
    assert m["by_theme"]["mateIn2"] == {"n": 2, "first_move_acc": 0.5, "full_solution_acc": 0.0}
    assert m["by_theme"]["fork"] == {"n": 1, "first_move_acc": 1.0, "full_solution_acc": 1.0}
    assert m["by_theme"]["pin"] == {"n": 0, "first_move_acc": None, "full_solution_acc": None}
    json.dumps(m)  # serializable


# --- saving / CLI -----------------------------------------------------------


def test_save_load_round_trip(tmp_path):
    df = _df(list(PUZZLES))
    results = evaluate(df, FakeBackend(ASSIGNMENTS[1]), batch_size=8)
    metrics = compute_metrics(results, df, themes=["fork"])
    run_info = {"run_name": "t", "model": "fake", "seed": 0}
    save_results(results, df, metrics, run_info, tmp_path / "out")
    puzzles, m2, info2 = load_results(tmp_path / "out")
    assert len(puzzles) == 4 and list(puzzles.PuzzleId) == list(df.PuzzleId)
    assert m2 == json.loads(json.dumps(metrics)) and info2 == run_info
    row = puzzles.set_index("PuzzleId").loc["A"]
    assert row["moves"] == dataclasses.asdict(results[0])["moves"]
    assert row["Themes"] == ["mateIn2", "backRankMate"] and row["band"] == 0


def test_test_split_requires_final():
    base = ["--model", "m", "--run-name", "r", "--seed", "0"]
    with pytest.raises(SystemExit):
        parse_args(base + ["--split", "test"])
    assert parse_args(base + ["--split", "test", "--final"]).final
    assert parse_args(base + ["--split", "dev"]).split == "dev"


@pytest.mark.parametrize(
    "capability, dtype",
    [((7, 5), "float16"), ((7, 0), "float16"), ((8, 0), "bfloat16"), ((8, 9), "bfloat16"), ((9, 0), "bfloat16")],
)
def test_dtype_for_capability(capability, dtype):
    # T4 = 7.5 must get fp16 (vLLM refuses bf16 below 8.0); L4 = 8.9 and A100 = 8.0 get bf16.
    assert dtype_for_capability(capability) == dtype


def test_greedy_generation_config_drops_model_sampling_defaults():
    # Qwen2.5-Instruct's generation_config.json: inheriting repetition_penalty=1.05 made HF "greedy" differ
    # from vLLM on 5/20 dev puzzles (docs/WRITEUP.md §6a).
    transformers = pytest.importorskip("transformers")
    from src.evaluate import greedy_generation_config

    qwen = transformers.GenerationConfig(
        do_sample=True, temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.05,
        eos_token_id=[151645, 151643], bos_token_id=151643, pad_token_id=151643,
    )
    g = greedy_generation_config(qwen, max_tokens=16)
    assert g.do_sample is False and g.max_new_tokens == 16
    assert g.repetition_penalty == 1.0 and g.num_beams == 1 and g.no_repeat_ngram_size == 0
    assert g.eos_token_id == [151645, 151643] and g.pad_token_id == 151643
    # Sampling knobs are not carried over (irrelevant under do_sample=False, but must not leak in).
    for k in ["temperature", "top_p", "top_k"]:
        assert getattr(g, k) != getattr(qwen, k), k


def test_hf_backend_is_pure_argmax():
    """HF backend output == a manual argmax loop, even if the model's config asks for a repetition penalty."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from huggingface_hub import try_to_load_from_cache
    from src.evaluate import HFBackend
    from src.prompts import build_messages

    tiny = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
    if not isinstance(try_to_load_from_cache(tiny, "config.json"), str):
        pytest.skip("tiny Qwen test model not in local HF cache")
    b = HFBackend(tiny, max_tokens=8, seed=0)
    assert b.model.generation_config.repetition_penalty == 1.0
    msgs = [build_messages(chess.Board()), build_messages(chess.Board("8/8/8/4k3/8/8/4K3/7R w - - 0 1"))]
    got = b.generate(msgs)
    eos = set(b.model.generation_config.eos_token_id if isinstance(b.model.generation_config.eos_token_id, list)
              else [b.model.generation_config.eos_token_id])
    for m, out in zip(msgs, got):
        ids = b.tokenizer(b.tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True), return_tensors="pt").input_ids
        new = []
        with torch.no_grad():
            for _ in range(8):
                nxt = b.model(ids).logits[0, -1].argmax().item()
                if nxt in eos:
                    break
                new.append(nxt); ids = torch.cat([ids, torch.tensor([[nxt]])], 1)
        assert out == b.tokenizer.decode(new, skip_special_tokens=True)
