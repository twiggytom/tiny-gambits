# CLAUDE.md

Research project comparing post-training methods (SFT vs. SFT + GRPO with Stockfish reward vs. SFT + GRPO with binary reward) for solving Lichess chess puzzles with `Qwen2.5-3B-Instruct`.

## How we work: component by component

This project is built **one component at a time**: data pipeline → chess utils / rewards → SFT → GRPO → eval harness → analysis. Don't one-shot the whole project or scaffold components that weren't asked for. Finish, test, and agree on the current component before starting the next.

## Log everything in `docs/WRITEUP.md`

`docs/WRITEUP.md` is the user's local-only (gitignored) notes file for the final writeup. **After finishing any decision or action, record it there before ending the turn:**

- **Decisions** (a value chosen, an open item settled, a plan change, a design choice): add a row to the *Design decision log* with the date, the decision, and the rationale.
- **Actions** (a component built, a pipeline run, a test or validation done, an experiment or training run): add or update that component's section, or *Testing and validation*, with what was done, the exact numbers and results, config/seed/versions, and anything surprising.
- Keep *Known limitations*, *Open items and next steps*, *Reproducibility*, and the "Last updated" date current.
- Write in enough detail to be lifted into the writeup directly: real counts and tables, not summaries like "it worked". Report failures and caveats too.
- If the file is missing, recreate it with the same section structure and say so.

## When to read `docs/PLAN.md`

`docs/PLAN.md` is the full experimental design and the source of truth. Read it:

- At the start of work on a **new component** (data pipeline, rewards, SFT, GRPO, eval harness, analysis)
- Before touching anything that sets an experimental value (model, LoRA, splits, steps, rewards, decoding, prompt template)
- When unsure whether a choice is already decided

For routine edits within a component you've already read the plan for, you don't need to re-read it.

**Much of the experimental design is still tentative or unspecified.** Whenever something needs specification or clarification (a value isn't in the plan, the plan is ambiguous, or a choice would affect results), ask before deciding. Don't fill gaps with your own assumptions.

## Rules that apply to every change

- **Never use the test set** during development, tuning, or debugging. Use the dev split.
- **Never change a frozen value** from `docs/PLAN.md` without asking first. If a value is changed or an open item is decided, update `docs/PLAN.md` in the same change.
- **Apply the setup move.** A puzzle's `FEN` is the position *before* the opponent's move; the first move in `Moves` must be played before prompting. The model's first move is the second move in `Moves`.
- **Moves are UCI** (`e2e4`, `e7e8q`). Any checkmating move counts as correct at any point in a puzzle (ends it as solved).
- **Everything seeded.** Any script with randomness takes a `--seed` argument.
- **Never hardcode secrets.** Read `HF_TOKEN` and `WANDB_API_KEY` from environment variables / Colab and Kaggle secrets.

## Code conventions

- Logic lives in importable Python modules under `src/`. Notebooks are thin runners for Colab/Kaggle: install, import, call, log.
- Code must run on both platforms: Kaggle (T4, fp16) and Colab (L4, bf16). Don't hardcode GPU type or precision; detect or pass as config.
- Configs (hyperparameters, paths) go in config files, not scattered in code.
- Chess logic (setup move, move parsing, legality, rewards, solution checking) must have `pytest` tests. These are the easiest places to introduce silent bugs.
- Use `python-chess` for all board logic. Don't hand-roll move parsing or legality checks.
- Save split PuzzleId lists to `data/splits/`; downstream code loads splits from these files rather than re-sampling.

## Repo structure (tentative — update as built)

```
docs/PLAN.md          # full experimental design
docs/WRITEUP.md       # local-only writeup notes (gitignored); update after each component
src/
  data.py             # loading, quality filters, dedup, splitting by game
  prompts.py          # the single fixed prompt template
  sft_data.py         # SFT examples (one per player move) + dev/test leakage filter
  chess_utils.py      # setup move, UCI parsing, legality, solution checking
  stockfish.py        # engine wrapper, node-limited eval, caching
  rewards.py          # Stockfish (dense) and binary (sparse) rewards
  train_sft.py
  train_grpo.py
  evaluate.py         # batched interactive eval harness, metrics, saved per-puzzle results (vLLM / HF backends)
  analysis.py         # bootstrap CIs, McNemar, plots
configs/              # one config per run (data.yaml = data pipeline)
notebooks/            # thin Colab / Kaggle runners
data/splits/          # saved PuzzleId lists per split
tests/
```

## Commands

```bash
pip install -r requirements.txt   # setup
python -m pytest                  # run tests (from repo root)
python -m src.data --config configs/data.yaml --seed 0   # build data/splits/ (~1 min)
python -m src.sft_data --config configs/data.yaml --split sft --seed 0   # build data/processed/sft.jsonl (also sft_15k, sft_5k)
python -m src.evaluate --config configs/eval.yaml --model <path|hub id> --backend vllm --split dev --run-name <name> --seed 0   # GPU eval; test split needs --final
# Local CPU smoke test (needs `pip install torch`; writes to ignored results/raw/):
python -m src.evaluate --model Qwen/Qwen2.5-0.5B-Instruct --backend hf --split dev --limit 20 --run-name raw/smoke --seed 0
```

_Training and eval commands to be added as those components are built._
