# CLAUDE.md

Research project comparing post-training methods (SFT vs. SFT + GRPO with Stockfish reward vs. SFT + GRPO with binary reward) for solving Lichess chess puzzles with `Qwen2.5-3B-Instruct`.

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
- **Moves are UCI** (`e2e4`, `e7e8q`). Any checkmating move counts as correct on a puzzle's final move.
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
src/
  data.py             # loading, quality filters, dedup, splitting by game
  prompts.py          # the single fixed prompt template
  chess_utils.py      # setup move, UCI parsing, legality, solution checking
  stockfish.py        # engine wrapper, node-limited eval, caching
  rewards.py          # Stockfish (dense) and binary (sparse) rewards
  train_sft.py
  train_grpo.py
  evaluate.py         # interactive eval harness + metrics
  analysis.py         # bootstrap CIs, McNemar, plots
configs/              # one config per run
notebooks/            # thin Colab / Kaggle runners
data/splits/          # saved PuzzleId lists per split
tests/
```

## Commands

_To be filled in as the project is built (setup, tests, training, eval)._
