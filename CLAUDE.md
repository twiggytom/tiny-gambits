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

## Starting a session (context is cleared often)

Read `docs/WRITEUP.md` → *Open items and next steps* (and the latest component section, e.g. §6a *Pilot plan*) to see where things stand before doing anything. The user usually commits and pushes themselves, so check `git status` before offering to.

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

## Hard-won gotchas (keep these in mind)

- **Precision:** pick bf16 vs fp16 by compute capability (`src.evaluate.dtype_for_capability`: ≥ 8.0 → bf16). **Never use `torch.cuda.is_bf16_supported()`**: it returns True on a T4 (emulated bf16), and vLLM then refuses to start. Reuse this helper in SFT/GRPO.
- **Never inherit a model's `generation_config.json`.** Qwen2.5-Instruct ships `temperature 0.7, top_p 0.8, top_k 20, repetition_penalty 1.05`, and HF `generate()` applies the repetition penalty even with `do_sample=False`. This broke HF greedy decoding on 5/20 puzzles. Eval: the HF backend uses `greedy_generation_config()`; vLLM uses explicit `SamplingParams`. **GRPO sampling and the GRPO pool filter must set every sampling param explicitly** (temperature, `top_p=1`, top_k off, `repetition_penalty=1`) and log them, including inside TRL's generation config.
- **vLLM is the only backend for reported evals.** HF is for local CPU checks and parity tests. After the fix, parity was 20/20 vs vLLM for the untrained 3B model.
- **Debug locally before burning GPU time.** The Mac has 24 GB RAM, and `Qwen/Qwen2.5-3B-Instruct` (+ 0.5B, a tiny test Qwen) is in the local HF cache. 3B in fp32 on CPU is fine for logit/argmax checks on ~20 prompts.
- **Reference numbers (dev):** untrained 3B = 0% first-move accuracy (it emits `d5`/`d7d6` regardless of board); random legal move ≈ 5.0%. A trained model should be far above 5%.

### Kaggle workflow

- Notebooks are thin runners: clone the public repo at a pinned `COMMIT` (Kaggle only sees **pushed** code), `nvidia-smi`, install pins, read secrets via `kaggle_secrets`, call `python -m src....`.
- **Run every shell command through the notebook's `run()` helper** (streams output, raises on nonzero exit), never `!cmd`. A failed `!cmd` doesn't stop the notebook, so later cells fail with misleading errors.
- Eval env pins live in `requirements-gpu.txt` (vllm 0.34.0, torch 2.14.0, transformers 5.20.2, tokenizers 0.23.3). Install them on top of the Kaggle image, not `requirements.txt`: Kaggle's pandas 2.3.3, numpy 2.1.3 and pyarrow 23 differ from the local pins and work fine. Kaggle runs Python 3.13.
- pip resolver warnings about unrelated preinstalled packages (gradio, litellm, cudf, ydata-profiling, google-adk) are expected and harmless.
- **Known conflict for training:** after the vLLM install, `datasets 4.8.5` wants `huggingface-hub<2.0` but gets 2.4.0. TRL needs `datasets`, so the SFT/GRPO notebooks need their own resolved install and pins (don't reuse the eval env blindly).
- Results come back as a zip that the user downloads from Kaggle's Output panel (`/kaggle/working/`) and hands over as a local path. Unzip it into `results/{run_name}/{split}/`. Each run's `run_info.json` must show the expected GPU and dtype (T4 → float16).

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
configs/              # one config per run (data.yaml = data pipeline, eval.yaml = eval harness)
requirements-gpu.txt  # pinned GPU eval env (Kaggle), from the Step 0 pip freeze
notebooks/            # thin Colab / Kaggle runners (kaggle_eval.ipynb = vLLM eval on one T4)
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

Kaggle eval: open `notebooks/kaggle_eval.ipynb`, set `MODEL` / `RUN_NAME` / `COMMIT` in the first cell, then Run All (GPU T4 x2, Internet on).

_Training commands to be added as those components are built._
