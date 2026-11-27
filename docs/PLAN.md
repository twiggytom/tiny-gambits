# PLAN.md — Post-Training Methods for Chess Puzzle Solving

Research question: how do different post-training methods (SFT, GRPO with dense vs. sparse reward) affect a small LLM's ability to solve chess puzzles?

> **For Claude Code:** This file is the source of truth for the project. Everything under "Frozen Design Choices" is fixed — do not change these values without asking. Items under "Open Items" are set during the pilot and then frozen here.

---

## Frozen Design Choices

### Model & Training

| Setting | Value |
|---|---|
| Base model | `Qwen2.5-3B-Instruct` |
| Precision | 4-bit base weights (QLoRA) via Unsloth; compute in fp16 on T4, bf16 on L4 |
| LoRA | rank 16, alpha 32, dropout 0, bias none, all linear layers (q/k/v/o/gate/up/down) — identical for SFT and GRPO |
| Board representation | FEN |
| Move notation | UCI (e.g. `e2e4`, `e7e8q`) |
| Output | Moves only, no reasoning |
| Output cap | 16 tokens (output hitting the cap without a valid move = unparseable) |
| Output parsing | Strip whitespace, lowercase; entire output must be exactly one UCI move, otherwise unparseable. A UCI-shaped output that isn't a constructible move (same from/to square, e.g. `e4e4`) counts as **illegal** |

### SFT

- 1 epoch over the SFT split
- **One example per player move** (multi-move puzzles give several examples): 150,000 puzzles → 369,697 examples (nested `sft_50k`: 50,000 → 123,301). The target is the bare UCI move. Built by `src/sft_data.py` into `data/processed/`
- **Leakage rule (all training data, SFT and GRPO):** drop any training position whose first 4 FEN fields match any player-move position of a dev or test puzzle
- Run on Kaggle, single NVIDIA T4, fp16
- **Optimizer settings (all SFT runs):** effective batch 32 (16 per device × grad-accum 2), cosine schedule, 3% warmup, AdamW 8-bit (betas 0.9/0.999, eps 1e-8), weight decay 0, grad-norm clip 1.0, max sequence length 256, length grouping on; loss on the assistant move + `<|im_end|>` only. Steps: sft_5k 387, sft_15k 1,158, sft_50k 3,854, sft (150k) 11,554 (HF Trainer counts the trailing partial accumulation as a step). Seed 0. LR **2e-4** (set by the pilot sweep, Open Items). Code: `src/train_sft.py`, config `configs/sft.yaml`, runner `notebooks/kaggle_sft.ipynb`
- Every run pushes its **LoRA adapter only** to a private HF Hub repo `{user}/tiny-gambits-sft-{5k|15k|50k|150k}-lr{lr}`, plus the latest resumable checkpoint as `last-checkpoint/` every 100 steps. For eval, the adapter is merged into the 16-bit base locally
- Adapter is **merged** into the base model after training → this merged model is the starting point for all GRPO runs (only the final 150k model's merged copy is pushed to the Hub)

### GRPO

| Setting | Value |
|---|---|
| Starting point | Merged SFT model (same checkpoint for every GRPO run) |
| Steps | 2,000 (final value confirmed by pilot) |
| Prompts per step | 8 |
| Completions per prompt | 16 |
| Sampling temperature | ~1.0 |
| Checkpoints | Every 250 steps |
| Seeds | 2 per condition (seeds `1` and `2`) |
| Compute | Colab Pro, NVIDIA L4 (24 GB), bf16 |

- **No checkpoint selection.** Every GRPO run uses the final checkpoint at the same step count.
- Intermediate checkpoints are used only for learning-curve plots.
- Training format: one position per prompt, one move per completion. Multi-move puzzles are split into one training position per player move.

### Rewards

| Reward | Definition |
|---|---|
| Stockfish (dense) | Change in win probability after the move, normalized to [-1, 1] |
| Binary (sparse) | 1 if the move is the solution move **or any checkmating move**, 0 otherwise |
| Illegal / unparseable | Same fixed penalty under both rewards |

### Stockfish

- Run locally via `python-chess` (no external APIs)
- Fixed Stockfish version, same binary on Colab and Kaggle (record version number here)
- Search limited by **fixed node count** (value set during pilot)
- Win probability via `score.wdl(model="sf").expectation()`
- **Cache evaluations** keyed by (position, move) — many of the 16 completions repeat

### Compute & Hardware

| Platform | Hardware | Used for |
|---|---|---|
| Kaggle (free) | 2× NVIDIA T4 (16 GB each) | SFT, GRPO pool difficulty filtering, all evaluation |
| Colab Pro | 1× NVIDIA L4 (24 GB) | GRPO runs only |
| CPU (both platforms) | — | Stockfish |
| Hugging Face Hub | — | Storing LoRA adapters and moving them between platforms |

**Hardware rules:**

- **GRPO always runs on the L4.** It fits 3B QLoRA GRPO with 16-token outputs comfortably, supports bf16, and costs far fewer compute units per hour than an A100. All 4 GRPO runs use the same GPU type.
- **A100 is a fallback only** (if the L4 is unavailable for an extended period). If used, record it for that run.
- **SFT uses one T4.** Unsloth trains on a single GPU; the second T4 is free for running filtering or eval jobs in parallel.
- **T4 has no bf16** → SFT and Kaggle inference use fp16. GRPO on the L4 uses bf16.
- **Evaluation:** run one eval job per T4 in parallel (e.g., two models at once).
- **No TPUs.** Unsloth, bitsandbytes, and vLLM require NVIDIA GPUs (CUDA).
- Run `!nvidia-smi` at the start of every session and log the GPU name with each run.

---

## Data

Source: Lichess puzzle database (`Lichess/chess-puzzles` on Hugging Face, CC0), pinned to revision `461725966ea46898ec6481372a6326de3281bfbd` (6,157,341 puzzles). Pipeline: `src/data.py`, config `configs/data.yaml`, seed `0`.

### Columns Used

| Column | Role |
|---|---|
| `PuzzleId` | Tracing results; saved ID lists per split |
| `FEN` | Input. Position **before** the opponent's setup move |
| `Moves` | Target. UCI solution line; **the first move is the opponent's setup move** |
| `Rating` | Stratifying splits; rating-band analysis |
| `Themes` | Per-theme analysis (test set only) |
| `GameId` | Game ID (part before the first `/` or `#`) used to split by game. In this revision every game has exactly one puzzle |
| `RatingDeviation`, `NbPlays`, `Popularity` | Quality filtering only, then dropped |

### Preprocessing (in order)

1. **Quality filter:** RatingDeviation < 100, NbPlays > 300, Popularity > 0 (3,176,257 remain)
2. **Deduplicate** on the first four FEN fields (piece placement, side to move, castling, en passant), ignoring move counters, for **both** the raw `FEN` and the prompt position (after the setup move). Keep the most-played puzzle (3,158,250 remain)
3. **Split by game:** all puzzles from the same game go into the same split. Splits are drawn per rating band from one seeded shuffle, in the order test → dev → GRPO candidates → SFT
4. **Apply the setup move:** the prompt position is `FEN` after playing the first move in `Moves`; the model's first move is the second move in `Moves`

### Splits

All splits are disjoint. Save the list of PuzzleIds for each split to disk.

| Split | Size | Sampling |
|---|---|---|
| SFT | 150,000 | 18,750 per rating band; nested subsets `sft_50k` (= the original 50k split, identical ids), `sft_15k`, `sft_5k`. Raised from 50,000 on 2026-11-26: test, dev, GRPO candidates and the nested subsets are unchanged |
| GRPO pool | 8,000 | Filtered by difficulty (see below) |
| Dev | 500 | Evenly across bands (63 × 4 lower, 62 × 4 upper); pilot, sanity checks, tuning |
| Test | 2,000 | 8 rating bands × 250; **touched only for final evaluation** |

**GRPO pool filtering:** draw 40,000 candidates (5,000 per rating band; `data/splits/grpo_candidates.txt`), sample the SFT model 16 times per position, keep positions it solves **sometimes but not always**, then sample 8,000 puzzles. The same filtered pool is used for both reward conditions. Record the pool's rating distribution.

> Limitation: this filter removes always-failed puzzles, which is where dense reward should have its biggest advantage, so results may understate the Stockfish reward's benefit. (Alternative considered: keep always-failed puzzles.)

**Theme coverage:** pick ~8–10 themes to analyze (e.g. fork, pin, mateIn1, mateIn2, discoveredAttack, sacrifice). Each must appear in ≥100 test puzzles; otherwise drop the theme (the test set is not topped up). `backRankMate` (38) and `hangingPiece` (83) are dropped. Test-set counts for the rest: fork 256, mateIn2 213, sacrifice 197, mateIn1 167, pin 129, discoveredAttack 128; other eligible tactical themes: kingsideAttack 164, advancedPawn 136, defensiveMove 126, attraction 112.

---

## Independent Variable

**Post-training technique** (1 SFT run + 4 GRPO runs → 5 evaluated models):

| # | Condition | Runs |
|---|---|---|
| 1 | Base + SFT | 1 |
| 2 | Base + SFT + GRPO, Stockfish reward (dense) | 2 seeds |
| 3 | Base + SFT + GRPO, binary reward (sparse) | 2 seeds |

The base instruct model (zero-shot) is not evaluated.

## Controlled Variables

Held constant across all conditions: model, LoRA settings, data splits, GRPO steps, prompts and completions per step, output cap, Stockfish version and node limit, prompt template, decoding settings.

---

## Evaluation

- **Protocol: interactive.** Model plays a move → opponent's reply from the solution is applied → repeat until the puzzle ends. Any checkmating move is accepted at any point in the puzzle and ends it as solved.
- **Decoding:** greedy (temperature 0)
- **Prompt template (v1, `src/prompts.py`):** one fixed template for all evaluated models (SFT and all GRPO runs), applied with Qwen's chat template
  - System: `You are a chess engine. Reply with a single move in UCI notation.`
  - User: `Position (FEN): {fen}` / `You are playing {White|Black}. Find the best move.` / `Answer with only the move in UCI notation (e.g. e2e4, e7e8q).` (three lines)
- **Compute:** Kaggle, 2× NVIDIA T4 (one eval job per GPU), vLLM, fp16
- **Harness:** `src/evaluate.py` (config `configs/eval.yaml`). Batched interactive loop (one generate call per ply across all active puzzles). It saves `puzzles.jsonl` (per puzzle, with every model move), `metrics.json` and `run_info.json` to `results/{run_name}/{split}/`, and **these are committed**. `results/raw/` is ignored scratch space
- **Test-set guard:** `--split test` is refused unless `--final` is passed

### Dependent Variables

1. First-move accuracy
2. Full-solution accuracy (every player move correct)
3. Illegal / unparseable rate: **per model move** (illegal or unparseable outputs ÷ all model moves attempted); the first-move-only rate is also reported
4. Accuracy by rating band
5. Accuracy by tactical theme
6. Stockfish win-probability evaluation of the model's moves (average centipawn loss as secondary). Computed **post hoc** from the saved per-move records (position + move), so models aren't re-run
7. GRPO learning curves (reward and dev accuracy every 250 steps)

### Statistics

- Bootstrap confidence intervals on all metrics
- McNemar's test for pairwise comparisons between models (same test puzzles)

---

## Hypotheses

Write these down **before** running the main experiment. Draft:

- H1: SFT + GRPO outperforms SFT alone.
- H2: Stockfish (dense) reward outperforms binary (sparse) reward, with the largest gap at higher rating bands.
- H3: Dense reward produces faster-rising learning curves than sparse reward.

---

## Pilot Phase (before main runs)

0. Kaggle bring-up: run the eval harness (vLLM, T4, fp16) on dev with the untrained `Qwen2.5-3B-Instruct` as an environment smoke test only (results in ignored `results/raw/`, not a reported condition)
1. SFT learning-rate sweep at **3B** on `sft_5k`: LR ∈ {1e-4, 2e-4, 5e-4}, picked by dev first-move accuracy (full-solution accuracy as tiebreak)
2. SFT data-scaling curve at **3B**: 5k / 15k / 50k / 150k at the chosen LR, evaluated on dev. The **150k** run **is** the final SFT model (condition 1); no separate main SFT run. (Changed 2026-11-26 from 50k: 50k reached only 16.2% dev first-move / 5.6% full-solution with the curve still rising, and a weak SFT model risks an under-filled, easy-skewed GRPO pool)
3. SFT sanity checks on dev
4. 20-step GRPO timing test at 3B on the L4 (check compute units used per hour in Colab's resource monitor)
5. 500-step GRPO run → set final step count
6. Set the open items below, then freeze them in this file

## Open Items

- [x] SFT learning rate: **2e-4** (frozen 2026-11-25). The sft_5k sweep (1e-4 / 2e-4 / 5e-4) could not separate the LRs on dev: first-move 0.2% / 0.6% / 0.8%, all below the random-legal baseline, McNemar p ≥ 0.375 for every pair, full-solution 0 for all. The first-move rule was therefore uninformative; 2e-4 was chosen for the lowest training loss (1.31 vs 1.36 / 1.37, mean of the last 10 logged) and the most varied outputs (49 distinct first moves vs 10 / 25). Set in `configs/sft.yaml` `lr`
- [ ] GRPO learning rate
- [ ] Stockfish node limit
- [ ] Stockfish version
- [x] Rating band boundaries: `<1000, 1000–1200, 1200–1400, 1400–1600, 1600–1800, 1800–2000, 2000–2200, ≥2200` (8 bands)
- [ ] Illegal-move penalty value
- [ ] Final list of analyzed themes
- [x] Prompt template text: v1, see Evaluation → Prompt template

---

## Stack

- `chess` (python-chess): board logic, legality, UCI parsing, Stockfish control
- `unsloth`, `trl`, `transformers`, `peft`, `datasets`: training
- `vllm`: GRPO sampling and evaluation
- `pandas`, `scipy`, `statsmodels`: data prep and statistics
- Weights & Biases: training curves + dev metrics for every run (project `tiny-gambits`)
- Hugging Face Hub: private repos for adapters and resumable checkpoints
- Secrets (`HF_TOKEN`, `WANDB_API_KEY`) are read from environment variables only; notebooks copy them from Colab/Kaggle secrets into the environment, never hardcode them

## Rules for Development

- Never use the test set during development, tuning, or debugging — use dev.
- Never change a frozen value without updating this file.
- Push every adapter to the Hugging Face Hub immediately after training.
- Checkpoint often; Colab sessions disconnect.
- Disconnect Colab runtimes when idle; idle GPUs still use compute units.
