"""V6 worker — the 20M / 14-layer tier-5 push, HOLD-OUT.

recurrent_v6 = recurrent_v5 made DEEPER (8 -> 14 layers, ~11.6M -> ~20M params, same
384 width) to raise per-step fidelity on the 20-digit multiply-reduce — the bottleneck
that pinned v5's tier 5 at ~0.80. Warm-started from v5 (8 layers copied in, 6 new layers
near-identity on top), so it starts at v5's competence and learns depth-wise refinement.

NO curriculum — the warm model already handles 20-digit operands, so we train directly on
the full operand range. Tier-5 is heavily oversampled (band weight 4.0): since the ranking
key is the *highest* tier >= 90%, the only thing that buys H>90=5 is tier 5 itself, so
capacity goes there (we do NOT try to rescue tier 2). Runs as one ~40-min chunk and
self-caps — launch via run_recurrent_v6_auto.py (seeds the warm-start, relaunches fresh
subprocesses to reclaim MPS memory, resumes seamlessly).

MEMORY: 14 layers at seq 1384 is heavy, so BATCH=8 (est. ~18GB, under the ~31GB ceiling);
EMPTY_CACHE_EVERY=1 holds it flat. THROUGHPUT: ~1.75x v5's per-step cost — a long run
(expected ~1-2 weeks to a decision), which is fine. ANNEAL: once tier-5 plateaus, shorten
STEPS to drive the cosine LR to 0 for the finishing lift (v5 went 0.72->0.83->0.87 that way).
GOAL: tier-5 val >= 0.90 -> H>90=5. Anything short is still a sharper 20-digit scaling point.
"""
import sys
from pathlib import Path

# ----------------------------- CONFIG -----------------------------
BANDS        = ["tier1", "tier2", "tier3", "tier4", "tier5"]
BAND_WEIGHTS = [0.3, 0.4, 0.7, 1.0, 2.0]   # EASED from tier5=4.0: homogeneous hard batches were
                                            #   eroding generation (exposure bias, tier5 drifted DOWN
                                            #   0.78->0.54). More tier3/4 scaffolding = diverse batches.
D_MODEL   = 384
HEADS     = 8
LAYERS    = 14          # deeper (not wider) -> ~20M; targets the mod-p reduction fidelity
BATCH     = 12          # bumped 8->12 (memory had headroom, ~18->~27GB): smoother gradients are
                        #   gentler on the warm transplant. If resident tops ~31GB, drop to 10.
STEPS     = 33000       # ANNEAL TRIGGER (was 150k): center plateaued ~0.745 at step 15k, peaks 0.86
                        #   -> drive the (already low) LR to 0 by step 33k (~18k steps) to consolidate
                        #   near the peak. Projection ~0.85-0.88, outside shot at 0.90.
MINUTES   = 40          # chunk; the auto driver relaunches fresh processes to reclaim memory
LR        = 2e-5        # LAST-SWING setting: 1e-4 degraded hard (tier5->0.40); 5e-5 recovered to
                        #   0.78 then eroded down to 0.54 (exposure bias). 2e-5 + eased tier5 mix is
                        #   the gentle final attempt. KILL-SWITCH: if tier5 CENTER isn't climbing
                        #   above ~0.75 by ~step 20k (~a day), stop — v5's 0.80 is the ceiling.
WARMUP    = 2000
CURRICULUM = False      # warm model already does 20 digits -> train full-range from the start
EMPTY_CACHE_EVERY = 1
PRINT_SECS = 60
EVAL_SECS  = 2400       # one eval per chunk (14 layers -> generation is ~1.75x slower than v5)
EVAL_N     = 50
EVAL_SUB   = 4          # generate 4 seqs at once during eval — bounds MPS memory at seq 1384
EVAL_VAL_ONLY = True    # val/held-out is the ranked metric; skip the train eval
EVAL_EVERY = 1000000    # disable step-based evals; eval on the chunk cap / time schedule
DEVICE  = "auto"
MODEL_DIR = "models/recurrent_v6"
OUT       = "models/recurrent_v6/weights.pt"
# ------------------------------------------------------------------

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "dev"))
import train_recurrent  # noqa: E402

out = str((REPO / OUT).resolve())
argv = ["train_recurrent.py", "--bands", *BANDS, "--holdout",
        "--model-dir", str((REPO / MODEL_DIR).resolve()),
        "--d-model", str(D_MODEL), "--heads", str(HEADS), "--layers", str(LAYERS),
        "--batch", str(BATCH), "--empty-cache-every", str(EMPTY_CACHE_EVERY),
        "--steps", str(STEPS), "--lr", str(LR), "--warmup", str(WARMUP),
        "--eval-secs", str(EVAL_SECS), "--eval-every", str(EVAL_EVERY),
        "--print-secs", str(PRINT_SECS), "--eval-n", str(EVAL_N), "--eval-sub", str(EVAL_SUB),
        *(["--eval-val-only"] if EVAL_VAL_ONLY else []),
        "--band-weights", *[str(w) for w in BAND_WEIGHTS],
        "--device", DEVICE, "--out", out, "--resume", "--init", out]
if CURRICULUM:
    argv += ["--curriculum", "--curr-frac", "0.30"]
if MINUTES:
    argv += ["--minutes", str(MINUTES)]
sys.argv = argv
train_recurrent.main()
