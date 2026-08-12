"""RAW / COMPLIANT worker -- the blaze of glory: raw (a,b,p), reduction learned in-network.

Trains the recurrent_raw two-phase scratchpad from scratch on tiers 1-3 with RAW operands
(a, b sampled UNREDUCED up to each tier's operand_bits -- NO a%p/b%p). The model learns to
reduce a mod p (phase 1) then multiply over raw b (phase 2). This is the problem the
organizer actually intends; the reduced-operand line (v3-v6) is out of spec. Held-out prime
split; operand-magnitude curriculum grows a,b from 1 to 20 digits so the reduction is
bootstrapped from the a<p trivial case outward.

Sequence: WA=20, SEQ_LEN=730 (shorter than the v5/v6 line). Verified arithmetically exact.
Launch via run_recurrent_raw_auto.py (chunked, memory-safe). The remaining wall is rollout
DRIFT over the two phases -- flip SCHED_SAMP=True for the on-policy hardening pass once the
operation is learned (train the shared cell on its own visited states).
"""
import sys
from pathlib import Path

# ----------------------------- CONFIG -----------------------------
BANDS     = ["tier1", "tier2", "tier3"]     # WA=20 covers raw operands up to 64 bits
D_MODEL   = 384
HEADS     = 8
LAYERS    = 8
BATCH     = 32          # seq 730 -> ~8-10GB at batch 32
STEPS     = 100000
MINUTES   = 40
LR        = 3e-4        # from scratch, cosine to 0
WARMUP    = 2000
CURRICULUM = True
CURR_FRAC  = 0.30       # grow RAW operand magnitude 1 -> 20 digits (reduction curriculum)
RAW_OPERANDS = True     # <-- the compliant switch: unreduced a, b
SCHED_SAMP = False      # flip True for the on-policy DRIFT-hardening pass (after it learns)
SS_MAX     = 0.25
SS_RAMP    = 0.50
EMPTY_CACHE_EVERY = 1
PRINT_SECS = 60
EVAL_SECS  = 2400
EVAL_N     = 100
EVAL_SUB   = 8
EVAL_VAL_ONLY = True
EVAL_EVERY = 1000000
DEVICE  = "auto"
MODEL_DIR = "models/recurrent_raw"
OUT       = "models/recurrent_raw/weights.pt"
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
        *(["--raw-operands"] if RAW_OPERANDS else []),
        *(["--sched-samp", "--ss-max", str(SS_MAX), "--ss-ramp", str(SS_RAMP)] if SCHED_SAMP else []),
        "--device", DEVICE, "--out", out, "--resume", "--init", out]
if CURRICULUM:
    argv += ["--curriculum", "--curr-frac", str(CURR_FRAC)]
if MINUTES:
    argv += ["--minutes", str(MINUTES)]
sys.argv = argv
train_recurrent.main()
