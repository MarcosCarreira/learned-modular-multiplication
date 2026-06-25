"""Auto-chunked v4-SHIP driver — the H>90=4 submission run.  PyCharm ▶ and leave it.

Runs run_recurrent_v4_ship.py as short fresh subprocesses (memory reclaimed each
chunk), resuming seamlessly until the finishing anneal completes. Watch
models/recurrent_v4_ship/weights.pt.metrics.csv: in no-holdout mode tier_val == the
FIT on all primes, so it reads higher than the holdout run — what matters is the
official-pipeline check afterward. Keep this the only MPS job.
"""
import subprocess
import sys
import time
from pathlib import Path

import torch

# ----------------------------- CONFIG -----------------------------
STEP_TARGET = 60000    # must match STEPS in run_recurrent_v4_ship.py
MAX_CHUNKS  = 24
PAUSE_S     = 4
# ------------------------------------------------------------------

REPO   = Path(__file__).resolve().parent
WORKER = REPO / "run_recurrent_v4_ship.py"
CKPT        = REPO / "models" / "recurrent_v4_ship" / "weights.pt"
CKPT_LATEST = REPO / "models" / "recurrent_v4_ship" / "weights.pt.latest"


def ckpt_step() -> int:
    for p in (CKPT_LATEST, CKPT):
        if p.exists():
            try:
                return int(torch.load(p, map_location="cpu", weights_only=True).get("step", 0))
            except Exception as e:  # noqa: BLE001
                print(f"[auto] could not read {p.name} step ({e})")
    return 0


prev_step, stuck = -1, 0
for chunk in range(1, MAX_CHUNKS + 1):
    s = ckpt_step()
    if s >= STEP_TARGET:
        print(f"[auto] checkpoint at step {s} >= target {STEP_TARGET} — done.", flush=True)
        break
    if s == prev_step:
        stuck += 1
        if stuck >= 2:
            print(f"[auto] no progress across 2 chunks (stuck at {s}); aborting.", flush=True)
            break
    else:
        stuck = 0
    prev_step = s
    print(f"\n[auto] ===== chunk {chunk}/{MAX_CHUNKS} : resuming from step {s} =====", flush=True)
    t0 = time.time()
    r = subprocess.run([sys.executable, str(WORKER)])
    print(f"[auto] chunk {chunk} exited code {r.returncode} after {(time.time()-t0)/60:.0f} min; "
          f"now at step {ckpt_step()}.", flush=True)
    time.sleep(PAUSE_S)
else:
    print(f"[auto] hit MAX_CHUNKS={MAX_CHUNKS}; re-run to continue.", flush=True)

print(f"[auto] FINAL step: {ckpt_step()}. Then: run_check_submission.py on "
      "models/recurrent_v4_ship to confirm H>90 + OA on the official pipeline.")
