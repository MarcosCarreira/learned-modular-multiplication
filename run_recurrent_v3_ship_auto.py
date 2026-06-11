"""Auto-chunked v3 ship — the fix for the MPS memory blowup.  PyCharm ▶ and leave it.

Long MPS training runs accumulate memory until the Mac chokes (PyCharm hit ~75GB).
The cure is to run training in SHORT chunks, each in a FRESH python subprocess: when
a chunk exits, the OS reclaims all its memory, and the next chunk resumes exactly
where it stopped (the trainer saves the global step + optimizer sidecar, so this is
identical to one long run — just without the creep).

This driver launches run_recurrent_v3_ship.py (MINUTES=75 per chunk) over and over,
reading the checkpoint's step between chunks, until the ship reaches STEP_TARGET.
You still see the live training logs; it stops itself when done. Re-run anytime to
continue. Keep this the ONLY MPS job running.
"""
import subprocess
import sys
import time
from pathlib import Path

import torch

# ----------------------------- CONFIG -----------------------------
STEP_TARGET = 40000   # must match STEPS in run_recurrent_v3_ship.py (anneal horizon)
MAX_CHUNKS  = 14      # safety ceiling (~75 min each => ~17h max); re-run to go further
PAUSE_S     = 4       # let the OS settle / free memory between chunks
# ------------------------------------------------------------------

REPO   = Path(__file__).resolve().parent
WORKER = REPO / "run_recurrent_v3_ship.py"
CKPT   = REPO / "models" / "recurrent_v3_ship" / "weights.pt"          # best (submission)
CKPT_LATEST = REPO / "models" / "recurrent_v3_ship" / "weights.pt.latest"  # latest (progress)


def ckpt_step() -> int:
    # Track the LATEST snapshot for progress (the best can lag behind the live step
    # when later steps don't beat it); fall back to best, then 0.
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
        print(f"[auto] checkpoint at step {s} >= target {STEP_TARGET} — ship is done.", flush=True)
        break
    if s == prev_step:
        stuck += 1
        if stuck >= 2:
            print(f"[auto] no step progress across 2 chunks (stuck at {s}); aborting. "
                  f"Check the worker output above for an error.", flush=True)
            break
    else:
        stuck = 0
    prev_step = s

    print(f"\n[auto] ===== chunk {chunk}/{MAX_CHUNKS} : resuming from step {s} "
          f"(fresh process, memory reclaimed) =====", flush=True)
    t0 = time.time()
    r = subprocess.run([sys.executable, str(WORKER)])
    print(f"[auto] chunk {chunk} exited code {r.returncode} after {(time.time()-t0)/60:.0f} min; "
          f"now at step {ckpt_step()}.", flush=True)
    time.sleep(PAUSE_S)
else:
    print(f"[auto] hit MAX_CHUNKS={MAX_CHUNKS}; stopping. Re-run to continue.", flush=True)

print(f"[auto] FINAL checkpoint step: {ckpt_step()}  ({CKPT})", flush=True)
print("[auto] When tier-3 has settled in the low-0.9s, point run_check_submission.py "
      "at models/recurrent_v3_ship and run it to confirm H>90 + OA on the official pipeline.")
