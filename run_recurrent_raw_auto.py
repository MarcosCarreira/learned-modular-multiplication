"""Auto-chunked RAW / COMPLIANT driver.  Run and leave it.

Runs run_recurrent_raw.py as short FRESH subprocesses (reclaim MPS memory, resume from the
latest checkpoint each chunk). From scratch -- no warm-start seed. Watch
models/recurrent_raw/weights.pt.metrics.csv: tierN_val on RAW operands. This is the
compliant problem (a,b unreduced); clearing any tier here means the model learned the
reduction in-network, unlike the v3-v6 line. Stops at STEP_TARGET; re-run to continue.
Keep this the ONLY MPS job running.
"""
import subprocess
import sys
import time
from pathlib import Path

import torch

STEP_TARGET = 100000
MAX_CHUNKS  = 400
PAUSE_S     = 4

REPO   = Path(__file__).resolve().parent
WORKER = REPO / "run_recurrent_raw.py"
CKPT        = REPO / "models" / "recurrent_raw" / "weights.pt"
CKPT_LATEST = REPO / "models" / "recurrent_raw" / "weights.pt.latest"


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
            print(f"[auto] no step progress across 2 chunks (stuck at {s}); aborting.", flush=True)
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
    print(f"[auto] hit MAX_CHUNKS={MAX_CHUNKS}; stopping. Re-run to continue.", flush=True)

print(f"[auto] FINAL checkpoint step: {ckpt_step()}", flush=True)
print("[auto] Read models/recurrent_raw/weights.pt.metrics.csv — RAW-operand tierN_val is the number.")
