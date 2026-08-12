"""Auto-chunked RAW tier-3 push.  Run and leave it.

On first launch, seeds models/recurrent_raw_t3/weights.pt from the finished recurrent_raw
checkpoint (verbatim, same arch, step reset to 0 for a fresh schedule), then runs
run_recurrent_raw_t3.py as short FRESH subprocesses. Launch only AFTER the recurrent_raw run
has finished (so the seed is its best). Watch models/recurrent_raw_t3/weights.pt.metrics.csv
tier3_val -- the target is >= 0.90 (compliant H>90=3). Stops at STEP_TARGET; re-run to continue.
"""
import subprocess
import sys
import time
from pathlib import Path

import torch

STEP_TARGET = 50000
MAX_CHUNKS  = 80
PAUSE_S     = 4

REPO   = Path(__file__).resolve().parent
WORKER = REPO / "run_recurrent_raw_t3.py"
SRC         = REPO / "models" / "recurrent_raw" / "weights.pt"        # finished recurrent_raw best
CKPT        = REPO / "models" / "recurrent_raw_t3" / "weights.pt"
CKPT_LATEST = REPO / "models" / "recurrent_raw_t3" / "weights.pt.latest"


def ckpt_step() -> int:
    for p in (CKPT_LATEST, CKPT):
        if p.exists():
            try:
                return int(torch.load(p, map_location="cpu", weights_only=True).get("step", 0))
            except Exception as e:  # noqa: BLE001
                print(f"[auto] could not read {p.name} step ({e})")
    return 0


# 0) Seed from the finished recurrent_raw checkpoint if we have nothing yet.
if not CKPT.exists() and not CKPT_LATEST.exists():
    if not SRC.exists():
        print(f"[auto] need recurrent_raw's checkpoint at {SRC} — let that run finish first. Aborting.", flush=True)
        raise SystemExit(1)
    print("[auto] seeding recurrent_raw_t3 from the finished recurrent_raw checkpoint (step 0) ...", flush=True)
    ck = torch.load(SRC, map_location="cpu", weights_only=True)
    ck["step"] = 0; ck["h90"] = -1; ck["frontier_val"] = -1.0
    CKPT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ck, CKPT)
    print(f"[auto] seeded -> {CKPT}", flush=True)

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
print("[auto] Read models/recurrent_raw_t3/weights.pt.metrics.csv — tier3_val is the target (>=0.90 = compliant H>90=3).")
