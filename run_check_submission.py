"""Pre-submission gauntlet.  PyCharm ▶, no args.

Runs every check the official pipeline applies, locally, against a submission
dir, and prints a PASS/FAIL checklist. Forces CPU by default to mirror the
official sandbox (where the model may run on CPU) and to exercise the
determinism path that matters there.

Green across the board here == ready to push to HuggingFace.
"""
import json
import os
import sys
import time
from pathlib import Path

# ----------------------------- CONFIG -----------------------------
SUBMISSION = "models/recurrent_v3_ship"   # dir to check
TOTAL      = 1100                       # full benchmark (100/tier)
FORCE_CPU  = True                       # mirror the official sandbox
TIME_BUDGET_S = 300                     # official: 5 min / 1100 problems
SIZE_LIMIT_GB = 20
# ------------------------------------------------------------------

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))
if FORCE_CPU:
    os.environ["MODCH_FORCE_CPU"] = "1"

from modchallenge.config import EvalConfig, PublicBenchmarkConfig          # noqa: E402
from modchallenge.evaluation.pipeline import evaluate_local                 # noqa: E402
from modchallenge.security.static_check import check_submission            # noqa: E402
from modchallenge.interface.submission_schema import SubmissionManifest    # noqa: E402

sub = (REPO / SUBMISSION).resolve()
results = []  # (label, ok, detail)


def add(label, ok, detail=""):
    results.append((label, ok, detail))


# 1) required files present
need = ["manifest.json", "model.py"]
have_w = any((sub / f).exists() for f in
             ("weights.pt", "weights.safetensors", "model.safetensors"))
add("required files (manifest.json, model.py)",
    all((sub / f).exists() for f in need), str(sub))
add("weight file present", have_w, "train the model first if this fails")

# 2) manifest schema + required non-empty descriptions
try:
    m = json.loads((sub / "manifest.json").read_text())
    SubmissionManifest.model_validate(m)
    ok = bool(m.get("model_description", "").strip()) and \
        bool(m.get("training_description", "").strip())
    add("manifest schema + descriptions", ok,
        f"entry={m.get('entry_class')} base={m.get('output_base')}")
except Exception as e:  # noqa: BLE001
    add("manifest schema + descriptions", False, repr(e))

# 3) static compliance scan
try:
    findings = check_submission(sub)
    add("static compliance (no banned imports / shortcuts)",
        not findings, "; ".join(f.format() for f in findings) or "clean")
except Exception as e:  # noqa: BLE001
    add("static compliance", False, repr(e))

# 4) artifact size
size = sum(f.stat().st_size for f in sub.rglob("*") if f.is_file())
add(f"artifact size < {SIZE_LIMIT_GB} GB",
    size < SIZE_LIMIT_GB * 1024**3, f"{size/1024**2:.1f} MB")

# 5) full pipeline: load -> isolation -> determinism -> inference -> score
print(f">>> running full pipeline on {SUBMISSION} "
      f"({'CPU' if FORCE_CPU else 'auto device'}, {TOTAL} problems) ...",
      flush=True)
try:
    cfg = EvalConfig(total_problems=TOTAL, timeout_seconds=1800)
    t0 = time.time()
    res = evaluate_local(sub, master_seed=PublicBenchmarkConfig().seed, config=cfg)
    wall = time.time() - t0
    add("pipeline ran end-to-end", True, "")
    add("deterministic (required for ranking)", bool(res.deterministic), "")
    hi = max((t.tier_id for t in res.tier_results
              if t.tier_id >= 1 and t.completed and t.accuracy >= 0.90), default=0)
    add("inference within ~time budget (hardware-dependent)",
        wall <= TIME_BUDGET_S, f"{wall:.1f}s local vs {TIME_BUDGET_S}s budget")
    print(f"\n    overall_accuracy = {res.overall_accuracy:.3f}   H>90 = {hi}")
    print("    per-tier:", "  ".join(
        f"{t.tier_id}:{t.accuracy:.2f}" for t in sorted(res.tier_results, key=lambda r: r.tier_id)))
except Exception as e:  # noqa: BLE001
    add("pipeline ran end-to-end", False, repr(e))

# summary
print("\n" + "=" * 62)
print("PRE-SUBMISSION CHECKLIST")
print("=" * 62)
allok = True
for label, ok, detail in results:
    allok &= ok
    print(f"  [{'PASS' if ok else 'FAIL'}]  {label}" + (f"   ({detail})" if detail else ""))
print("=" * 62)
print("ALL GREEN — ready to push to HuggingFace (run_upload_hf.py)."
      if allok else "Some checks FAILED — fix before submitting.")
