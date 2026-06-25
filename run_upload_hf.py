"""Push a submission dir to HuggingFace and print its commit hash.  PyCharm ▶.

Prereqs:
  * a HuggingFace account, and that account declared to the competition
    organizers as one of your team's submission identities (do this first).
  * a token with WRITE access to create/upload. Either set TOKEN below, or run
    `huggingface-cli login` once and leave TOKEN = None.

Uploads ONLY the files a submission needs (manifest.json, model.py, weights),
keeping junk like __pycache__ out of the repo. Prints repo_id + commit_hash —
that pair is what you give the organizers.
"""
import os
import sys
from pathlib import Path

# The "xet" upload backend can hang mid-file on the FIRST push to a new repo.
# Force the classic LFS multipart path, which is slower-listed but far more robust.
os.environ["HF_HUB_DISABLE_XET"] = "1"

from huggingface_hub import HfApi

# ----------------------------- CONFIG -----------------------------
SUBMISSION = "models/recurrent_v4"
REPO_ID    = "MCarreira/recurrent-v4"   # #6 candidate: H>90=4, OA 0.400. (docker "timeout" was x86-on-ARM emulation; Playground ran #5 in 0.65s -> v4 is fine. Confirm v4 on the Playground after upload.)
PRIVATE    = True            # keep private while testing; make public before the deadline
TOKEN      = None            # None -> use your cached `huggingface-cli login`
# ------------------------------------------------------------------

REPO = Path(__file__).resolve().parent
folder = (REPO / SUBMISSION).resolve()
if "YOUR_HF_USERNAME" in REPO_ID:
    sys.exit("Edit REPO_ID to your HuggingFace username/repo first.")

api = HfApi(token=TOKEN)
api.create_repo(REPO_ID, repo_type="model", private=PRIVATE, exist_ok=True)
api.upload_folder(
    folder_path=str(folder),
    repo_id=REPO_ID,
    repo_type="model",
    # Explicit allowlist: ONLY the three submission files. Keeps the trainer's
    # sidecars (weights.pt.opt / .latest / .latest.opt / .metrics.csv) and any
    # backup *.pt out of the repo.
    allow_patterns=["manifest.json", "model.py", "weights.pt"],
)
sha = api.repo_info(REPO_ID).sha
print("\n" + "=" * 60)
print(f"  repo_id     : {REPO_ID}")
print(f"  commit_hash : {sha}")
print(f"  private     : {PRIVATE}")
print("=" * 60)
print("Test it the way organizers will:")
print(f"  modchallenge evaluate-hf {REPO_ID} {sha} --token hf_xxx --total 110")
print("Then make the repo PUBLIC and submit repo_id + commit_hash before the deadline.")
