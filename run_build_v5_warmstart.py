"""Build the recurrent_v5 warm-start checkpoint from the trained v4 model.  PyCharm ▶.

v5 is v4 widened (WIDTH 10 -> 20): the per-step a*single-digit operation is identical,
only the sequence is longer (SEQ_LEN ~1384 vs 394). So we seed v5 from v4's trained
weights instead of training cold:

  * the whole transformer body (encoder.*), token embeddings, final LayerNorm and
    output head are IDENTICAL in shape -> copied verbatim (this is the learned
    "multiply a digit, add, reduce mod p" circuit, the valuable part);
  * the Abacus place embeddings are digit-position-aligned, so v4's rows transfer
    directly (digit position k means the same thing); v5's new higher digit positions
    are seeded from the nearest learned row, and the no-place row is copied;
  * the position embeddings cannot be copied by index (layouts differ across widths).
    v4's pos_emb is std~1 but is a LEARNED positional manifold (not noise — confirmed
    by measuring it), and the warm encoder depends on its structure. So we INTERPOLATE
    v4's trained code to the longer sequence (the standard context-extension trick),
    handing the encoder a smooth, learned, correctly-scaled positional signal from
    step 0 instead of a structureless random one. (An earlier "flat init at v4 scale"
    was a no-op: v4's scale is already ~1, so the issue was missing STRUCTURE, not
    magnitude.)

Writes models/recurrent_v5/weights.pt at step 0. The auto driver runs this for you
if the seed is missing; you can also run it by hand once. Requires the trained
models/recurrent_v4/weights.pt to exist.
"""
import importlib.util
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "dev"))

V4_CKPT = REPO / "models" / "recurrent_v4" / "weights.pt"
V5_DIR = REPO / "models" / "recurrent_v5"
OUT = V5_DIR / "weights.pt"


def load_arch(d):
    spec = importlib.util.spec_from_file_location("rec_model_v5", Path(d) / "model.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    if not V4_CKPT.exists():
        raise SystemExit(f"need trained v4 weights at {V4_CKPT} — train/upload v4 first.")
    v5 = load_arch(V5_DIR)
    ck = torch.load(V4_CKPT, map_location="cpu", weights_only=True)
    v4_sd, v4_cfg = ck["state_dict"], ck["config"]
    print(f"v4 config: {v4_cfg}")
    print(f"v5 geometry: SEQ_LEN={v5.SEQ_LEN}  NB={v5.NB}  WIDTH={v5.WIDTH}  NPLACE={v5.NPLACE}")

    model = v5.ScratchpadDecoder(d_model=v4_cfg["d_model"], nhead=v4_cfg["nhead"],
                                 num_layers=v4_cfg["num_layers"], dim_ff=v4_cfg["dim_ff"],
                                 seq_len=v5.SEQ_LEN)
    new_sd = model.state_dict()
    copied, notes = 0, []
    for k, v in new_sd.items():
        if k not in v4_sd:
            notes.append(f"fresh   : {k} (absent in v4)")
            continue
        src = v4_sd[k]
        if src.shape == v.shape:
            new_sd[k] = src.clone()
            copied += 1
        elif k == "place_emb.weight":
            v4_rows, v5_rows = src.shape[0], v.shape[0]
            nd = v4_rows - 1                          # v4 digit-position rows 0..nd-1
            v.data[:nd] = src[:nd]                    # aligned digit positions (warm)
            # New higher digit positions (nd..v5_rows-2): seed from the nearest
            # LEARNED row (row nd-1) so they start warm-scaled, not N(0,1) noise.
            v.data[nd:v5_rows - 1] = src[nd - 1].unsqueeze(0)
            v.data[v5_rows - 1] = src[v4_rows - 1]    # no-place -> no-place
            new_sd[k] = v
            notes.append(f"warm    : {k} ({nd} digit rows copied; {v5_rows - 1 - nd} seeded from "
                         f"nearest learned row; no-place copied)")
        elif k == "pos_emb.weight":
            # v4's pos_emb is std~1 but is a LEARNED positional manifold, not noise —
            # the warm encoder depends on its structure. Layouts differ across widths
            # so we can't copy by index, but we can INTERPOLATE v4's trained code to
            # the longer sequence (the standard context-extension trick). This hands
            # the encoder a smooth, learned, correctly-scaled positional signal from
            # step 0 instead of random — the real jumpstart (scale was never the issue).
            import torch.nn.functional as Fx
            src_t = src.t().unsqueeze(0)                       # (1, d, L4=394)
            interp = Fx.interpolate(src_t, size=v.shape[0],    # -> (1, d, L5=1384)
                                    mode="linear", align_corners=True)
            v.data.copy_(interp.squeeze(0).t())               # (L5, d)
            new_sd[k] = v
            notes.append(f"interp  : {k} (v4 {src.shape[0]} -> {v.shape[0]} positions, "
                         f"learned manifold stretched; std {float(src.std()):.3f}->{float(v.std()):.3f})")
        else:
            notes.append(f"fresh   : {k} (shape {tuple(src.shape)} -> {tuple(v.shape)})")

    model.load_state_dict(new_sd)
    print(f"copied verbatim: {copied} tensors")
    for n in notes:
        print(f"  {n}")

    V5_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": model.config,
                "bands": ["tier1", "tier2", "tier3", "tier4", "tier5"],
                "h90": -1, "frontier_val": -1.0, "step": 0}, OUT)
    # clear any stale sidecars so the trainer starts this seed cleanly
    for suf in (".latest", ".latest.opt", ".opt", ".metrics.csv"):
        sp = Path(str(OUT) + suf)
        if sp.exists():
            sp.unlink()
    print(f"\nwrote warm-start seed -> {OUT} (step 0).")
    print("Next: run run_recurrent_v5_auto.py")


if __name__ == "__main__":
    main()
