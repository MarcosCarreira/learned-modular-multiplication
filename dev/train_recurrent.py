"""Train the autoregressive scratchpad model (recurrent_v1).  DEV-ONLY.

Teacher-forced next-token training on the scratchpad
  BOS a SEP b SEP p SEP  P SEP k SEP R EOS  (P=a*b, k=P//p, R=P%p),
loss on the completion only. Eval = autoregressive generation, extract R,
exact-match on TRAIN and HELD-OUT primes. The tier-3 held-out number is the
prize: does learning the algorithm generalize to unseen primes?

Examples:
    python dev/train_recurrent.py --device cpu --bands tier1 tier2 --steps 300 --eval-every 100
    python dev/train_recurrent.py --bands tier1 tier2 tier3 --minutes 40
"""
from __future__ import annotations

import argparse, csv, importlib.util, math, random, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "dev"))
import data  # noqa: E402

SUB_DIR = REPO / "models" / "recurrent_v1"


def load_arch(model_dir=None):
    d = Path(model_dir) if model_dir else SUB_DIR
    spec = importlib.util.spec_from_file_location("rec_model", d / "model.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def to_t(x, dev):
    return torch.tensor(x, dtype=torch.long, device=dev)


def make_batch(train_pools, batch, rng, arch, band_w=None):
    bands = [b for b, ps in train_pools.items() if ps]
    weights = [band_w.get(b, 1.0) for b in bands] if band_w else None
    seqs = np.empty((batch, arch.SEQ_LEN), np.int64)
    for i in range(batch):
        band = rng.choices(bands, weights=weights, k=1)[0]
        ps = train_pools[band]
        p = ps[rng.randrange(len(ps))]
        a = rng.randrange(p); b = rng.randrange(p)
        seqs[i] = arch.build_full(a, b, p)
    return seqs


def make_batch_curriculum(train_pools, batch, rng, arch, cap, keep_full, band_w=None):
    """Length-curriculum batch: operands capped to < cap (short products) for an
    anti-forgetting fraction; else full range [0,p). Bootstraps long multiplication
    by growing `cap` over training. `band_w` reweights how often each tier is drawn
    (e.g. oversample tier3 so capacity isn't spent over-perfecting an easier tier)."""
    bands = [b for b, ps in train_pools.items() if ps]
    weights = [band_w.get(b, 1.0) for b in bands] if band_w else None
    seqs = np.empty((batch, arch.SEQ_LEN), np.int64)
    for i in range(batch):
        band = rng.choices(bands, weights=weights, k=1)[0]
        ps = train_pools[band]
        p = ps[rng.randrange(len(ps))]
        hi = p if rng.random() < keep_full else min(p, cap)
        a = rng.randrange(hi); b = rng.randrange(hi)
        seqs[i] = arch.build_full(a, b, p)
    return seqs


@torch.no_grad()
def exact_match(model, primes, dev, arch, n=2000, seed=7, sub=16):
    if not primes:
        return float("nan")
    rng = random.Random(seed)
    prompts = np.empty((n, arch.PROMPT_LEN), np.int64)
    # ans holds the true product mod p, which at tier 5+ reaches ~2^64 — beyond
    # int64's range. Keep it as Python ints (unbounded) to avoid OverflowError;
    # prompts stays int64 (it only holds small digit/marker tokens 0..13).
    ans = [0] * n
    for i in range(n):
        p = primes[rng.randrange(len(primes))]
        a = rng.randrange(p); b = rng.randrange(p)
        prompts[i] = arch.build_prompt(a, b, p); ans[i] = (a * b) % p
    # Generate in small sub-batches with a cache release between, so the eval's
    # KV-cache peak is bounded by `sub` sequences (not all `n` at once).
    correct = 0
    for s in range(0, n, sub):
        pc = prompts[s:s + sub]
        gen = model.generate(to_t(pc, dev), arch.GEN_LEN).cpu().numpy()
        for j in range(len(pc)):
            R = arch.from_field(gen[j][arch.R_OFF:arch.R_OFF + arch.WIDTH])
            correct += int(R == ans[s + j])
        if dev.type == "mps" and hasattr(torch, "mps"):
            torch.mps.empty_cache()
    return correct / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bands", nargs="+", default=["tier1", "tier2", "tier3"])
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--minutes", type=float, default=None)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=0,
                    help="linear LR warmup steps (stabilizes big models)")
    ap.add_argument("--clip", type=float, default=1.0,
                    help="gradient-norm clip (0 = off); prevents loss blow-ups")
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--dim-ff", type=int, default=1024)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--holdout", action="store_true", default=True)
    ap.add_argument("--no-holdout", dest="holdout", action="store_false")
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--eval-secs", type=float, default=0.0,
                    help="run the (expensive) accuracy eval + save every N seconds (0 = step-based only)")
    ap.add_argument("--print-secs", type=float, default=0.0,
                    help="cheap heartbeat (step/loss/curr/throughput, NO eval) every N seconds")
    ap.add_argument("--eval-n", type=int, default=2000)
    ap.add_argument("--eval-sub", type=int, default=16,
                    help="sequences generated at once during eval (lower = less MPS "
                         "memory at long seq lengths; e.g. 4 for WIDTH-20 v5)")
    ap.add_argument("--eval-val-only", action="store_true",
                    help="skip the train-pool eval; val (held-out) is the ranked metric "
                         "and held-out~=train is established -> ~halves eval time")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model-dir", default=str(SUB_DIR),
                    help="submission dir whose model.py defines the architecture")
    ap.add_argument("--out", default=str(SUB_DIR / "weights.pt"))
    ap.add_argument("--resume", action="store_true", help="warm-start from --init (default --out)")
    ap.add_argument("--init", default=None)
    ap.add_argument("--curriculum", action="store_true",
                    help="length curriculum: grow operand magnitude over training")
    ap.add_argument("--curr-frac", type=float, default=0.5,
                    help="fraction of steps over which the cap ramps to full")
    ap.add_argument("--curr-keep-full", type=float, default=0.3,
                    help="fraction of examples always at full range (anti-forget)")
    ap.add_argument("--band-weights", nargs="+", type=float, default=None,
                    help="per-band sampling weights in --bands order (e.g. 0.5 1.0 2.5 to "
                         "oversample tier3); default uniform")
    ap.add_argument("--empty-cache-every", type=int, default=0,
                    help="on MPS, release the cached allocator every N training steps "
                         "(bounds memory growth on long runs; 0 = off)")
    args = ap.parse_args()
    if args.d_model % args.heads != 0:
        raise SystemExit(f"--d-model ({args.d_model}) must be divisible by --heads ({args.heads}).")
    band_w = None
    if args.band_weights:
        if len(args.band_weights) != len(args.bands):
            raise SystemExit(f"--band-weights ({len(args.band_weights)}) must match "
                             f"--bands ({len(args.bands)}).")
        band_w = dict(zip(args.bands, args.band_weights))
        print(f"band sampling weights: {band_w}")

    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    dev = pick_device(args.device)
    arch = load_arch(args.model_dir)
    init_path = Path(args.init) if args.init else Path(args.out)
    resume_key = (-1, -1.0)
    base_step = 0          # global steps already completed (true continuation)
    opt_state = None       # optimizer moments saved alongside the best checkpoint
    if args.resume and init_path.exists():
        ck = torch.load(init_path, map_location=dev, weights_only=True)
        model = arch.ScratchpadDecoder(**ck["config"]).to(dev)
        # Continuation (init == out): the BEST file seeds best_key (so we never
        # overwrite a better submission ckpt), but we CONTINUE TRAINING from the
        # LATEST snapshot if one exists — so chunked restarts move forward instead
        # of snapping back to the best (which can sit behind the live step when the
        # 2-vs-3 tradeoff makes recent steps not beat it). Fall back to best if no
        # latest snapshot yet. Warm-start into a NEW dir (init != out): take weights
        # from the given file with a fresh schedule/best.
        if init_path.resolve() == Path(args.out).resolve():
            resume_key = (ck.get("h90", -1), ck.get("frontier_val", -1.0))
            latest_path = Path(str(init_path) + ".latest")
            if latest_path.exists():
                lk = torch.load(latest_path, map_location=dev, weights_only=True)
                model.load_state_dict(lk["state_dict"])
                base_step = int(lk.get("step", 0))
                opt_path, src = Path(str(init_path) + ".latest.opt"), "LATEST"
            else:
                model.load_state_dict(ck["state_dict"])
                base_step = int(ck.get("step", 0))
                opt_path, src = Path(str(init_path) + ".opt"), "best"
            if opt_path.exists():
                try:
                    opt_state = torch.load(opt_path, map_location=dev, weights_only=True)
                except Exception as e:  # noqa: BLE001
                    print(f"WARN: could not read optimizer sidecar ({e}); fresh moments")
            print(f"RESUMED {src} state @ step {base_step} "
                  f"(best on disk: H>=90 {resume_key[0]}, frontierVal {resume_key[1]:.3f})")
        else:
            model.load_state_dict(ck["state_dict"])
            print(f"WARM-STARTED from {init_path} (fresh schedule/best for new dir)")
    else:
        if args.resume:
            print(f"--resume set but {init_path} not found; starting fresh.")
        model = arch.ScratchpadDecoder(d_model=args.d_model, nhead=args.heads,
                                       num_layers=args.layers, dim_ff=args.dim_ff,
                                       seq_len=arch.SEQ_LEN).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    pools = data.build_pools(args.bands, holdout=args.holdout, seed=args.seed)
    print(f"device={dev}  params={n_params/1e6:.2f}M  seq_len={arch.SEQ_LEN}  "
          f"mode={'HOLD-OUT' if args.holdout else 'SHIPPING'}")
    print("prime pools (train vs val):"); print(pools.summary())

    rng = random.Random(args.seed + 1)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    if opt_state is not None:
        try:
            opt.load_state_dict(opt_state)
            print(f"restored optimizer moments (continuing from step {base_step})")
        except Exception as e:  # noqa: BLE001
            print(f"WARN: optimizer state not restored ({e}); fresh moments")

    def lr_factor(step):                       # linear warmup -> cosine decay
        if args.warmup and step < args.warmup:
            return step / max(1, args.warmup)
        prog = (step - args.warmup) / max(1, args.steps - args.warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))
    # LR is applied manually from the GLOBAL step each iteration (in the loop), not
    # via a LambdaLR whose internal counter resets to 0 on resume — driving it off
    # the global step is what lets the schedule continue seamlessly across sessions.
    loss_fn = nn.CrossEntropyLoss(reduction="none")
    # loss only on completion tokens: target position t (in seq[1:]) is a
    # completion token when t >= PROMPT_LEN-1
    comp = (torch.arange(arch.SEQ_LEN - 1, device=dev) >= arch.PROMPT_LEN - 1).float()

    best_key, t0, run = resume_key, time.time(), 0.0
    last_eval, last_print, steps_since = time.time(), time.time(), 0
    if base_step >= args.steps:
        print(f"NOTE: already at/past --steps ({base_step} >= {args.steps}); "
              f"raise STEPS to train further.")
    for step in range(base_step + 1, args.steps + 1):
        model.train()
        for g in opt.param_groups:             # manual LR from the global step
            g["lr"] = args.lr * lr_factor(step)
        if args.curriculum:
            frac = min(1.0, step / max(1, int(args.steps * args.curr_frac)))
            d = 1 + int(round(frac * (arch.WIDTH - 1)))      # 1..WIDTH digits
            cap = 10 ** d
            seq = to_t(make_batch_curriculum(pools.train, args.batch, rng, arch,
                                             cap, args.curr_keep_full, band_w), dev)
        else:
            seq = to_t(make_batch(pools.train, args.batch, rng, arch, band_w), dev)   # (B,L)
        logits = model(seq[:, :-1])                                       # (B,L-1,V)
        tgt = seq[:, 1:]
        per = loss_fn(logits.reshape(-1, arch.VOCAB), tgt.reshape(-1)).reshape(tgt.shape)
        loss = (per * comp).sum() / (comp.sum() * seq.shape[0])
        opt.zero_grad(); loss.backward()
        if args.clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        opt.step()
        run += loss.item(); steps_since += 1
        if (args.empty_cache_every and dev.type == "mps" and hasattr(torch, "mps")
                and step % args.empty_cache_every == 0):
            torch.mps.empty_cache()   # release accumulated MPS blocks DURING training

        now = time.time()
        capped = args.minutes and (now - t0) / 60.0 >= args.minutes
        eval_due = (step % args.eval_every == 0
                    or (args.eval_secs and now - last_eval >= args.eval_secs)
                    or step == args.steps or capped)
        print_due = (not eval_due) and args.print_secs and (now - last_print >= args.print_secs)
        if eval_due or print_due:
            loss_avg = run / max(1, steps_since)
            spm = steps_since / max(1e-9, now - last_print) * 60.0     # steps/min
            curr = ""
            if args.curriculum:
                d = 1 + int(round(min(1.0, step / max(1, int(args.steps * args.curr_frac))) * (arch.WIDTH - 1)))
                curr = f"[curr {d}d]"
            run, steps_since, last_print = 0.0, 0, now
            if eval_due:
                model.eval()
                # Insurance: persist a resumable LATEST snapshot BEFORE the (expensive,
                # occasionally fragile) generation eval, so a crash in eval never loses
                # the chunk's training progress. Overwritten with real metrics after eval.
                torch.save({"state_dict": model.state_dict(), "config": model.config,
                            "bands": args.bands, "h90": -1, "frontier_val": -1.0,
                            "step": step}, str(args.out) + ".latest")
                torch.save(opt.state_dict(), str(args.out) + ".latest.opt")
                if dev.type == "mps" and hasattr(torch, "mps"):
                    torch.mps.empty_cache()      # drop the training workspace BEFORE eval gen
                parts, tier_val, tier_tr = [], {}, {}
                for band in args.bands:
                    va = exact_match(model, pools.val[band], dev, arch, n=args.eval_n, sub=args.eval_sub)
                    tr = (float("nan") if args.eval_val_only
                          else exact_match(model, pools.train[band], dev, arch, n=args.eval_n, sub=args.eval_sub))
                    tier_val[band] = va; tier_tr[band] = tr
                    parts.append(f"{band}[val {va:.3f}]" if args.eval_val_only
                                 else f"{band}[tr {tr:.3f} | val {va:.3f}]")
                if dev.type == "mps" and hasattr(torch, "mps"):
                    torch.mps.empty_cache()      # release MPS blocks after eval gen
                h90, frontier = data.frontier_key(tier_val)
                print(f"step {step:>6}  loss {loss_avg:.4f}  {curr}  " + "  ".join(parts)
                      + f"  H>=90 {h90}  frontierVal {frontier:.3f}", flush=True)
                if (h90, frontier) > best_key:
                    best_key = (h90, frontier)
                    torch.save({"state_dict": model.state_dict(), "config": model.config,
                                "bands": args.bands, "h90": h90, "frontier_val": frontier,
                                "step": step}, args.out)
                    torch.save(opt.state_dict(), Path(str(args.out) + ".opt"))
                    print(f"    ^ saved best (H>=90 {h90}, frontierVal {frontier:.3f}, "
                          f"step {step}) -> {args.out}")
                # Always-on LATEST snapshot (separate from the best/submission ckpt):
                # lets chunked restarts continue forward from the true latest state.
                torch.save({"state_dict": model.state_dict(), "config": model.config,
                            "bands": args.bands, "h90": h90, "frontier_val": frontier,
                            "step": step}, str(args.out) + ".latest")
                torch.save(opt.state_dict(), str(args.out) + ".latest.opt")
                # Append a metrics row so progress / plateau is observable from file
                # (accumulates across chunked restarts). Excluded from HF upload.
                mlog = Path(str(args.out) + ".metrics.csv")
                new_log = not mlog.exists()
                with mlog.open("a", newline="") as fh:
                    w = csv.writer(fh)
                    if new_log:
                        cols = ["time", "step", "loss"]
                        for b in args.bands:
                            cols += [f"{b}_tr", f"{b}_val"]
                        w.writerow(cols + ["h90", "frontier"])
                    row = [time.strftime("%Y-%m-%dT%H:%M:%S"), step, f"{loss_avg:.5f}"]
                    for b in args.bands:
                        row += [f"{tier_tr[b]:.4f}", f"{tier_val[b]:.4f}"]
                    w.writerow(row + [h90, f"{frontier:.4f}"])
                last_eval = now
            else:
                print(f"step {step:>6}  loss {loss_avg:.4f}  {curr}  "
                      f"{spm:.0f} steps/min  elapsed {(now-t0)/60:.0f}m", flush=True)
        if capped:
            print(f"time cap {args.minutes} min reached at step {step}"); break

    print(f"done in {(time.time()-t0)/60:.1f} min; best (H>=90 {best_key[0]}, frontierVal {best_key[1]:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
