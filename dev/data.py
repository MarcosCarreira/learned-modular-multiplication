"""Data layer for the cross-prime modular-multiplication trainer.

This is DEV-ONLY code (it imports sympy for prime enumeration, which is
blocklisted inside a submission). It produces training/validation batches for
learning  a*b mod p  with the property that matters for this competition:
generalisation to primes NEVER seen in training.

Design choices that encode the strategy:

* Primes are organised into bit-BANDS that mirror the official tiers
  (tier 2 = [2^4, 2^8), tier 3 = [2^9, 2^16), ...).
* Within each band the primes are split into a TRAIN pool and a disjoint
  HELD-OUT VAL pool. The val metric therefore measures "did the model learn
  the field operation" not "did it memorise the 5 public primes" — exactly
  what the secret-seed official eval rewards.
* Operands are sampled in [0, p): the submission legally reduces a%p, b%p
  before the network runs, so the network only ever sees reduced operands.
  We mirror that here.

Everything is torch-free and returns plain Python / numpy, so it can be unit
tested without a GPU.
"""
from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, field

import numpy as np
from sympy import primerange


def frontier_key(tier_val: dict[str, float], solved_thresh: float = 0.90):
    """Leaderboard-aligned selection key from per-tier held-out val accuracies.

    Mirrors the official ranking instead of an equal-weighted mean (which a
    saturated tier1 dominates):
      primary   = highest tier index with val >= solved_thresh   (~highest_tier_above_90)
      secondary = mean val over the NOT-yet-solved tiers only     (frontier progress)
    Solved tiers drop out of the secondary term, so tier1 stops inflating it.
    Returns (highest_solved_tier:int, frontier_val:float).
    """
    def idx(t: str) -> int:
        return int("".join(ch for ch in t if ch.isdigit()) or 0)

    solved = [idx(t) for t, v in tier_val.items()
              if not math.isnan(v) and v >= solved_thresh]
    highest = max(solved, default=0)
    vals = [v for v in tier_val.values() if not math.isnan(v)]
    unsolved = [v for t, v in tier_val.items()
                if not math.isnan(v) and v < solved_thresh]
    if unsolved:
        frontier = sum(unsolved) / len(unsolved)        # progress on the frontier
    else:
        # all tiers solved: keep rewarding within-tier improvement (total
        # accuracy), offset by 1.0 so it stays above any unsolved state.
        frontier = 1.0 + (sum(vals) / len(vals) if vals else 0.0)
    return (highest, frontier)

# --- output representation -------------------------------------------------
# Base-10, fixed width, MSB-first, zero-padded. WIDTH=5 covers every answer
# for p < 10^5, i.e. tiers 1-3 in full. Bump for higher tiers.
WIDTH = 5


def digits_fixed(n: int, width: int = WIDTH) -> list[int]:
    """Non-negative int -> fixed-width zero-padded base-10 digits, MSB-first."""
    out = [0] * width
    i = width - 1
    while n > 0 and i >= 0:
        out[i] = n % 10
        n //= 10
        i -= 1
    return out


# --- bands (mirror the official tier prime ranges) -------------------------
# (name, min_bits, max_bits)  ->  primes in [2^min_bits, 2^max_bits)
BANDS: dict[str, tuple[int, int]] = {
    "tier1": (1, 3),    # {2,3,5,7} after intersecting with the fixed set
    "tier2": (4, 8),    # [16, 256)   ~ 48 primes
    "tier3": (9, 16),   # [512, 65536) ~ 6500 primes
    "tier4": (17, 32),  # [2^17, 2^32) -- sampled, huge
}


@dataclass
class PrimePools:
    """Train / held-out-val prime pools, per band."""

    train: dict[str, list[int]] = field(default_factory=dict)
    val: dict[str, list[int]] = field(default_factory=dict)

    @property
    def all_train(self) -> list[int]:
        return [p for ps in self.train.values() for p in ps]

    @property
    def all_val(self) -> list[int]:
        return [p for ps in self.val.values() for p in ps]

    def summary(self) -> str:
        rows = []
        for band in self.train:
            rows.append(
                f"  {band:>6}: train={len(self.train[band]):>5}  "
                f"val(held-out)={len(self.val.get(band, [])):>4}"
            )
        return "\n".join(rows)


def _stable_rng(seed: int, *tags: str) -> random.Random:
    h = hashlib.sha256(("|".join([str(seed), *tags])).encode()).digest()
    return random.Random(int.from_bytes(h[:8], "big"))


# Bands whose ENTIRE prime set can appear in the official eval and is small
# enough to enumerate -> the shipping model should train on all of them, so we
# never hold any out. tier1's official primes are the fixed set {2,3,5,7};
# tier2's secret 5 are drawn from only ~48 primes total.
ALWAYS_FULL: frozenset[str] = frozenset({"tier1"})


def build_pools(
    bands: list[str],
    *,
    holdout: bool = True,
    val_frac: float = 0.2,
    max_train_primes: int = 4000,
    max_val_primes: int = 400,
    seed: int = 0,
) -> PrimePools:
    """Enumerate primes per band and split into train / held-out-val pools.

    Two modes:
      holdout=True  (dev/measurement): tiers 2+ hold out a disjoint val pool so
                    the val metric reports honest generalization to UNSEEN
                    primes. tier1 is always full (its official primes are fixed).
      holdout=False (shipping): train on ALL enumerable primes per band; val
                    mirrors train (a fit check, not a generalization check).
                    Use this for the final model on the low, fully-enumerable
                    tiers where every possible official prime can be covered.

    For large bands (tier3+) the prime set is subsampled for cheap epochs, but
    train and val stay DISJOINT whenever holdout=True.
    """
    pools = PrimePools()
    for band in bands:
        if band not in BANDS:
            raise KeyError(f"unknown band {band!r}; known: {list(BANDS)}")
        lo_bits, hi_bits = BANDS[band]
        lo, hi = 2 ** lo_bits, 2 ** hi_bits

        if hi - lo <= 1_000_000:
            primes = list(primerange(lo, hi))
        else:
            # Too many to enumerate; sample candidates and snap to primes.
            rng = _stable_rng(seed, band, "enum")
            from sympy import nextprime
            seen: set[int] = set()
            target = max_train_primes + max_val_primes
            while len(seen) < target:
                c = rng.randrange(lo, hi)
                p = int(nextprime(c))
                if p < hi:
                    seen.add(p)
            primes = sorted(seen)

        full = (not holdout) or (band in ALWAYS_FULL)
        if full:
            train = primes[:max_train_primes]
            pools.train[band] = sorted(train)
            pools.val[band] = sorted(train)        # fit check, not held-out
        else:
            rng = _stable_rng(seed, band, "split")
            rng.shuffle(primes)
            n_val = max(1, int(round(len(primes) * val_frac)))
            pools.val[band] = sorted(primes[:n_val][:max_val_primes])
            pools.train[band] = sorted(primes[n_val:][:max_train_primes])
    return pools


def sample_batch(
    primes: list[int],
    batch_size: int,
    rng: random.Random,
    *,
    width: int = WIDTH,
    edge_frac: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Draw a batch of (a, b, p) with a,b in [0,p) and target a*b mod p.

    Returns int8 arrays:
        a_d (B,width), b_d (B,width), p_d (B,width), y_d (B,width)
    all base-10 MSB-first digits. Includes a small fraction of the structural
    edge cases the official set always contains (a or b in {0,1}).
    """
    a_d = np.empty((batch_size, width), np.int64)
    b_d = np.empty((batch_size, width), np.int64)
    p_d = np.empty((batch_size, width), np.int64)
    y_d = np.empty((batch_size, width), np.int64)
    for i in range(batch_size):
        p = primes[rng.randrange(len(primes))]
        if rng.random() < edge_frac:
            a = rng.choice([0, 1, p - 1])
            b = rng.choice([0, 1, p - 1])
        else:
            a = rng.randrange(p)
            b = rng.randrange(p)
        y = (a * b) % p
        a_d[i] = digits_fixed(a, width)
        b_d[i] = digits_fixed(b, width)
        p_d[i] = digits_fixed(p, width)
        y_d[i] = digits_fixed(y, width)
    return a_d, b_d, p_d, y_d


def sample_batch_balanced(
    train_pools: dict[str, list[int]],
    batch_size: int,
    rng: random.Random,
    *,
    width: int = WIDTH,
    edge_frac: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Like sample_batch but balances draws across BANDS, not across primes.

    Each example first picks a band uniformly, then a prime within that band's
    train pool. Without this, tier1's 3 primes would be ~3/4000 of the data and
    never learned; band-balancing keeps the near-term targets (tiers 1-2) well
    represented while still exposing the model to tier3+.
    """
    bands = [b for b, ps in train_pools.items() if ps]
    a_d = np.empty((batch_size, width), np.int64)
    b_d = np.empty((batch_size, width), np.int64)
    p_d = np.empty((batch_size, width), np.int64)
    y_d = np.empty((batch_size, width), np.int64)
    for i in range(batch_size):
        band = bands[rng.randrange(len(bands))]
        ps = train_pools[band]
        p = ps[rng.randrange(len(ps))]
        if rng.random() < edge_frac:
            a = rng.choice([0, 1, p - 1]); b = rng.choice([0, 1, p - 1])
        else:
            a = rng.randrange(p); b = rng.randrange(p)
        y = (a * b) % p
        a_d[i] = digits_fixed(a, width); b_d[i] = digits_fixed(b, width)
        p_d[i] = digits_fixed(p, width); y_d[i] = digits_fixed(y, width)
    return a_d, b_d, p_d, y_d


def sample_batch_wrapmix(
    train_pools: dict[str, list[int]],
    batch_size: int,
    rng: random.Random,
    *,
    width: int = WIDTH,
    edge_frac: float = 0.05,
    easy_frac: float = 0.5,
    wmax: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Band-balanced sampler with a CUSTOM data distribution (Saxena et al. 2024).

    Uniform a,b in [0,p) yields mostly HIGH-wrap products (wrap = floor(a*b/p)),
    the hard case; the model rarely sees the easy LOW-wrap structure. So for an
    `easy_frac` of examples we explicitly pick a small target wrap w in [0,wmax]
    and construct (a,b) with floor(a*b/p)==w exactly, mixing easy and hard in the
    same batch (a fixed mixture, not a time-based curriculum).
    """
    bands = [b for b, ps in train_pools.items() if ps]
    a_d = np.empty((batch_size, width), np.int64)
    b_d = np.empty((batch_size, width), np.int64)
    p_d = np.empty((batch_size, width), np.int64)
    y_d = np.empty((batch_size, width), np.int64)
    for i in range(batch_size):
        band = bands[rng.randrange(len(bands))]
        ps = train_pools[band]
        p = ps[rng.randrange(len(ps))]
        r = rng.random()
        if r < edge_frac:
            a = rng.choice([0, 1, p - 1]); b = rng.choice([0, 1, p - 1])
        elif r < edge_frac + easy_frac:
            # low-wrap example: choose target wrap w, build b so floor(a*b/p)==w
            a = rng.randrange(1, p)
            w = rng.randrange(0, wmax + 1)
            lo = (w * p + a - 1) // a              # smallest b with a*b >= w*p
            hi = ((w + 1) * p - 1) // a            # largest  b with a*b <  (w+1)*p
            lo = max(lo, 0); hi = min(hi, p - 1)
            b = rng.randint(lo, hi) if lo <= hi else rng.randrange(p)
        else:
            a = rng.randrange(p); b = rng.randrange(p)
        y = (a * b) % p
        a_d[i] = digits_fixed(a, width); b_d[i] = digits_fixed(b, width)
        p_d[i] = digits_fixed(p, width); y_d[i] = digits_fixed(y, width)
    return a_d, b_d, p_d, y_d


def _base_sample(train_pools, n, rng, width, wrapmix, easy_frac, wmax):
    if wrapmix:
        return sample_batch_wrapmix(train_pools, n, rng, width=width,
                                    easy_frac=easy_frac, wmax=wmax)
    return sample_batch_balanced(train_pools, n, rng, width=width)


def build_repeated_set(train_pools, size, rng, *, width=WIDTH,
                       wrapmix=False, easy_frac=0.5, wmax=8):
    """Pre-generate a fixed 'repeated set' of `size` examples (Charton & Kempe
    2024, two-set training). Returns digit arrays (size, width) x4."""
    return _base_sample(train_pools, size, rng, width, wrapmix, easy_frac, wmax)


def sample_twoset(rep, train_pools, batch_size, rng, *, p_rep=0.25,
                  width=WIDTH, wrapmix=False, easy_frac=0.5, wmax=8):
    """Two-set sampler: each example is drawn from the small fixed `rep` set
    with prob p_rep (repetition fosters learning), else freshly generated
    (diversity prevents overfit)."""
    rep_a, rep_b, rep_p, rep_y = rep
    s = rep_a.shape[0]
    fa, fb, fp, fy = _base_sample(train_pools, batch_size, rng, width, wrapmix, easy_frac, wmax)
    take_rep = np.array([rng.random() < p_rep for _ in range(batch_size)])
    ridx = np.array([rng.randrange(s) for _ in range(batch_size)])
    sel = take_rep[:, None]
    a = np.where(sel, rep_a[ridx], fa)
    b = np.where(sel, rep_b[ridx], fb)
    p = np.where(sel, rep_p[ridx], fp)
    y = np.where(sel, rep_y[ridx], fy)
    return a, b, p, y


if __name__ == "__main__":
    # Quick self-check / smoke test.
    pools = build_pools(["tier1", "tier2", "tier3"], seed=0)
    print("prime pools (train vs held-out val):")
    print(pools.summary())
    # disjointness invariant
    for band in pools.train:
        assert not (set(pools.train[band]) & set(pools.val[band])), f"{band} leak!"
    print("disjointness: OK")
    rng = random.Random(1)
    a, b, p, y = sample_batch(pools.train["tier2"], 6, rng)
    print("\nsample tier2 batch (digits MSB-first):")
    for i in range(6):
        av = int("".join(map(str, a[i]))); bv = int("".join(map(str, b[i])))
        pv = int("".join(map(str, p[i]))); yv = int("".join(map(str, y[i])))
        ok = (av * bv) % pv == yv
        print(f"  a={av:>5} b={bv:>5} p={pv:>5} -> y={yv:>5}  check={ok}")
