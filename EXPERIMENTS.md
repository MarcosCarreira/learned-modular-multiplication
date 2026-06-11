# Modular Arithmetic Challenge — experiment tracker (team MCSC, MAC01-T00008)

Goal: maximize `highest_tier_above_90` (primary), then `overall_accuracy`.
Reference baselines (public benchmark, OA): always_zero 0.079, digit_transformer
0.121, dlp_grokking 0.127.

## Status

Official public-benchmark pipeline (100 problems/tier), ranked by `H>90` then OA.

| Submission | OA | tier2 | tier3 | H>90 | State |
|---|---|---|---|---|---|
| #2 cross_prime_v1 (feed-forward) | 0.157 | ~0.49 | floor | 1 | superseded |
| #3 recurrent-ship (digit scratchpad) | 0.212 | 0.98 | — | 2 | superseded |
| #4 recurrent-v2-ship (Horner) | 0.283 | 0.97 | 0.72 | 2 | superseded |
| **#5 recurrent-v3-ship (Horner + reduce-step + Abacus, tier3-oversampled anneal)** | **0.306** | **0.99** | **0.93** | **3** | **LIVE & REGISTERED** `MCarreira/recurrent-v3-ship` @ `01f6f8a3fa56e3a3c17852dcf468560a0b6b8b16` |

**#5 cleared tier 3** (0.93 ≥ 0.90 on the official benchmark) → `H>90 = 3`, OA 0.306,
deterministic, ~56 s/1100 CPU. Cloud-verified via `run_check_submission_hf.py`.
Beats #4 on both keys. Holdout (unseen-prime) generalisation that underwrote it:
v2-Horner 0.82 ± 0.03 over 3 seeds; v3 holdout reached tier-3 0.945. Caveat: the
private final eval uses a secret seed; tier-3 0.93 on 100 samples is ~1.2 SE above
0.90 (thin margin) — resume `run_recurrent_v3_ship_auto.py` for more margin and
update before the 2026-08-12 deadline if desired. Beats every reference baseline
(dlp_grokking 0.127).

## Results log (2026-06-08/09)

| Experiment | Result | Verdict |
|---|---|---|
| cross_prime_v1 (d256) | tier2 0.49 (16k+resume) | **SUBMITTED #2** |
| run_shipping_v2 (d384 capacity) | tier2 0.29 @8k, tracks d256 exactly | capacity NOT the bottleneck |
| run_shipping_wrap / _data (wrap-mix + two-set) | tier2 0.40 @16k = same as plain d256 | data tricks give ~0 lift at our scale |
| run_variant_p (residue classifier) | tier2 0.22 @10k, behind digits | digits > residue classification |
| run_recip_experiment (tier3, holdout, 19k) | tier2 val 0.12 (2.4x floor, real unseen-prime signal); **tier3 train+val flat 0.05 throughout** | μ helps tier2 generalize a little; does NOT crack tier3 — encoder can't learn iterative reduction |
| **run_recurrent (scratchpad, holdout, 25k)** | **tier2 tr 0.86 (BROKE the 0.5 ceiling), val 0.35 (unseen primes, 3x recip!)**; tier3 still 0 (can't gen 10-digit product yet) | **BREAKTHROUGH on tier2** — first approach to compute (not memorize) and generalize. Step-limited, still climbing. |

**Core finding:** tier 2 ≈ 0.5 is a robust ceiling — resists capacity, wrap-mix,
and two-set. Tier 3 cross-prime generalization is unsolved by every approach
tried. Reciprocal features gave a sliver of tier-2 unseen-prime generalization
but did not crack tier 3 — a feed-forward encoder doesn't learn the iterative
reduction even with μ handed to it.

## NEXT (resume here)

1. Optionally finish `run_recip_experiment.py` to ~15k to confirm tier-3 *train*
   stays flat (the deciding number). If flat -> reciprocal is a clean negative.
2. **Untried lever: recurrent / iterative architecture** (allowed by rules) that
   can do multi-step computation (product -> scale by μ -> subtract) — the
   plausible path to learned Barrett / tier-3. Biggest open idea.
3. If tier 3 stays out of reach: write up the frontier map (no approach crosses
   tier-3 generalization, and why) as the Contributor-Network result — more
   valuable than a few points on tier 2. cross_prime_v1 stays the leaderboard entry.

## Risks / watch-items

- **Rules pivot to EFFICIENCY is materializing (now two signals).** (1) Zulip
  "challenge triviality" 2026-06-09: Tao said if accuracy gets trivial they may
  emphasize *speed/efficiency over accuracy*. (2) Blog comments (2026-06-09): a
  commenter argued it's not an "open problem" (representability is known, NN
  Turing-completeness); **Tao conceded and said they'll add "efficiently" to the
  basic question.** So efficiency is becoming the scored axis. This specifically
  pressures our autoregressive scratchpad (84-token gen, 43s/1100) vs a
  feed-forward or compiled solver (<5s). Mitigations: KV-cache (done); levers if
  it lands = shorter scratchpad, smaller model (v2 is only 4.8M), or distil the
  scratchpad into a one-shot student. Do NOT publicly argue "accuracy is trivial."
- **Hand-coded circuits not auto-banned.** Tao (blog comments): they're permissive,
  will "adjudicate letter AND spirit" case-by-case; a hand-coded arithmetic circuit
  "might be independently interesting." So a verilog2nn-style entry may not be DQ'd.
  Our trained+generalizing model is clean on both letter and spirit — the
  "learned not represented" framing is exactly the spirit axis they'd judge on.
- Leaderboard not yet public (experimental phase); standings unknown to all.

## The core finding so far

Tier 1: solved. Tier 2: feed-forward plateaus ~0.49 (memorizes ~48 tables;
capacity/data tricks give ~0 lift). The **autoregressive scratchpad broke this**
— tier 2 0.98, `H>90 = 2` (submission #3, `recurrent-ship`). Tiers 3+ were at the
floor for every feed-forward approach (incl. reciprocal features). The **v2
interleaved-Horner scratchpad** (`run_recurrent_v2.py`) is the first to move tier 3:
tier-3 *val ≈ train* climbing through ~0.6 on UNSEEN primes (run in progress) —
i.e. it generalizes the operation, the open problem. Output format matters (digit
decoding > residue classification — the p-variant lost).

## Runners (all arg-free; open in PyCharm, ▶)

| Runner | Lever it tests | Writes to | Source |
|---|---|---|---|
| `run_shipping.py` | baseline digit model (d256) | cross_prime_v1 | — |
| `run_resume.py` | more steps (warm start) | cross_prime_v1 | — |
| `run_shipping_v2.py` | **capacity** (d384, 8 layers) | cross_prime_v2 | — |
| `run_shipping_wrap.py` | **data: wrap-mixture** | cross_prime_wrap | Saxena 2410.03569 |
| `run_shipping_data.py` | **data: wrap-mix + two-set** | cross_prime_data | + Charton-Kempe 2410.07041 |
| `run_variant_p.py` | output_base="p" residue head | cross_prime_p_v1 | (negative result) |
| `run_recip_experiment.py` | reciprocal/Barrett features (tier3) | recip_v1 | — |
| `run_recurrent.py` | **autoregressive scratchpad (iterative, tier3)** | recurrent_v1 | Lee et al. 2023 |
| `run_check_submission.py` | pre-submission gauntlet | — | — |
| `run_upload_hf.py` | push to HF | — | — |
| `run_eval.py` | score any model(s) via official pipeline | — | — |

Smoke tests: `run_smoke.py` (digit), `run_smoke_p.py` (residue).

## Levers from the bibliography (ranked, with compliance notes)

1. **Wrap-mixture data** (Saxena et al.) — mix easy low-wrap / hard high-wrap
   products. Pure data, compliant. `run_shipping_wrap.py`. **Built.**
2. **Two-set training** (Charton & Kempe) — repeat a small fixed set with
   p=0.25. Pure data, compliant. Aimed at faster table memorization.
   `run_shipping_data.py` (combined with #1). **Built.**
3. **Capacity** — d_model 384+. `run_shipping_v2.py`. **Built.**
4. **Circular/angular loss** (Saxena, 2nd lever) — puts the answer on a circle.
   CAVEAT: works for *addition* mod q (rotation), but *multiplication* mod p is
   not rotation on that circle (needs discrete log) — likely the same wall as
   the discrete-log approach. Parked.
5. **Fourier weights / init** (Gromov 2301.02679) — closed-form modular-arithmetic
   weights. CAVEAT: a full closed-form solution is a by-construction circuit
   (prohibited); usable only as an *initialization* that is then genuinely
   trained (provenance matters). Speculative. Parked.

## Decision tree

- **Tier 2 → 0.90** via any of capacity / wrap / data → re-upload, `H>90` = 2,
  decisive lead. Compare `run_shipping_v2`, `run_shipping_wrap`, `run_shipping_data`,
  pick the winner, then combine (d384 + best data recipe).
- **Tier 2 stuck ~0.5** across all of them → exact 48-table memorization is the
  hard limit for this architecture family; bank ~0.49 and shift to the tier-3
  frontier via `run_recip_experiment.py` (the citable cross-prime result).

## Compliance invariants (every submission)

Answer from trained params (weight-perturbation collapses it); `a%p`, `b%p`
legal two-operand reductions only; no `a*b%p` formed in code; no sympy/Barrett/
CRT/lookup in the forward path; deterministic on CPU (`MODCH_FORCE_CPU=1`);
dev-only sympy stays in `dev/`, never in a submission dir.

---

*Experiments, code, and a first draft of the write-up were produced with the assistance of a large language model (Claude), under the author's direction and verification.*
