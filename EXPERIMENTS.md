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
| #5 recurrent-v3-ship (Horner + reduce-step + Abacus, tier3-oversampled anneal) | 0.306 | 0.99 | 0.93 | 3 | superseded by #6 — `MCarreira/recurrent-v3-ship` @ `01f6f8a3fa56e3a3c17852dcf468560a0b6b8b16` |
| **#6 recurrent-v4 (WIDTH-10, base-10 `a·digit` Horner, tier4-anneal)** | **0.40** | **0.97** | **0.95** (T4 **0.96**) | **4** | **LIVE & REGISTERED** `MCarreira/recurrent-v4` @ `467ad5d7180bbfae21b55185cfeca22e2e18b458` |
| recurrent-v5 (WIDTH-20, same recipe widened, tier5-anneal) | 0.47 | 0.98 | 0.90 (T4 0.88, **T5 0.80**) | 3 | NOT submitted — scaling result `MCarreira/recurrent-v5` @ `3a6a45aea9...e54569ff52` |
| recurrent-v6 (WIDTH-20, **2× DEPTH ~20M**, tier5-focus + anneal) | 0.46 | 0.98 | 0.90 (T4 0.86, **T5 0.72**) | 3 | NOT submitted — **capacity NEGATIVE** `MCarreira/recurrent-v6` @ `275c14078f...2740c50f12` |

**recurrent-v6 — CAPACITY IS NOT THE LEVER (official Playground, 2026-07-02).** The
clean "does more capacity clear tier 5?" test: same recipe made **deeper** (8→14
layers, ~11M→~20M, warm-started from v5 with the 6 new layers near-identity),
tier-5-weighted mix, finishing anneal. **Result: it made tier 5 WORSE.** Dev looked
promising (touched H>90=5 on favourable draws, settled center t5 ~0.86) but the
official secret-seed benchmark: T1 1.00, T2 0.98, T3 0.90, **T4 0.86, T5 0.72**,
H>90=3, OA 0.46, 108 s. Official tier5 **0.72 < v5's 0.80** — doubling params LOWERED
it. Diagnostic: the dev→official gap WIDENED with capacity (~0.06 at 11M → ~0.14 at
20M) = the bigger model overfit the fixed-seed dev eval instead of generalizing
(cherry-picking the dev-best checkpoint compounds it). CONCLUSION (corroborates
Sneiderman/neural-horner): the tier-5 ceiling is **drift/generalization-limited, not
capacity-limited**; the only remaining lever is exposure-bias / drift reduction
(scheduled sampling, DAgger), not scale. A strong citable negative. v4 stays the
submission; v5 stays the scaling point. Folded into WRITEUP as "Is the ceiling
capacity or drift? A depth test that says drift."


**recurrent-v5 — THE 20-DIGIT SCALING POINT (official Playground, 2026-06-28).**
Same recipe, one literal widened (`WIDTH 10→20`), warm-started from v4, curriculum
1→20 digits then finishing anneal. Official: T1 1.00, T2 0.98, T3 0.90, T4 0.88,
**T5 0.80**, OA 0.47, **53.5 s**/1100 (≈10× v4, still well under 300 s budget), 44 MB.
**Frontier T3** — tier4 came in 0.88, *just* under the 0.90 bar (dev eval wobbled
0.90–0.93; the official seed resolved below), so v5 is `H>90 = 3` and **v4 stays the
live submission** (H>90=4 > 3 on the primary key; would have been a downgrade to
swap). v5's value is the **scaling curve 0.98 / 0.90 / 0.88 / 0.80 across 2/5/10/20
digits** — gentle ~0.18 decline over a 16× operand-magnitude range, the operation
generalising to 20-digit unseen primes. Honest cost: v5's T3/T4 (0.90/0.88) sit
below v4's (0.95/0.96) — a fixed ~11M-param budget spread across five tiers of longer
sequences loses per-tier sharpness; "reaches 20 digits with graceful degradation,"
not "wider is uniformly better." Making v5 a *safe* H>90=4 (or H>90=5) would need a
bigger model + much more compute for marginal-to-no leaderboard gain → not pursued.

**#6 cleared tier 4** on the official SAIR Playground (1100 cases, the challenge's
own hardware, 2026-06-25): tier1 1.00, tier2 0.97, tier3 0.95, **tier4 0.96** →
`H>90 = 4`, OA **0.40**, **5.107 s** for 1100 problems (≈60× under the 300 s
budget), 43 MB, deterministic. Beats #5 on both keys. **Timing caveat learned the
hard way:** a local `x86`-emulated Docker run (Docker Desktop on Apple silicon →
QEMU) clocked the same model at 600–900 s and looked like a budget timeout; the
authoritative Playground ran it in ~5 s. **Trust the Playground, not emulated local
Docker, for timing.**

**#5 cleared tier 3** (0.93 ≥ 0.90 on the official benchmark) → `H>90 = 3`, OA 0.306,
deterministic, ~56 s/1100 CPU. Cloud-verified via `run_check_submission_hf.py`.
Beats #4 on both keys. Holdout (unseen-prime) generalisation that underwrote it:
v2-Horner 0.82 ± 0.03 over 3 seeds; v3 holdout reached tier-3 0.945. Caveat: the
private final eval uses a secret seed; tier-3 0.93 on 100 samples is ~1.2 SE above
0.90 (thin margin) — resume `run_recurrent_v3_ship_auto.py` for more margin and
update before the 2026-08-12 deadline if desired. Beats every reference baseline
(dlp_grokking 0.127).

**v4 — THE SCALING RESULT (REGISTERED AS #6, 2026-06-25; H>90=4 official).** `recurrent_v4` = same
recipe widened to tier 4: WIDTH 5→10, base-10 `a·digit` Horner (NB=10 steps, one
per digit), d384/8L, holdout, tier-4-oversampled. First tried base-1000 chunks
(NB=4) to cut steps — **abandoned**: the 3-digit×3-digit per-step multiply was
unlearnable (tier-2 stalled 0.34, tier-3 stuck at 0). Lesson: per-step
learnability > step-count. Base-10 (v3's proven trivial step, widened) works — and **CLEARED TIER 4 ON THE
HOLDOUT**: after a finishing LR anneal (forced by shortening STEPS 200k→120k), v4
clears **all four tiers on UNSEEN primes** — tier-2 0.96, tier-3 0.93, tier-4
**0.905** (held 0.89–0.93 across 8 post-anneal evals), `H>90 = 4` on the holdout,
frontierVal ~1.95. tier-4 climbed 0.26→0.56→0.67→0.76 (plateau) → **0.91** (anneal).
**Main result = the OPERATION SCALES NEARLY FLAT**: same algorithm, more digits,
generalisation carries across the operand-length doubling within ~0.02 (tier-3 0.93
→ tier-4 0.905) instead of collapsing like a table — strongest evidence it learned
the operation, citable independent of the leaderboard. Single seed (converged).
Cost model linear (NB steps = digits) = the P-N-P composition depth, supplied by
recurrence. Memory pinned <~31GB via batch 32 + per-step `mps.empty_cache` +
sub-batched eval + 25-min chunk restarts. Folded into `WRITEUP` as the headline.

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

Answer from trained params (weight-perturbation collapses it); no `a*b%p` formed
in code; no sympy/Barrett/CRT/lookup in the forward path; deterministic on CPU
(`MODCH_FORCE_CPU=1`); dev-only sympy stays in `dev/`, never in a submission dir.

**CORRECTION (organizer ruling, 2026-07-02, reaffirmed 2026-07-14):** the earlier
invariant "`a%p`, `b%p` legal two-operand reductions" is **WRONG** — input must be
raw `(a, b, p)`; reducing the full-width operands mod p is a *core part of the task*
and may not be done in preprocessing (`operand_bits` is ~2–4× the prime bits on
purpose). The whole v3–v6 line pre-reduced and is therefore **out of spec**. The
`digit_transformer` reference example that appeared to bless pre-reduction was
confirmed an error in the rules, not a precedent. See the Final result below.

## Final result (compliant) — 2026-08-12

**Submission (scored):** `MCarreira/recurrent-raw` @
`f9a757ab5c52d12f0beb6fc76b43a9ff326f327f` — takes RAW `(a, b, p)`, reduces the
full-width operands **in-network**. One learned primitive
`cell(acc, m, digit) = (acc·R + m·digit) mod p` on a fixed two-phase schedule:
Phase 1 reduces `a mod p` (Horner over `a`'s raw digits, `m=1`), Phase 2 multiplies
over `b`'s raw digits (`m = a_red`). Verified arithmetically exact over 60k cases;
tiers 1–3 range (raw operands to 64 bits). Official Playground (1100 cases, det., 11 s):

| tier | official | dev | dev→official shave |
|---|---|---|---|
| T1 | **1.00** | 1.00 | — |
| T2 | **0.97** | 0.99 | −0.02 |
| T3 | 0.69 | 0.84 | **−0.15** |
| T4–T10 | ~0.02 (emits 0) | — | — |

→ **Frontier T2, compliant `H>90 = 2`, OA 0.28.** First *valid* entry; replaced the
out-of-spec `recurrent-v4`.

**Tier 3 is drift-limited — confirmed THREE independent ways (all negative):**

1. **Capacity — no.** 2× depth (v6, 8→14 layers, ~20M) *lowered* the frontier tier on
   the sub-problem and widened the dev→official gap (more overfitting, not more correct
   rollout).
2. **Harder on-policy — no.** `recurrent_raw_t3d`: scheduled sampling 0.30→0.55,
   warm-started from the T3 best, tracked the prior run step-for-step and plateaued at
   the same place. Stopped.
3. **Inference-time self-consistency — no (decisive).** 3-model rule allowed; submitted
   two voting variants (same weights, majority vote over 7 sampled rollouts, seeded for
   determinism, config in manifest). Both **COMPLETED** on the Playground. Tier 3:
   **0.69 greedy / 0.68 @ temp 0.5 / 0.69 @ temp 0.8** — flat. Raising temperature
   diversified rollouts and the vote still didn't move → the tier-3 errors are
   **systematic** (the model reduces the same cases wrong every sample), not stochastic.
   Self-consistency can only rescue stochastic noise; here it rescues nothing.

**Verdict:** the long-rollout ceiling is **systematic drift**, not capacity and not
sampling variance — corroborated independently by Sneiderman (`neural-horner`, base-2
GRU) and Lei (radix-4 fork; L2-SP/EWC found to *relocate* interference, not remove it).
Breaking it needs a lever that changes what the parameters *compute* on the drift-prone
cases (expert-relabel DAgger with correct-continuation targets, or a shorter reduction
rollout) — filed, not pursued. **Competition arc closed: compliant `H>90 = 2` banked.**

New files this phase: `models/recurrent_raw{,_t3,_t3d,_vote05,_vote08}/`,
`run_recurrent_raw*.py`, `run_upload_votes.py`; voting + manifest-config +
deterministic seed in `models/recurrent_raw/model.py` (env `MODCH_VOTES`/`MODCH_TEMP`
for local sweeps; manifest `votes`/`temp`/`seed` for the self-contained submission).
