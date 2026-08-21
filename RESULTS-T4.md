# Measured on a Colab Tesla T4

**Commit `a6c0285`.**  \
*(The `dev/HD-car` history was rewritten after these measurements were taken, which renamed
every commit on that branch. `a6c0285` is the surviving name for the commit this file was
measured at; the hash it originally quoted no longer resolves and has been removed rather than
left as a dead citation. Commits from before the branch diverged — `b2523ba`, for instance — are
untouched.)* Tesla T4 (15360 MiB, compute 7.5, driver 580.82.07, CUDA driver 13.0) ·
Linux x86_64 · Python 3.12.13 · **OpenCV 4.14.0** · NumPy 2.0.2 · numba 0.60.0 · cupy present.

Everything here is produced by committed code:

```bash
pytest tests/ -q                                              # §1, §2
python bench_post.py --sizes 480 720 1080                     # §3, §4
python bench_post.py --sizes 480 --with-sequential --skip-equivalence
python bench_t4.py                                            # §2 long form, §5, §6
python bench_fill.py --sizes 240 480 720 1080                 # §7
```

---

## 1. What running on real hardware found

CUDASIM runs NumPy semantics; a T4 runs compiled PTX. Two defects showed up only here.

**A tiled median that had never compiled.** `post_kernels.py` declared its shared array as
`cuda.shared.array((TILE_Y + 2 * HALO, TILE_X + 2 * HALO), uint8)`. The shape must be a
compile-time constant; written inline the arithmetic types as `int64` and matches no overload:

```
Overload of function 'array': With argument(s):
'(UniTuple(int64 x 2), class(uint8))': No match.
```

Under CUDASIM the shape is just a Python tuple, so `median5_tiled_kernel` passed the simulator
for as long as it existed **and had never once run on a GPU** — and v2 depends on it, so v2 had
never worked on hardware either. Fixed by hoisting to `SH_TILE_H` / `SH_TILE_W`.
`test_every_kernel_actually_compiles_on_real_hardware` now marks the bug class, and **skips
under CUDASIM by design**: a green simulator suite is necessary and not sufficient.

**A CuPy backend that was a different algorithm.** `step_kernel_cp_v1.cu` had Zivkovic's
complexity-reduction step commented out — the component prune and the `nmodes` decrement both.
Nothing compared CuPy against anything. Restored, and the parity test ran here for the first
time: identical masks on all 20 frames, **464 mode drops on both backends**, weights identical,
means and variances within 1.6e-5.

**66 / 66 tests pass**, CuPy included.

---

## 2. Correctness

### The Q8 premise, on a fourth OpenCV build

The blur reproduces `cv2.GaussianBlur`'s fixed-point path exactly. That claim is build-sensitive,
so it has been re-derived on every build the project has touched:

| build | Q8 chain vs `cv2.GaussianBlur` | `cvtColor` formula |
|---|---|---|
| 4.13.0, Apple NEON | 0 differ | 0 differ |
| 5.0.0, Apple NEON | 0 differ | 0 differ |
| 5.0.0, x86 + Intel IPP | 0 differ | 0 differ |
| **4.14.0, x86 (this run)** | **0 differ** | **0 differ** |

An *ideal* float64 Gaussian differs from `cv2` on ~15% of pixels by one grey level. That gap is
OpenCV's own departure from a true Gaussian, and reproducing OpenCV means reproducing it.

### Kernels against OpenCV, on compiled PTX

Zero mismatches across ten shapes up to 1080p, ragged 37×53, degenerate **1×64 and 64×1**,
all-zero, all-255, border-lit, column-0-lit, and mask values {0, 1, 127, 255}. The degenerate
widths matter specifically: that is the `_reflect101` `n == 1` case, which loops forever without
the early return. It ran on hardware and terminated.

### The equivalence gate, long form

120 frames at 1080p, **every frame, mask and composite**, against the host chain:

| backend | mask diff | composite diff | verdict |
|---|---|---|---|
| Numba (host, reference) | 0 | 0 | IDENTICAL |
| v0 CUDA + host post | 0 | 0 | IDENTICAL |
| v1 GPU post + blur | 0 | 0 | IDENTICAL |
| v2 fused + tiled | 0 | 0 | IDENTICAL |

**1 492 992 000 pixel positions compared against the reference, zero differences** — three
non-reference backends × 120 frames × 2 073 600 px × (mask + composite). Foreground in 120/120
frames. The sweep runs on synthetic frames; the CDnet parity run still needs the dataset.

---

## 3. End to end

| | v0 | v1 | v2 | v2 vs v0 | MB/frame |
|---|---|---|---|---|---|
| 480p | 10.22 ms · 97.9 FPS | 3.74 · 267.3 | **3.26 · 307.1** | **3.14×** | 6.97 → 3.28 |
| 720p | 20.82 ms · 48.0 FPS | 6.98 · 143.3 | **6.02 · 166.2** | **3.46×** | 15.67 → 7.37 |
| 1080p | 55.03 ms · 18.2 FPS | 13.75 · 72.8 | **11.26 · 88.8 FPS** | **4.89×** | 35.25 → 16.59 |

### Against the proposal's targets

| target | result |
|---|---|
| **>30 FPS at 1080p on a T4** | **88.8 FPS** — met, ~3× the target |
| **>20× over sequential Python** | **887×** at 480p (sequential: 3034.62 ms/frame, 0.3 FPS) |

887× is true and flattering; that baseline is a per-pixel interpreter loop. The honest question
is what the GPU bought over competent CPU code:

| | Numba CPU | v2 GPU | speedup |
|---|---|---|---|
| 480p | 43.03 ms · 23.2 FPS | 3.26 ms · 307.0 FPS | 13.21× |
| 720p | 52.39 ms · 19.1 FPS | 5.91 ms · 169.3 FPS | 8.87× |
| 1080p | 124.07 ms · 8.1 FPS | 12.11 ms · 82.6 FPS | **10.25×** |

(Single passes, not medians — quote §3's table for the pipeline. The 480p Numba figure barely
differs from 720p, which is not credible scaling; read that row's 13.21× as noise around the
~10× the other two rows agree on.)

### A speedup that shrank because the baseline got fairer

An earlier run reported **5.72×** for v2 over v0 at 1080p. It is **4.89×** here, and nothing
about v2 got slower — v0 got faster, 66.12 ms → 55.03 ms. The host path was converting to
float32 twice: `main.py` cast the frame, then `to_planar` cast the transpose again, 2.2 ms a
frame at 1080p for an identical array. Removing that improved every host-side path including the
baseline. The old 5.72× was partly measuring waste in the thing being beaten.

---

## 4. Per-stage, all three versions — the bottleneck moved as predicted

Sync-bounded wall clock (`perf_counter` with every device stage synchronised before the clock is
read — not CUDA events), 16 timed frames per version, fresh model per version, measured at
commit `bc55214` on the same T4. v0's chain is split finer than v1/v2's because those are
exactly the stages v1 moves onto the device, so the columns line up by construction.

**480p (854×480)**

| stage | v0 | v1 | v2 |
|---|---|---|---|
| host cvtColor | 0.285 ms · 2.7% | — | — |
| f32 H2D + model + mask/bg D2H | 4.517 · 42.5% | — | — |
| host threshold + median | 1.476 · 13.9% | — | — |
| ingest (H2D+cvt+model+post+D2H) | — | 1.175 · 32.0% | 1.149 · 36.6% |
| host `fill_holes` | 0.965 · 9.1% | 0.877 · 23.9% | 0.882 · 28.1% |
| host blur + composite | 3.385 · 31.8% | — | — |
| composite (H2D+blur+D2H) | — | 1.616 · 44.0% | 1.108 · 35.3% |
| **total** | **10.627** | **3.668** | **3.140** |

**720p (1280×720)**

| stage | v0 | v1 | v2 |
|---|---|---|---|
| host cvtColor | 0.576 ms · 2.8% | — | — |
| f32 H2D + model + mask/bg D2H | 8.778 · 42.2% | — | — |
| host threshold + median | 2.948 · 14.2% | — | — |
| ingest (H2D+cvt+model+post+D2H) | — | 1.940 · 27.9% | 1.984 · 33.1% |
| host `fill_holes` | 2.155 · 10.4% | 1.938 · 27.8% | 1.972 · 32.9% |
| host blur + composite | 6.333 · 30.5% | — | — |
| composite (H2D+blur+D2H) | — | 3.083 · 44.3% | 2.044 · 34.1% |
| **total** | **20.790** | **6.962** | **6.000** |

**1080p (1920×1080)**

| stage | v0 | v1 | v2 |
|---|---|---|---|
| host cvtColor | 1.220 ms · 2.2% | — | — |
| f32 H2D + model + mask/bg D2H | 28.585 · 52.0% | — | — |
| host threshold + median | 6.119 · 11.1% | — | — |
| ingest (H2D+cvt+model+post+D2H) | — | 3.651 · 25.8% | 3.510 · 29.5% |
| host `fill_holes` | 4.622 · 8.4% | 4.441 · 31.4% | **4.719 · 39.7%** |
| host blur + composite | 14.383 · 26.2% | — | — |
| composite (H2D+blur+D2H) | — | 6.057 · 42.8% | 3.665 · 30.8% |
| **total** | **54.930** | **14.149** | **11.894** |

Totals sit close to §3's end-to-end medians (55.03 / 13.75 / 11.26 at 1080p; here 54.93 /
14.15 / 11.89 — v0 −0.2%, v1 +2.9%, v2 +5.6%). These are separate runs under different
protocols: §3 is the median of five interleaved passes, this a single instrumented pass. The
stage boundaries add no synchronisation of their own — the pipeline methods already sync
internally — so the offsets are what two runs on a shared T4 look like; no variance was
recorded, so they are reported as deltas rather than dismissed as noise. Quote §3 for
end-to-end figures; read this section for how the frame divides. Across the two runs the stage
*ranking* is stable — fill_holes is v2's largest 1080p stage in both — while the shares
themselves move by up to 3.8 percentage points (fill 35.9% → 39.7%), which is the honest
resolution of a single-run percentage on a shared T4.

For charting, v0's five stages fold into the three buckets v1/v2 report — *mask production*
(cvtColor + model + threshold/median), *fill*, *blur+composite* — giving v0 mask production of
**6.278 / 12.302 / 35.924 ms** at the three sizes. The charts in `figs_final.py` and the
notebook use exactly these folds.

**What the three columns say, read together at 1080p:**

- **The ingest transformation is 28.59 ms → 3.65 ms.** v0's biggest stage — planar-float32
  upload plus the model kernel plus two copies back — becomes v1's smallest meaningful one.
- **Tiling is visible end to end, not just in isolation:** v1's composite 6.06 ms → v2's
  3.67 ms. That is the 2.36× kernel-level tiling win of §5 surviving contact with the transfers
  around it.
- **`fill_holes` is the largest single stage of v2 at 1080p — 39.7%.** At 480p the three stages
  are within a millisecond of each other and ingest is nominally largest; at 720p they are a
  three-way tie (33.1 / 32.9 / 34.1). The clean statement is: the higher the resolution, the
  more the host flood fill dominates. (An earlier v2-only run put it at 36.7%; the ranking is
  stable across runs, the percentage point is not.)

The plan predicted, before any of this was built: *"the frame budget should be dominated by the
MOG2 kernel's model-state traffic and the host `fill_holes` — **not** by the blur. If so, say so:
the next real win is streams, not a better blur."*

**Confirmed at 1080p.** The next win is CUDA streams overlapping frame N+1's ingest with frame
N's flood fill.

---

## 5. Kernel 2 — blur and composite, isolated

| | host `cv2` | GPU naive | GPU tiled | host/tiled | **naive/tiled** |
|---|---|---|---|---|---|
| 480p | 3.300 ms | 0.876 | **0.370** | 8.9× | **2.36×** |
| 720p | 6.397 ms | 1.874 | **0.793** | 8.1× | **2.36×** |
| 1080p | 13.740 ms | 3.690 | **1.554** | 8.8× | **2.37×** |

Interleaved across variants this time; the previous run batched them. The tiling ratio came back
**2.36 / 2.36 / 2.37**, against 2.37 / 2.36 / 2.37 batched — the effect is not a measurement
artefact.

---

## 6. Kernel 0 — frame ingest, decomposed at 1080p

"Ingest is 14× faster" bundles three changes, only one of which is the conversion kernel:

| stage | ms |
|---|---|
| A host `cvtColor` | 1.270 |
| B host float32 + transpose | 15.243 |
| C H2D 12 B/px, **allocating** (what v0 does) | 6.700 |
| D H2D 12 B/px, preallocated | 5.140 |
| E H2D 3 B/px, preallocated | 1.464 |
| F conversion kernel on device | **0.184** |

**old (A+B+C) 23.21 ms → new (E+F) 1.65 ms, 14.1×**

| cause of the 21.6 ms saved | ms | share |
|---|---|---|
| host colour convert + transpose removed | 16.51 | 76% |
| 12 B/px → 3 B/px upload | 3.68 | 17% |
| device allocation removed (v0 allocates 25 MB/frame) | 1.56 | 7% |
| conversion kernel added back | −0.18 | — |

The conversion kernel costs **0.184 ms** — nearly free, which is what made the 3 B/px upload
possible, since the conversion had to happen somewhere. The largest single win is not a kernel:
it is deleting a 25 MB numpy transpose from the host.

Stage B read 6.603 ms in the previous run and 15.243 ms here, with the stages now interleaved
rather than batched. A 2.3× swing on a host memory operation is a property of the shared VM, not
of the code; the 76% share is robust to it because everything else moved far less.

---

## 7. The flood fill, and why the pass count is the result

`bench_fill.py` implements morphological reconstruction, **asserts it agrees with `floodFill`
pixel-for-pixel**, and times both.

| | floodFill | reconstruct | ratio | **passes** |
|---|---|---|---|---|
| 320×240 | 0.24 ms | 4.07 | 17.2× | **132** |
| 854×480 | 1.08 ms | 43.65 | 40.4× | **266** |
| 1280×720 | 2.62 ms | 143.00 | 54.5× | **399** |
| 1920×1080 | 5.50 ms | 615.89 | 112.0× | **593** |

Run on an Apple M-series host the ratio at 1080p was 287×; here it is 112×. **The pass counts
are identical on both machines** — 132 / 266 / 399 / 593. That is the point the file argues:
milliseconds are a property of the machine, the pass count is a property of the algorithm and
the image. Each pass is a grid-wide dilate, and a CUDA implementation a hundred times faster per
pass still needs 593 of them in sequence.

Both timings are CPU. No CUDA reconstruction was written or measured, and the claim is "this
formulation costs hundreds of dependent full-frame passes", not "no parallel flood fill could
ever be worthwhile".

---

## 8. Predictions, written before measuring

| | predicted | measured |
|---|---|---|
| host blur+composite @1080p | 4–8 ms | **13.7 ms** |
| GPU blur kernels @1080p | 0.2–0.4 ms | **1.55 ms** (tiled) |
| **tiled vs naive** | *"probably close to naive"* | **2.36×**, twice, batched and interleaved |
| ingest+blur saved @1080p | 4–8 ms | **~34 ms** |
| bottleneck after the change | host `fill_holes`, not the blur | **confirmed, 39.7% at 1080p** |
| bus traffic reduction @1080p | −38.5% | **−52.9%** (the estimate omitted v0's `bg_prob` D2H) |

**Five of the six missed; only the bottleneck call held.** They did not miss the same way, and
lumping them together would hide the interesting half:

- **Three underestimated the host and the bus** — the host blur, the ingest saving and the bus
  reduction were all larger than predicted. This is why the optimisation target became the host
  rather than the arithmetic.
- **One underestimated what a GPU kernel costs** — the blur kernels were predicted at
  0.2–0.4 ms and measured at 1.55 ms.
- **One went the other way entirely** — shared-memory tiling was expected to barely beat the
  naive blur and won **2.36×** (2.36 / 2.36 / 2.37 interleaved; 2.37 / 2.36 / 2.37 batched). Here
  the GPU did *better* than predicted, and saying so is the point of keeping the table.

---

## 9. Not measured here

**F1 / IoU on CDnet — now re-measured.** The original dataset hosts are offline, so the team
mirrors `baseline/highway` on Hugging Face
(`haiduonghuynhle/changedetection-2012-highway`, sha256 in the record file). Re-run at the
current HEAD, full 470–1700 window: the shipping chain reproduces
**F1 0.9843 / IoU 0.9691 / P 0.9863 / R 0.9823, 0 empty frames — digit for digit** the figures
first measured at `b2523ba`, as expected since the host chain never changed. So do the rejected
chains quoted in the report (BGR 0.8272; CLOSE precision 0.9634; contour 0.5541). Raw output:
`benchmarks/records/eval_highway_full.txt`.

The first GPU-backend parity sweep over the real 1231 frames
(`eval_highway.py --model cuda_v2 --parity-vs numba`) observed 2 differing mask pixels
out of 94,540,800 and stopped under the original strict gate (captured at
`benchmarks/records/parity_cuda_v2_vs_numba_t4.txt`). The rerun at `f6c0bfc` with the
seed-tracing gate classified both, for cuda_v2 **and** cuda_v1: one flip is a direct
output-threshold straddle (bg_prob 0.4999971 vs 0.5000015), and the other traces to a
proven one-ulp seed at the `dist2 < Tb·var` branch (margins +0.0002746582 vs
−0.00030517578 on ulp-close states) whose state divergence then cascaded. Verdict
FLOAT-BOUNDARY; the integer stages are exact. Raw output, plus the real-footage FPS
table and the x86 F1 re-reproduction from the same session:
`benchmarks/records/t4_final_run.txt`.

**The 81.8%-on-host profile** quoted in the notebook is a one-off from earlier in the project,
kept because it is what motivated v1. §4 is the version this repository reproduces.
