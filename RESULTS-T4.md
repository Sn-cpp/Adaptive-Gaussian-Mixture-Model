# Measured on a Colab Tesla T4

Environment: **Tesla T4** (15360 MiB, compute 7.5, driver 580.82.07, CUDA driver 13.0) ·
Linux x86_64 · Python 3.12.13 · OpenCV 5.0.0 (**Intel IPP 2026.0.0**, pthreads) ·
NumPy 2.0.2 · numba 0.60.0.

Everything below was produced by running the repository at `dev/HD-car` on that machine.
Timings are medians of interleaved passes; each pass rebuilds its model.

Reproduce with:

```bash
pytest tests/ -q                              # §1, §2 correctness
python bench_post.py --sizes 480 720 1080     # §3 the v0/v1/v2 table, §4 per-stage
python bench_post.py --sizes 480 --with-sequential --skip-equivalence
python bench_t4.py                            # §2 long equivalence, §3 Numba baseline, §5
python bench_fill.py                          # the flood-fill comparison
```

`bench_t4.py` exists because four of the tables here were first produced by one-off cells typed
into a Colab notebook. They are kept as committed code so this file is checkable rather than
merely reported.

---

## 1. What running on real hardware found

The kernels had only ever been verified under `NUMBA_ENABLE_CUDASIM=1`. The first run on a
T4 failed five tests, all one root cause, and **not in the new code**:

```
gmm_mask/gpu/post_kernels.py:79
    cuda.shared.array((TILE_Y + 2 * HALO, TILE_X + 2 * HALO), uint8)

Overload of function 'array': With argument(s):
'(UniTuple(int64 x 2), class(uint8))': No match.
```

`cuda.shared.array` requires a compile-time constant shape. Written inline, the arithmetic
types as `int64` and matches no overload. **Under CUDASIM the shape is just a Python tuple, so
it never noticed** — `median5_tiled_kernel` passed the simulator for as long as it existed and
had never once run on a GPU. v2 depends on it, so **v2 had never worked on real hardware
either**, and the v0/v1/v2 comparison it belonged to had never run at all (the v0 baseline was
separately dead: `gmm_mask_cuda.py:21` read `self.d_mask` before assigning it).

Fixed by hoisting to `SH_TILE_H` / `SH_TILE_W`. `blur_kernels.py` already precomputed its
shapes, which is why the new blur kernels compiled and the pre-existing median did not.

`tests/test_post_chain.py::test_every_kernel_actually_compiles_on_real_hardware` now marks this
bug class. It **skips under CUDASIM by design**: a green simulator suite is a necessary and not
a sufficient condition, and the test exists to say so.

**Result after the fix: 52 / 52 tests pass on the T4.**

---

## 2. Correctness

### The Q8 premise, re-derived on the target machine

Colab's OpenCV is a **different dispatch path** from the Apple Silicon NEON build the work was
developed against — x86 with IPP 2026.0.0 enabled. This was the largest risk in the plan.

| check | result |
|---|---|
| cumulative Q8 kernel | `[9,11,15,17,19,22,23,24,23,22,19,17,15,11,9]`, sum 256 — identical |
| fixed-point chain vs `cv2.GaussianBlur` | **0 / 66 654 pixels differ** |
| *ideal* float64 Gaussian vs `cv2` | **15.2%** of pixels differ, max 1 grey level |
| fixed-point formula vs `cv2.cvtColor` | **0 pixels differ** |

Three builds now agree: 4.13.0 (NEON), 5.0.0 (NEON), 5.0.0 (x86 + IPP).

### Kernels against OpenCV, on compiled PTX

| check | shapes | mismatches |
|---|---|---|
| blur naive / tiled vs `cv2.GaussianBlur` | 32×48, 37×53, 120×160, 240×320, 480×854, **1080×1920**, 15×15, 8×9, **1×64, 64×1** | **0** |
| composite naive / tiled vs `cv2.copyTo` | same | **0** |
| all-0, all-255, border-lit, column-0-lit | 64×64 | **0** |
| mask values {0, 1, 127, 255} | 64×64 | **0** |
| `bgr2ycrcb_planar_kernel` vs `cv2.cvtColor` | 240×320 incl. saturation rows | **0** |

`1×64` and `64×1` matter specifically: that is the `_reflect101` `n == 1` case, which loops
forever without the early return. It ran on hardware and terminated.

### The equivalence gate, long form

120 frames at 1080p, every frame, every pixel, **mask and composite**, against the host chain:

| backend | mask diff | composite diff | verdict |
|---|---|---|---|
| Numba (host, reference) | 0 | 0 | IDENTICAL |
| v0 CUDA + host post | 0 | 0 | IDENTICAL |
| v1 GPU post + blur | 0 | 0 | IDENTICAL |
| v2 fused + tiled | 0 | 0 | IDENTICAL |

**1 990 656 000 pixels compared, zero differences.** Foreground present in 120/120 frames, so
this is not a comparison of empty masks.

---

## 3. End to end — the v0/v1/v2 table, running for the first time

| | v0 | v1 | v2 | v2 vs v0 | MB/frame |
|---|---|---|---|---|---|
| 480p | 11.31 ms · 88.4 FPS | 3.76 · 266.0 | **3.19 · 313.5** | **3.55×** | 6.97 → 3.28 |
| 720p | 23.74 ms · 42.1 FPS | 7.06 · 141.6 | **5.86 · 170.6** | **4.05×** | 15.67 → 7.37 |
| 1080p | 66.12 ms · 15.1 FPS | 14.07 · 71.1 | **11.56 · 86.5 FPS** | **5.72×** | 35.25 → 16.59 |

### Against the proposal's targets

| target | result |
|---|---|
| **>30 FPS at 1080p on a T4** | **86.5 FPS** — met, ~2.9× the target |
| **>20× over sequential Python** | **1141×** at 480p (sequential: 3632.71 ms/frame, 0.3 FPS) |

1141× is true and flattering — that baseline is a per-pixel interpreter loop. The honest
baseline for "what did the GPU buy over competent CPU code" is Numba:

| | Numba CPU | v2 GPU | speedup |
|---|---|---|---|
| 480p | 25.43 ms · 39.3 FPS | 3.22 ms · 311.0 FPS | 7.91× |
| 720p | 56.78 ms · 17.6 FPS | 6.39 ms · 156.6 FPS | 8.89× |
| 1080p | 165.95 ms · 6.0 FPS | 15.46 ms · 64.7 FPS | **10.74×** |

(v2 reads 15.46 ms here against 11.56 ms in the table above: single pass versus median of five
interleaved passes. A 34% swing between the two is exactly why the measurement protocol takes
medians on a shared T4 — quote the 11.56 ms figure.)

---

## 4. Per-stage, v2 — and the bottleneck moved as predicted

| stage | 480p | 720p | 1080p |
|---|---|---|---|
| mask (H2D + kernels + D2H) | 1.452 ms · 33.0% | 2.017 · 32.8% | 3.790 · 30.9% |
| **host `fill_holes`** | 1.608 ms · 36.6% | 2.026 · 33.0% | **4.403 · 35.9%** |
| composite (H2D + kernels + D2H) | 1.337 ms · 30.4% | 2.103 · 34.2% | 4.074 · 33.2% |

The plan predicted: *"After it lands the frame budget should be dominated by the MOG2 kernel's
model-state traffic and the host `fill_holes` — **not** by the blur. If so, say so: the next
real win is streams, not a better blur."*

**Confirmed.** `fill_holes` is now the single largest stage at every resolution. The next win
is CUDA streams overlapping frame N+1's ingest with frame N's flood fill, not a faster kernel.

---

## 5. Kernel 0 and Kernel 2, isolated

### Blur + composite (kernels only, no transfers)

| | host `cv2` | GPU naive | GPU tiled | tiled vs host | **tiled vs naive** |
|---|---|---|---|---|---|
| 480p | 3.103 ms | 0.878 | **0.371** | 8.4× | **2.37×** |
| 720p | 6.359 ms | 1.874 | **0.793** | 8.0× | **2.36×** |
| 1080p | 13.346 ms | 3.532 | **1.488** | 9.0× | **2.37×** |

### Frame ingest, decomposed at 1080p

The "ingest is 9.6× faster" headline bundles three separate changes, only one of which is the
conversion kernel. Attributing all of it to the kernel would overstate it:

| stage | ms |
|---|---|
| A host `cvtColor` | 1.737 |
| B host float32 + transpose | 6.603 |
| C H2D 12 B/px, **allocating** (what v0 does) | 6.415 |
| D H2D 12 B/px, preallocated | 5.014 |
| E H2D 3 B/px, preallocated | 1.385 |
| F conversion kernel on device | **0.151** |

**old (A+B+C) 14.75 ms → new (E+F) 1.54 ms, 9.6×**

| cause of the 13.2 ms saved | ms | share |
|---|---|---|
| host colour convert + transpose removed | 8.34 | 63% |
| 12 B/px → 3 B/px upload | 3.63 | 28% |
| device allocation removed (v0 allocates 25 MB per frame) | 1.40 | 11% |
| conversion kernel added back | −0.15 | — |

The conversion kernel costs **0.151 ms**. Being nearly free is what made the 3 B/px upload
possible, since the conversion had to happen somewhere. The largest single win in this section
is not a kernel: it is deleting a 25 MB numpy transpose from the host.

---

## 6. Predictions, written before measuring

| | predicted | measured |
|---|---|---|
| host blur+composite @1080p | 4–8 ms | **13.3 ms** |
| GPU blur kernels @1080p | 0.2–0.4 ms | **1.49 ms** (tiled) |
| **tiled vs naive** | *"probably close to naive"* | **2.37× faster** |
| ingest+blur saved @1080p | 4–8 ms | **~25 ms** |
| bottleneck after the change | host `fill_holes`, not the blur | **confirmed, 35.9%** |
| bus traffic reduction @1080p | −38.5% | **−52.9%** (the prediction omitted v0's `bg_prob` D2H) |

**The tiling prediction was wrong, and cleanly so.** The written reasoning was that L2 already
serves row-strided reuse well, so a shared tile would buy little. It buys **2.37×**, and the
ratio is 2.37 / 2.36 / 2.37 at 480p / 720p / 1080p — far too stable to be noise. The tiled
kernels earn their complexity.

---

## 7. Not measured here

**F1 / IoU on CDnet.** `changedetection.net` no longer resolves from this VM, so the dataset
could not be fetched. Quality has to be scored where the data lives:

```bash
HIGHWAY_DIR=/path/to/highway python eval_highway.py --colorspace both
HIGHWAY_DIR=/path/to/highway python eval_highway.py --model cuda_v2 --parity-vs numba
```

The equivalence half of that gate is covered by §2 above on synthetic frames at full
resolution; what remains is the F1 number itself, which the host chain has not changed.
