# Measured on a Colab Tesla T4

Environment: Tesla T4 (15360 MiB, driver 580.82.07), compute 7.5 · Python 3.12.13 ·
OpenCV 5.0.0 (x86, **Intel IPP 2026.0.0**, pthreads) · NumPy 2.0.2 · numba 0.60.0.

All timings are medians of 7 interleaved rounds of 20 calls.

## 1. Correctness on real hardware

The kernels had only ever been verified under `NUMBA_ENABLE_CUDASIM=1`, which runs
NumPy semantics rather than compiled PTX. Re-run on the T4:

| check | shapes | mismatches |
|---|---|---|
| blur naive vs `cv2.GaussianBlur` | 32×48, 37×53, 120×160, 240×320, 480×854, **1080×1920**, 15×15, 8×9, **1×64, 64×1** | **0** |
| blur tiled vs `cv2.GaussianBlur` | same | **0** |
| composite naive/tiled vs `cv2.copyTo` | same | **0** |
| all-0, all-255, border-lit, column-0-lit | 64×64 | **0** |
| mask values {0, 1, 127, 255} | 64×64 | **0** |
| `bgr2ycrcb_planar_kernel` vs `cv2.cvtColor` | 240×320 incl. saturation rows | **0** |

`1×64` and `64×1` matter specifically: they are the `_reflect101` `n == 1` case, which
hangs forever without the early return. It ran on real hardware and terminated.

The Q8 premise was also re-derived on Colab's OpenCV, which is a **different dispatch path**
from the Apple Silicon NEON build it was developed against (IPP enabled here):

- cumulative Q8 kernel: `[9,11,15,17,19,22,23,24,23,22,19,17,15,11,9]`, sum 256 — identical
- fixed-point chain vs `cv2.GaussianBlur`: **0 / 66 654 pixels differ**
- an *ideal* float64 Gaussian vs `cv2`: **15.2%** of pixels differ, max 1 grey level
- fixed-point `cvtColor` formula vs `cv2.cvtColor`: **0 pixels differ**

Three OpenCV builds now agree: 4.13.0 (NEON), 5.0.0 (NEON), 5.0.0 (x86 + IPP).

## 2. Kernel 2 — blur + composite (kernels only, no transfers)

| | host `cv2` | GPU naive | GPU tiled | tiled vs host | **tiled vs naive** |
|---|---|---|---|---|---|
| 480p | 3.103 ms | 0.878 | **0.371** | 8.4× | **2.37×** |
| 720p | 6.359 ms | 1.874 | **0.793** | 8.0× | **2.36×** |
| 1080p | 13.346 ms | 3.532 | **1.488** | 9.0× | **2.37×** |

### The prediction that was wrong

Written down before measuring: *"tiled will probably land close to naive, because L2 already
serves row-strided reuse well. If tiled ≈ naive, that is a result to report, not a rewrite to
undo."*

**Falsified.** Shared-memory tiling is **2.37× faster than naive**, and the ratio is the same
to two decimal places at all three resolutions — 2.37, 2.36, 2.37. L2 does not absorb the
15-tap row reuse; staging the tile does. The v2 tiled kernels earn their complexity.

The other prediction — host blur at 1080p ~4–8 ms — was also wrong, in the conservative
direction: it is **13.3 ms**.

## 3. Kernel 0 — frame ingest, decomposed

The headline "ingest is 9.6× faster" bundles three separate changes, only one of which is the
conversion kernel. Attributing all of it to the kernel would overstate it, so:

| stage @1080p | ms |
|---|---|
| A host `cvtColor` | 1.737 |
| B host float32 + transpose | 6.603 |
| C H2D 12 B/px, **allocating** (what v0 does) | 6.415 |
| D H2D 12 B/px, preallocated | 5.014 |
| E H2D 3 B/px, preallocated | 1.385 |
| F conversion kernel on device | **0.151** |

**old (A+B+C) = 14.75 ms → new (E+F) = 1.54 ms, 9.6×**

Attribution of the 13.2 ms saved:

| cause | ms | share |
|---|---|---|
| host colour convert + transpose removed | 8.34 | 63% |
| 12 B/px → 3 B/px upload | 3.63 | 28% |
| device allocation removed (v0 allocates every frame) | 1.40 | 11% |
| conversion kernel added back | −0.15 | — |

The conversion kernel costs **0.151 ms** — essentially free, and that is what made the whole
transfer story possible. The largest single win is not a kernel at all: it is deleting a
25 MB numpy transpose from the host.

## 4. Combined, per frame

| | ingest + blur, before | after | saved |
|---|---|---|---|
| 480p | 6.33 ms | 0.86 ms | 5.48 ms (7.4×) |
| 720p | 12.94 ms | 1.59 ms | 11.34 ms (8.1×) |
| 1080p | **28.10 ms** | **3.03 ms** | **~25 ms (9.3×)** |

(Using the decomposed 14.75 ms ingest figure, which is the conservative one. Measuring the
old ingest as one uninterrupted loop gave 25.6 ms rather than 14.75 — repeated 25 MB device
allocations compound. The smaller number is the one quoted.)

Predicted: "4–8 ms/frame at 1080p, dominated by removing host work rather than by the kernel
being fast." The direction was right and the magnitude was not: **~25 ms**, and 63% of it is
indeed removed host work.

## Still to measure

`bench_post.py` end to end (v0/v1/v2 with the full model kernel and the flood-fill round trip),
`eval_highway.py --parity-vs`, and the >30 FPS @1080p / >20× sequential targets. Those need the
working tree pushed to a branch Colab can clone; the numbers above were produced by running the
kernels standalone.
