# CSC14116 — Applied Parallel Programming

## RESEARCH PROPOSAL

### Real-Time Foreground Segmentation and Selective Blur for Traffic Video, via a Parallelized Gaussian Mixture Model

---

**List of members:**

| Full Name | Student ID |
|---|---|
| Huỳnh Lê Hải Dương | 22127081 |
| Nguyễn Đức Tín | 22127415 |

**Keywords:** Gaussian Mixture Model, Background Subtraction, CUDA, Numba, CDnet 2014, Real-Time Video Processing

**Repository:** https://github.com/Sn-cpp/Adaptive-Gaussian-Mixture-Model

**List of references:**

- Zivkovic, Z. (2004). Improved Adaptive Gaussian Mixture Model for Background Subtraction. *ICPR 2004*. — the algorithm we implement.
- Stauffer, C. & Grimson, W. (1999). Adaptive Background Mixture Models for Real-Time Tracking. *CVPR 1999*. — the original formulation Zivkovic improves on.
- Wang, Y. et al. (2014). CDnet 2014: An Expanded Change Detection Benchmark Dataset. *CVPR Workshops*. — the dataset and the scoring protocol.
- Barnich, O. & Van Droogenbroeck, M. (2011). ViBe: A Universal Background Subtraction Algorithm for Video Sequences. *IEEE TIP 20(6)*. — the conservative-update idea we evaluate as an option.
- OpenCV MOG2 source (reference implementation): https://github.com/opencv/opencv/blob/master/modules/video/src/bgfg_gaussmix2.cpp
- Numba CUDA documentation: https://numba.readthedocs.io/en/stable/cuda/

---

## Content

### 1. Problem Statement

**Problem:**
Separate moving objects from a static background in a video stream, then blur the background while keeping the objects sharp. This is the core of automated traffic monitoring — counting vehicles, flagging stopped cars, anonymising a scene before storage — and the segmentation step is what everything downstream depends on.

At Full HD (1920×1080) each frame holds about 2 million pixels. A Gaussian Mixture Model with up to K=5 components per pixel holds 25 float32 of state per pixel — 207 MB at 1080p — and a 15×15 Gaussian blur costs 30 multiply-accumulates per background pixel once it is separated into two passes (225 if it is not). In sequential Python the whole pipeline runs far below 1 FPS — `bench_post.py --with-sequential` measures **3035 ms/frame at 480p** on a Colab T4 host, 0.3 FPS. Real time needs 30. That gap is why this problem belongs on a GPU.

**Dataset / Input:**

- **Dataset:** CDnet 2014, `baseline/highway` — 1700 frames at 320×240 of a fixed traffic camera, with per-frame hand-labelled ground truth.
- **Source:** http://changedetection.net/ — note that as of this writing the download host no longer resolves, so quality has to be scored against a local copy (`HIGHWAY_DIR=...`).
- **Why this one:** it ships pixel-accurate ground truth, a region of interest (`ROI.bmp`), and a scoring window (`temporalROI.txt` = frames 470–1700). That means every quality claim in this project is a number against a published label set, not an opinion about a screenshot.
- **Scoring protocol:** F1 and IoU over frames 470–1700, counting only pixels whose ground truth is 0 or 255 inside the ROI. CDnet labels shadows as 50 and object boundaries as 170 and defines both as *don't care*; scoring them is the easiest way to publish a wrong number, so we exclude them explicitly.
- **Benchmark sizes:** the same sequence upscaled to 854×480, 1280×720 and 1920×1080 for throughput measurement. Quality is always scored at the native 320×240, where the ground truth lives.

**Why GPU-suitable:**

1. **GMM update:** each pixel's mixture is independent — no data flows between pixels. A 1080p frame maps to 2,073,600 threads, each reading one pixel and updating K=5 components in registers.
2. **Post-processing:** thresholding is per-pixel; the median filter is a 5×5 stencil with regular access, ideal for shared-memory tiling.
3. **Gaussian blur:** a separable 2D stencil, the textbook tiling case.

Two kernel launches for the model and the mask, and two for the blur — separable, so the horizontal and vertical passes cannot be fused without a grid-wide barrier. All one thread per pixel on a T4 (2560 CUDA cores), which is enough threads to fill it several times over; we have not profiled achieved occupancy and do not claim a figure for it.

---

### 2. Background

**Adaptive Gaussian Mixture Model (Zivkovic, 2004):**

Each pixel keeps up to K Gaussians. For each incoming frame the pixel value is compared against them; on a match within a threshold, that Gaussian's mean and variance are updated by an exponential moving average with learning rate α, and its weight is increased. On no match, the weakest Gaussian is replaced by a new one centred on the current value. Gaussians are kept sorted by weight; the leading ones whose cumulative weight passes `TB` form the background model. Zivkovic adds a complexity-reduction term that prunes components a pixel does not need, so K adapts per pixel.

**Per-pixel GMM update pseudocode:**

```
for each pixel p:
    matched = false
    bg_weight = 0
    for k in 0..K-1:
        w[k] = (1-α)·w[k] − α·CT               # decay + prune term
        d2   = ||x − μ[k]||²
        if d2 < Tb·σ²[k]:                      # inside the background set
            bg_weight += w[k]
        if not matched and d2 < Tg·σ²[k]:      # this component explains x
            matched = true
            w[k] += α
            ρ     = α / w[k]
            μ[k] += ρ·(x − μ[k])
            σ²[k] = clamp(σ²[k] + ρ·(d2 − σ²[k]), σ²_min, σ²_max)
        if w[k] < prune: drop component k
    if not matched: replace the weakest component with (x, σ²_init, α)
    normalise w; sort by weight
    mask[p]    = background ? 0 : 255
    bg_prob[p] = bg_weight                     # confidence, kept for the mask stage
```

**Colour space.** The model can read BGR or YCrCb. This is not cosmetic: measured on `highway` with the identical downstream chain, BGR scores **F1 0.827** and YCrCb **F1 0.984**. Separating brightness from colour lets a car's shadow — which changes luma but not chroma — fail the match without dragging the car with it. YCrCb is the default.

**Mask post-processing.** Raw MOG2 output is speckled and hollow: sensor noise flips single pixels, and inside a large uniform surface (a car roof) motion changes nothing, so the middle comes back empty. Our chain is threshold → median 5 → flood-fill the holes.

**Pipeline architecture:**

```
video frame (BGR uint8)
    ↓  H2D: three bytes per pixel — the only thing that goes up
[Kernel 0] BGR→YCrCb + planar layout         ← per-pixel, bit-exact with cv2.cvtColor
[Kernel 1] MOG2 update  → mask, bg_prob      ← per-pixel, no dependencies
[Kernel 2] threshold    → binary mask        ← per-pixel  (fused into K1 in v2)
[Kernel 3] median 5×5   → refined mask       ← 5×5 stencil, shared-memory tiled
    ↓  D2H: one byte per pixel
[Host]     flood fill   → filled mask        ← sequential in OpenCV; see §3
    ↑  H2D: one byte per pixel
[Kernel 4] separable Gaussian blur, horizontal   ← 15-tap, Q8 fixed point
[Kernel 5] vertical pass ⨝ composite         ← the select is free here
    ↓  D2H: three bytes per pixel
display / write
```

**Key parallelism insight:** the GMM update has no cross-pixel dependency at all, and every mask stage is either per-pixel or a small stencil. The one exception is the flood fill, and §3 explains why it stays on the host.

---

### 3. The Challenge

1. **Large per-pixel state (memory bandwidth).** K=5 Gaussians per pixel, each with a weight, a 3-channel mean and a variance — 5 + 15 + 5 = **25 float32 per pixel**. At 1080p that is 51.8M values, **207 MB** of model state resident and touched every frame. (Resident: the kernel loops over each pixel's *active* components, so it does not rewrite all of it.) (An earlier draft said 31M/120 MB, having counted the means and forgotten the weights and variances; the arrays are `means (K,C,H,W)`, `vars (K,H,W)`, `weights (K,H,W)` and the figure above is their `nbytes`.) That arithmetic intensity is low enough that we expect the kernel to be bandwidth-bound rather than compute-bound — an expectation from the byte count, not a roofline measurement — so the state is stored planar, `means[k][c][y][x]`, which makes adjacent threads read adjacent addresses and lets every access coalesce.

2. **Branch divergence in the update.** Matched / not-matched / replace-weakest sends threads in a warp down different paths. We have not isolated its cost with a profiler, and say so rather than quoting a figure; what we can report is the end-to-end effect, which is in `RESULTS-T4.md`.

3. **Not everything parallelises usefully, and saying so is part of the work.** OpenCV implements the hole fill as a scan-line flood fill, which is sequential. A data-parallel formulation exists — morphological reconstruction by dilation — so the question is whether it is worth it, not whether it is possible. `bench_fill.py` implements both, checks they agree pixel-for-pixel, and times them: at 1080p the reconstruction needs **593 full-frame dilate passes**, and the wall-clock ratio is 112× on one host and 287× on another *while the pass count is identical on both*. That is the durable result — milliseconds belong to the machine, the pass count belongs to the algorithm. Each pass is a grid-wide step, and no amount of GPU shortens the *sequence* of them.

4. **Host-side work and host↔device transfer are the things to optimise, not the kernel.** This is what motivates v1. In v0 the GPU computes the confidence map and copies it back so OpenCV can threshold it — float32, 4 bytes a pixel, 8 MB per frame at 1080p. Moving the threshold and median onto the device means one byte a pixel comes back instead, already refined.

5. **A mask that scores well can still be wrong.** A closing wide enough to bridge a gap in one car also bridges the road between two cars, and the summary barely notices: in the scored table a 15×15 closing moves F1 by 0.006 while precision falls 0.9863 → 0.9634. Every candidate is looked at, not just scored. (The specific two-cars-merged observation was an eyeball check during that run; it is not reproducible from this checkout.)

6. **Split development environment.** macOS on Apple Silicon has no CUDA. We develop and verify on the CPU locally, run the CUDA kernels under `NUMBA_ENABLE_CUDASIM=1` for correctness, and measure on a Colab T4.

**What we hope to learn:** how to find the real bottleneck in a GPU pipeline (it was the host, not the arithmetic — 76% of the ingest saving was host work deleted, only 17% the smaller upload), how to keep four implementations of one algorithm bit-identical, and when a stage should *not* be moved to the GPU.

---

### 4. Resources

**Hardware:**

- **Development:** MacBook (Apple Silicon) — sequential and Numba baselines, CUDASIM correctness runs
- **GPU:** Google Colab, NVIDIA T4 (16 GB, 2560 CUDA cores, 320 GB/s)

**Software:** Python 3.12, Numba (<0.62, so that `numba.cuda` still resolves against Colab's `numba-cuda`), NumPy, OpenCV, CuPy for the RawKernel backend, pytest.

**Starting point:** from scratch. No codebase forked; the implementation follows Zivkovic (2004) and is checked against OpenCV's `BackgroundSubtractorMOG2` for bit-level agreement.

**Special machines needed:** none beyond Colab's free T4.

---

### 4b. Measured CPU Baseline

`python src/cpu_baseline.py` — the course-template entry point, runnable with no GPU and no
dataset. Output pasted verbatim (Apple M-series host, Python 3.12, OpenCV 5.0):

```
Input shape: (12, 240, 320, 3), dtype: uint8
Input size: 2.8 MB

CPU baseline results:
  Time:       4.047 s  (337.2 ms/frame)
  Agreement with cv2.MOG2: 1.000000
  Throughput: 2.97 frames/s

OK — this is the reference every GPU version is tested against.
```

`python benchmarks/profile_cpu.py` puts **99.8% of that time in `mog2_step`** — the
reproducible bottleneck analysis behind the first kernel (the model). The later kernels are
justified by the per-stage pipeline profile in `RESULTS-T4.md` §4, taken after the model had
moved and the bottleneck had shifted to the host stages. (On the Colab T4
host the same sequential pipeline measures 3035 ms/frame at 854×480; see `RESULTS-T4.md`.)

### 5. Goals and Deliverables

**75% — Minimum viable:**

- Sequential Python MOG2 + blur pipeline running end to end on `highway`
- F1/IoU scored against CDnet ground truth on frames 470–1700
- Per-stage timing at 480p
- Agreement with `cv2.createBackgroundSubtractorMOG2()`

**100% — Target:**

- Three implementations: sequential Python → Numba CPU → CUDA, all producing identical masks
- **Kernel 1** MOG2 update, one thread per pixel, planar coalesced state
- **Kernel 2** separable Gaussian blur fused with the composite, shared-memory tiled, in Q8 fixed point so it is bit-exact with `cv2.GaussianBlur` rather than approximately Gaussian
- Benchmarks at 480p / 720p / 1080p
- **Performance target:** >30 FPS at 1080p on a T4, >20× over sequential Python
- Composite output: sharp vehicles, blurred road
- Jupyter notebook with code, explanation and benchmark charts

**How this maps to the course's optimization ladder** (Project Description, Part 1.2):
Level 0 = `GMM_Mask_CPU`; Level 1 (naive GPU port, "correct, not fast") = v0; Level 2 (memory:
coalesced planar state, 3 B/px ingest, shared-memory tiling, 35.25→16.59 MB bus) = v1 and v2's
tiled kernels; Level 3 (compute: kernel fusion) = v2's fused threshold; the Q8 integer blur is
the correctness story and ships in v1 and v2 alike; Level 4 (streams) is identified by
measurement as the next step and deliberately not claimed.

**125% — Stretch goals:**

- Separable blur; kernel fusion; CUDA streams overlapping transfer with compute
- YCrCb model input; adaptive K (Zivkovic)
- **Post-processing on the GPU, in three versions** — the focus of the remaining work:

| Version | What runs where | Why |
|---|---|---|
| **v0** | mask on GPU; colour convert, threshold, median, fill, blur, composite on host | baseline; uploads 12 bytes/pixel and copies 5 back |
| **v1** | colour convert, threshold, median, blur and composite as CUDA kernels; fill on host | the frame goes up as 3 bytes/pixel instead of 12, and the confidence map stops coming back; the mask still makes a round trip for the host fill |
| **v2** | threshold **fused** into the model kernel's epilogue; median and blur shared-memory tiled | fewer launches; each pixel read once per block instead of 25 (median) or 15 (blur) times |

  Measured bus traffic per frame at 1080p, computed by `bench_post.py` from the array shapes
  rather than quoted: **35.25 MB → 16.59 MB, a 2.12× reduction.**

  Measured on a Colab T4 (full numbers and reproduction commands in `RESULTS-T4.md`):
  **88.8 FPS at 1080p**, 4.89× over v0 and 10.25× over Numba CPU, with v0, v1 and v2 producing
  byte-identical masks *and* composites over 1.49 billion pixel positions. Frame ingest goes
  **23.21 ms → 1.65 ms** and blur+composite **13.74 ms → 1.55 ms**; three quarters of the
  ingest saving is host work deleted rather than kernel speed, since the conversion kernel
  itself costs 0.184 ms. Two results worth stating because they were predicted wrongly:
  shared-memory tiling was expected to barely beat the naive blur, and wins **2.36×** (2.36 /
  2.36 / 2.37 across the three resolutions, and again under a second protocol); and after the change the largest single stage is
  the host flood fill at 39.7% (1080p), not any kernel.

  The median is worth a note: on a binary mask a median **is a majority vote**, so the kernel counts instead of sorting, and is bit-exact with `cv2.medianBlur` rather than an approximation of it. The threshold fuses because it reads and writes one pixel; the median does not, because it needs its neighbours' post-threshold values and CUDA has no grid-wide barrier — fusing it would be a race that mostly does not show up in testing.

**Demo plan:** side-by-side Original | Mask | Blurred; live FPS counter; speedup bar chart across implementations and resolutions; per-stage timing breakdown for v0/v1/v2.

---

### 6. Evaluation

Two harnesses, both in the repository and both runnable by a marker:

- `eval_highway.py` — F1, IoU, precision, recall and empty-frame count for every candidate mask chain, on frames 470–1700 with the CDnet protocol.
- `bench_post.py` — per-stage timing for v0/v1/v2 at each resolution, with interleaved repeats, and an equality check asserting that v1 and v2 produce the *same* mask as v0. A speedup that changes the output is not a speedup.

- `eval_highway.py --model X --parity-vs Y` — per-frame `np.array_equal` between two backends over the whole scored window. This is the equivalence gate, and an unchanged F1 is deliberately *not* accepted as one: F1 is a four-decimal summary of every scored pixel in 1231 frames and a mask can move by hundreds of them without shifting it.

Correctness is layered, and each layer is a file a marker can run:

- `tests/test_parity.py` — sequential Python == Numba == v0 == v1 == v2, compared on **every frame** for the mask and `bg_prob`, and on the model state (weights, means, variances, active-component counts) for all five; plus agreement with `cv2.createBackgroundSubtractorMOG2`, and a CuPy comparison that runs where CuPy is installed.
- `tests/test_blur.py` — the Q8 blur, the colour conversion, the borders and the composite, each against OpenCV with **zero tolerance**; plus the device ingest against the host ingest end to end.
- `tests/test_post_chain.py` — the threshold and median kernels against the host chain, plus a check that every kernel compiles on real hardware (deliberately skipped under CUDASIM: the simulator cannot fail it, which is the point).
- `tests/test_scoring.py` — the CDnet protocol itself, on a fixture whose TP/FP/FN are countable by hand. Shadows (50), unknown boundaries (170) and out-of-ROI pixels must all be excluded; the fixture is built so that including them visibly changes the score.

All of these run without a GPU under `NUMBA_ENABLE_CUDASIM=1`, except the compile check, which is skipped there on purpose.

**Measured agreement with OpenCV's own MOG2:** bit-identical on synthetic sequences (0 of 30 720 pixels over 20 frames; 0 of 92 160 under heavy noise; 0 of 204 800 at 64×80), and 22 pixels of 1 536 000 — 0.0014%, all in a single frame — on real video. The residue is the float32 boundary: OpenCV accumulates in a different order and contracts its own FMAs, so a pixel within an ulp of `Tb·σ²` falls on either side of the comparison. Synthetic frames put almost nothing that close to the threshold; camera noise does. We report both numbers rather than the flattering one.

The blur is a stronger claim than the model, and worth separating: it is integer arithmetic end to end, so its equality with `cv2.GaussianBlur` holds independently of GPU architecture and compiler flags, where the float32 MOG2 kernel's parity depends on how FMAs contract.

---

### 7. Risks

1. **The dataset host disappears.** It did — the CDnet download host
   (`wordpress-jodoin.dmi.usherb.ca`) stopped resolving mid-project, and `changedetection.net`
   now serves only an HTML landing page.
   Mitigation, applied: every quality figure carries the commit it was measured at,
   `tests/test_scoring.py` pins the CDnet protocol itself on a hand-checkable fixture, every
   pipeline runs end-to-end on synthetic frames with the source labelled, and `eval_highway.py`
   re-scores in one command the moment a local copy exists.

2. **The CUDA simulator passes code that real hardware rejects.** It did — a tiled median with a
   non-constant shared-memory shape was green under CUDASIM for weeks and had never compiled on
   a GPU. Mitigation, applied: `test_every_kernel_actually_compiles_on_real_hardware`, which
   *skips under the simulator by design* and forces every kernel through nvvm on the T4.

3. **Floating-point parity drifts across OpenCV builds and GPU architectures.** Partially
   materialised: the float32 model disagrees with cv2 by 22 px of 1.5M on real video.
   Mitigation, applied: the blur and colour conversion are integer end-to-end (bit-exact on four
   OpenCV builds, ARM and x86), the model's residual is measured and reported rather than
   rounded away, and parity is a per-frame `np.array_equal` gate, not an F1 comparison.

### Division of Labour (as delivered)

The weekly plan this section originally held assigned the model to Hải Dương
and the blur, notebook and report to Đức Tín. The split changed as the work
progressed, and this table records what actually happened — it matches the
git history, which is the honest reference for who wrote what.

| | Delivered |
|---|---|
| **Đức Tín** | The background model, end to end: sequential Python reference, the Numba CPU version, and the first CUDA model kernel (`gmm_mask/cpu/*`, `gmm_mask/gpu/gmm_mask_cuda.py`). Push-relabel and GrabCut experiments (retained on branch `dev/HD`). |
| **Hải Dương** | Post-processing on the GPU (v1/v2), Kernel 2 (the Q8 blur, colour conversion, composite), the test suite, the measurement harnesses (`bench_post.py`, `bench_t4.py`, `bench_fill.py`, `eval_highway.py`), the notebook and the reports. |
