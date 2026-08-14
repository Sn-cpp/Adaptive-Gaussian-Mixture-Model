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

At Full HD (1920×1080) each frame holds about 2 million pixels. A Gaussian Mixture Model with K=5 components updates 5 Gaussian distributions per pixel, so roughly 10 million distribution updates per frame, and a 15×15 Gaussian blur costs 225 multiply-accumulates per background pixel. In sequential Python the whole pipeline runs at **2.8 FPS at 320×240** (measured) and far below 1 FPS at 1080p. Real time needs 30. That gap is why this problem belongs on a GPU.

**Dataset / Input:**

- **Dataset:** CDnet 2014, `baseline/highway` — 1700 frames at 320×240 of a fixed traffic camera, with per-frame hand-labelled ground truth.
- **Source:** http://changedetection.net/ (public download, no registration for the 2014 dataset).
- **Why this one:** it is publicly downloadable, and it ships pixel-accurate ground truth, a region of interest (`ROI.bmp`), and a scoring window (`temporalROI.txt` = frames 470–1700). That means every quality claim in this project is a number against a published label set, not an opinion about a screenshot.
- **Scoring protocol:** F1 and IoU over frames 470–1700, counting only pixels whose ground truth is 0 or 255 inside the ROI. CDnet labels shadows as 50 and object boundaries as 170 and defines both as *don't care*; scoring them is the easiest way to publish a wrong number, so we exclude them explicitly.
- **Benchmark sizes:** the same sequence upscaled to 854×480, 1280×720 and 1920×1080 for throughput measurement. Quality is always scored at the native 320×240, where the ground truth lives.

**Why GPU-suitable:**

1. **GMM update:** each pixel's mixture is independent — no data flows between pixels. A 1080p frame maps to 2,073,600 threads, each reading one pixel and updating K=5 components in registers.
2. **Post-processing:** thresholding is per-pixel; the median filter is a 5×5 stencil with regular access, ideal for shared-memory tiling.
3. **Gaussian blur:** a separable 2D stencil, the textbook tiling case.

Two kernel launches for the model and the mask, one for the blur, all with near-full occupancy on a T4 (2560 CUDA cores).

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
video frame (BGR)
    ↓  colour convert (YCrCb) + planar layout
[Kernel 1] MOG2 update  → mask, bg_prob      ← per-pixel, no dependencies
[Kernel 2] threshold    → binary mask        ← per-pixel  (fused into K1 in v2)
[Kernel 3] median 5×5   → refined mask       ← 5×5 stencil, shared-memory tiled
    ↓  D2H: one byte per pixel
[Host]     flood fill   → filled mask        ← inherently sequential, see §3
[Kernel 4] separable Gaussian blur ⨝ composite
    ↓
display / write
```

**Key parallelism insight:** the GMM update has no cross-pixel dependency at all, and every mask stage is either per-pixel or a small stencil. The one exception is the flood fill, and §3 explains why it stays on the host.

---

### 3. The Challenge

1. **Large per-pixel state (memory bandwidth).** K=5 Gaussians per pixel, each with weight, 3-channel mean and variance: at 1080p that is about 31M float32 values, ~120 MB of model state read and written every frame. The kernel is bandwidth-bound, so the state is stored planar — `means[k][c][y][x]` — which makes adjacent threads read adjacent addresses and lets every access coalesce.

2. **Branch divergence in the update.** Matched / not-matched / replace-weakest sends threads in a warp down different paths. We measure the cost rather than assume it.

3. **Not everything parallelises, and saying so is part of the work.** The hole fill is a scan-line flood fill. Its data-parallel equivalent, morphological reconstruction, needs one dilate per pixel of propagation distance — measured at **344 ms against 2.2 ms** for the sequential version at 1080p. Profiling says keep it on the CPU; the interesting result is the measurement, not the kernel.

4. **Host↔device transfer is the thing to optimise, not the kernel.** This is what motivates v1. In v0 the GPU computes the confidence map and copies it back so OpenCV can threshold it — float32, 4 bytes a pixel, 8 MB per frame at 1080p. Moving the threshold and median onto the device means one byte a pixel comes back instead, already refined.

5. **A mask that scores well can still be wrong.** A closing wide enough to bridge a gap in one car will also bridge the road between two cars, and F1 barely notices: in our own testing a 15×15 closing across a 10-pixel gap merged two objects while F1 stayed near 0.96. Every candidate is checked for this, not just scored.

6. **Split development environment.** macOS on Apple Silicon has no CUDA. We develop and verify on the CPU locally, run the CUDA kernels under `NUMBA_ENABLE_CUDASIM=1` for correctness, and measure on a Colab T4.

**What we hope to learn:** how to find the real bottleneck in a GPU pipeline (it was the transfer, not the arithmetic), how to keep four implementations of one algorithm bit-identical, and when a stage should *not* be moved to the GPU.

---

### 4. Resources

**Hardware:**

- **Development:** MacBook (Apple Silicon) — sequential and Numba baselines, CUDASIM correctness runs
- **GPU:** Google Colab, NVIDIA T4 (16 GB, 2560 CUDA cores, 320 GB/s)

**Software:** Python 3.12, Numba (<0.62, so that `numba.cuda` still resolves against Colab's `numba-cuda`), NumPy, OpenCV, CuPy for the RawKernel backend, pytest.

**Starting point:** from scratch. No codebase forked; the implementation follows Zivkovic (2004) and is checked against OpenCV's `BackgroundSubtractorMOG2` for bit-level agreement.

**Special machines needed:** none beyond Colab's free T4.

---

### 5. Goals and Deliverables

**75% — Minimum viable:**

- Sequential Python MOG2 + blur pipeline running end to end on `highway`
- F1/IoU scored against CDnet ground truth on frames 470–1700
- Per-stage timing at 480p
- Agreement with `cv2.createBackgroundSubtractorMOG2()`

**100% — Target:**

- Three implementations: sequential Python → Numba CPU → CUDA, all producing identical masks
- **Kernel 1** MOG2 update, one thread per pixel, planar coalesced state
- **Kernel 2** separable Gaussian blur fused with the composite, shared-memory tiled
- Benchmarks at 480p / 720p / 1080p
- **Performance target:** >30 FPS at 1080p on a T4, >20× over sequential Python
- Composite output: sharp vehicles, blurred road
- Jupyter notebook with code, explanation and benchmark charts

**125% — Stretch goals:**

- Separable blur; kernel fusion; CUDA streams overlapping transfer with compute
- YCrCb model input; adaptive K (Zivkovic)
- **Post-processing on the GPU, in three versions** — the focus of the remaining work:

| Version | What runs where | Why |
|---|---|---|
| **v0** | mask on GPU; threshold, median, fill on host (OpenCV) | baseline; copies 4 bytes/pixel back |
| **v1** | threshold + median as CUDA kernels; fill on host | mask stays resident, 1 byte/pixel returns |
| **v2** | threshold **fused** into the model kernel's epilogue; median shared-memory tiled | two kernels instead of three; each pixel read once per block instead of 25 times |

  The median is worth a note: on a binary mask a median **is a majority vote**, so the kernel counts instead of sorting, and is bit-exact with `cv2.medianBlur` rather than an approximation of it. The threshold fuses because it reads and writes one pixel; the median does not, because it needs its neighbours' post-threshold values and CUDA has no grid-wide barrier — fusing it would be a race that mostly does not show up in testing.

**Demo plan:** side-by-side Original | Mask | Blurred; live FPS counter; speedup bar chart across implementations and resolutions; per-stage timing breakdown for v0/v1/v2.

---

### 6. Evaluation

Two harnesses, both in the repository and both runnable by a marker:

- `eval_highway.py` — F1, IoU, precision, recall and empty-frame count for every candidate mask chain, on frames 470–1700 with the CDnet protocol.
- `bench_post.py` — per-stage timing for v0/v1/v2 at each resolution, with interleaved repeats, and an equality check asserting that v1 and v2 produce the *same* mask as v0. A speedup that changes the output is not a speedup.

Correctness is layered: the CUDA kernels are checked against the host chain pixel-for-pixel (`tests/test_post_chain.py`, runs under CUDASIM without a GPU), the host chain is scored against CDnet ground truth, and the MOG2 model with post-processing disabled is checked bit-for-bit against OpenCV.

---

### Weekly Schedule

| | Week 1 | Week 2 | Week 3 | Week 4 |
|---|---|---|---|---|
| **Hải Dương** | Sequential MOG2, OpenCV parity harness | Numba CPU (`prange`), profiling | CUDA model kernel, planar coalesced state | `eval_highway.py`, post-processing v1/v2 kernels, final benchmarks |
| **Đức Tín** | Sequential blur + composite | Numba blur, FPS framework | CUDA blur with shared-memory tiling | Mask-quality experiments, demo video, notebook and report |
