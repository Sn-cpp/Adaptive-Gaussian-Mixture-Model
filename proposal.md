# CSC14116 — Applied Parallel Programming

## RESEARCH PROPOSAL

### Real-Time Video Background Blur via Parallelized Gaussian Mixture Model

---

**Group name:** `<your group's name>`

**List of members:**

| Full Name | Student ID |
|---|---|
| Dương Huỳnh Lê Hải | `<ID>` |
| `<Teammate Name>` | `<ID>` |

**Keywords:** Gaussian Mixture Model, Background Subtraction, CUDA, Numba, Real-Time Video Processing

**List of references:**

- Stauffer, C. & Grimson, W. (1999). Adaptive Background Mixture Models for Real-Time Tracking. *CVPR 1999*.
- Zivkovic, Z. (2004). Improved Adaptive Gaussian Mixture Model for Background Subtraction. *ICPR 2004*.
- KaewTraKulPong, P. & Bowden, R. (2002). An Improved Adaptive Background Mixture Model for Realtime Tracking with Shadow Detection. *Video-Based Surveillance Systems*.
- Numba CUDA documentation: https://numba.readthedocs.io/en/stable/cuda/
- OpenCV MOG2 source (reference implementation): https://github.com/opencv/opencv/blob/master/modules/video/src/bgfg_gaussmix2.cpp

---

## Content

### 1. Problem Statement

**Problem:**
Real-time video background blur — the feature used by Zoom, Microsoft Teams, and Google Meet — requires per-frame foreground/background segmentation followed by selective Gaussian blurring of the background region. At Full HD resolution (1920×1080), each frame contains approximately 2 million pixels. A Gaussian Mixture Model with K=5 components requires updating 5 Gaussian distributions per pixel (10 million distribution updates per frame), and a 15×15 Gaussian blur kernel requires 225 multiply-accumulate operations per background pixel. In pure sequential Python, this pipeline achieves <1 FPS — far below the 30 FPS minimum required for real-time video. This makes GPU parallelization essential.

**Dataset / Input:**

- **Dataset name and source:** Live webcam feed via OpenCV `cv2.VideoCapture(0)`, or pre-recorded 1080p MP4 test videos (e.g., standard surveillance datasets from changedetection.net or self-recorded clips).
- **Input size for benchmarking:** Single video frames at three resolutions: 640×480 (VGA), 1280×720 (HD), and 1920×1080 (Full HD). We benchmark per-frame latency and throughput (FPS) at each resolution.
- **How we will load it:** OpenCV (`cv2.VideoCapture`) for frame extraction → convert to NumPy arrays → transfer to GPU via Numba `cuda.to_device()`.

**Why GPU-suitable:**

Both core operations exhibit embarrassingly parallel data access patterns:

1. **GMM Update:** Each pixel's mixture model is independent — no data flows between pixels during the update step. A 1080p frame maps to 2,073,600 independent threads, each reading one pixel value and updating K=5 Gaussian components in private registers/local memory.
2. **Gaussian Blur:** A 2D stencil computation with regular, predictable memory access. Each output pixel reads a local (15×15) neighborhood, making it ideal for shared memory tiling to reduce redundant global memory reads. The blur applies only to background pixels (identified by the GMM mask), but we launch threads for all pixels to maintain uniform warp execution.

Total parallelism per frame: **~2M threads** for GMM + **~2M threads** for blur = two kernel launches with massive occupancy on modern GPUs (e.g., T4 with 2560 CUDA cores).

---

### 2. Background

This project implements two classic computer vision algorithms and composes them into a real-time video processing pipeline:

**Adaptive Gaussian Mixture Model (Stauffer & Grimson, 1999):**

Each pixel is modeled as a mixture of K Gaussians (K=3–5). For each incoming frame, every pixel value is compared against its K distributions. If a match is found (within 2.5 standard deviations), the matched distribution's parameters are updated via an exponential moving average (learning rate α ≈ 0.01). If no match, the weakest distribution is replaced. Distributions are ranked by weight/σ; the first B distributions whose cumulative weight exceeds threshold T form the background model. Pixels not matching any background distribution are classified as foreground.

**Per-pixel GMM update pseudocode:**

```
for each pixel (x, y):
    value = frame[y, x]
    matched = False
    for k = 0 to K-1:
        if |value - means[y,x,k]| < 2.5 * sqrt(variances[y,x,k]):
            # Update matched Gaussian
            rho = alpha / weights[y,x,k]
            means[y,x,k]    += rho * (value - means[y,x,k])
            variances[y,x,k] += rho * ((value - means[y,x,k])^2 - variances[y,x,k])
            weights[y,x,k]   = (1 - alpha) * weights[y,x,k] + alpha
            matched = True
        else:
            weights[y,x,k] *= (1 - alpha)
    if not matched:
        # Replace least probable Gaussian
        replace weakest distribution with (weight=alpha, mean=value, variance=initial_var)
    # Normalize weights
    # Classify: foreground if no background Gaussian matched
```

**Gaussian Blur:**

A 2D convolution with a Gaussian kernel of size N×N (N=15 for visible blur at 1080p). The kernel is separable (can decompose into two 1D passes for O(N) per pixel instead of O(N²)), but we implement the full 2D version first for clarity, then optimize. Blur is applied only to background-classified pixels; foreground pixels retain their original (sharp) values.

**Pipeline architecture:**

```
Webcam → BGR frame → Grayscale
    ↓
[Kernel 1] GMM update → foreground_mask (0/255)
    ↓
[Kernel 2] Gaussian blur (background only)
    ↓
[Composite] foreground (sharp) + background (blurred)
    ↓
Display / write video
```

**Key parallelism insight:** Both the GMM update and the Gaussian blur operate on each pixel independently. There are no cross-pixel data dependencies in the GMM update, and the blur's read-only stencil access pattern is trivially parallelizable with shared memory tiling for the input neighborhood.

---

### 3. The Challenge

Several aspects make this project non-trivial to parallelize effectively:

1. **Large per-pixel state (memory bandwidth):** The GMM model stores K=5 Gaussians per pixel, each with (weight, mean, variance). For 1080p, this is `1920 × 1080 × 5 × 3 = 31M float32 values` (~120 MB). The kernel is memory-bandwidth-bound, not compute-bound — global memory access patterns and coalescing are critical for performance.

2. **Branching in GMM update:** The matching logic (`if matched / if not matched / replace weakest`) causes warp divergence on GPU. Different pixels may follow different code paths depending on their history, reducing SIMT efficiency. We will investigate the performance impact and consider branchless alternatives.

3. **Halo regions in Gaussian blur:** A 15×15 blur kernel means each thread needs to read a neighborhood extending 7 pixels beyond its own position. Shared memory tiling requires loading halo pixels and handling boundary conditions (clamping or reflection), which adds complexity to the kernel and increases shared memory usage per block.

4. **Host↔Device transfer overhead:** At 1080p, each grayscale frame is ~2 MB. The GMM state arrays (~120 MB) should remain on the GPU across frames (persistent allocation), but each new frame must be transferred from host to device. At 30 FPS, this is 60 MB/s of transfer — manageable on PCIe but not negligible. We will use pinned memory and explore CUDA streams to overlap transfer with computation.

5. **Foreground mask quality:** The raw GMM output is noisy (salt-and-pepper noise, incomplete foreground regions). Morphological operations (erosion/dilation) are needed for cleanup — these are also stencil operations and can be parallelized, but add another kernel launch to the pipeline.

6. **Development environment constraint:** macOS with Apple Silicon has no CUDA support. We develop and debug the CPU-parallel version locally, then test CUDA kernels on Google Colab (T4 GPU). This split-environment workflow requires careful code organization.

**What we hope to learn:** How to identify and resolve memory bandwidth bottlenecks in GPU kernels, the practical impact of warp divergence from conditional branching, and how to use shared memory tiling for stencil computations in Numba CUDA.

---

### 4. Resources

**Hardware:**

- **Development:** MacBook (Apple Silicon M-series) — for CPU sequential and Numba parallel baselines
- **GPU testing:** Google Colab with NVIDIA T4 GPU (16 GB VRAM, 2560 CUDA cores, 320 GB/s memory bandwidth)

**Software:**

- Python 3.11, Numba 0.65, NumPy 2.4, OpenCV 4.13
- Jupyter Notebook for the final deliverable
- Conda for environment management

**Starting point:** We are starting from scratch. No existing codebase is being forked or adapted. The implementation follows the Stauffer-Grimson algorithm as described in the original paper.

**Reference materials:**

- Stauffer, C. & Grimson, W. (1999). *Adaptive Background Mixture Models for Real-Time Tracking.* CVPR. — The original algorithm we implement.
- Zivkovic, Z. (2004). *Improved Adaptive Gaussian Mixture Model for Background Subtraction.* ICPR. — Improved version (adaptive K), used for comparison.
- OpenCV `BackgroundSubtractorMOG2` source code — C++ reference for correctness verification.
- Numba CUDA documentation (https://numba.readthedocs.io/en/stable/cuda/) — API reference for kernel implementation.
- NVIDIA CUDA C Best Practices Guide, Chapter 9 (Memory Optimization) — shared memory tiling patterns.

**Special machines needed:** Google Colab (free tier with T4 GPU) is sufficient. No additional hardware is required.

---

### 5. Goals and Deliverables

**75% — Minimum viable (if behind schedule):**

- Working sequential Python/NumPy GMM + Gaussian blur pipeline
- Process pre-recorded video with correct foreground segmentation
- FPS measurement and per-stage timing breakdown at 480p
- Comparison with OpenCV `cv2.createBackgroundSubtractorMOG2()` for correctness

**100% — Target (expected outcome):**

- Three complete implementations: Sequential → Numba CPU Parallel → CUDA GPU
- CUDA Kernel 1 (GMM update): one thread per pixel, basic global memory access
- CUDA Kernel 2 (Gaussian blur): one thread per pixel, shared memory tiling for the blur window
- Speedup benchmarks at 480p, 720p, and 1080p resolutions
- **Performance target:** >30 FPS at 1080p on T4 GPU, representing >20× speedup over sequential Python
- Composite output: sharp foreground + blurred background, visually comparable to Zoom/Teams
- Complete Jupyter notebook as the deliverable with code, explanations, and benchmark charts

**125% — Stretch goals (if ahead of schedule):**

- Separable Gaussian blur (two 1D passes instead of one 2D pass) for additional speedup
- Kernel fusion: combine GMM update + mask output in a single kernel launch
- CUDA streams: overlap host→device frame transfer with previous frame's computation
- Color-space GMM: use YCrCb (3-channel) instead of grayscale for better segmentation quality
- Morphological post-processing kernel (erosion + dilation) for mask cleanup
- Adaptive K (Zivkovic, 2004): dynamically adjust the number of Gaussians per pixel

**Demo plan:**

- Side-by-side live video display: Original | Foreground Mask | Blurred Output
- Real-time FPS counter overlay on each implementation
- Speedup bar chart: Sequential vs CPU Parallel vs GPU at each resolution
- Per-kernel timing breakdown (GMM update, blur, composite, transfer)

---

### Weekly Schedule

| | Week 1 (Jul 1 – Jul 6) | Week 2 (Jul 7 – Jul 13) | Week 3 (Jul 14 – Jul 20) | Week 4 (Jul 21 – Jul 27) |
|---|---|---|---|---|
| **Member 1** | Proposal, env setup, sequential GMM implementation | Numba CPU parallel GMM (`prange`), profiling with `cProfile` | CUDA GMM kernel (V1: naive → V2: coalesced access), benchmark | Optimization, full pipeline integration, final benchmarks, report |
| **Member 2** | Proposal, sequential Gaussian blur + composite function | Numba CPU parallel blur, FPS measurement framework | CUDA blur kernel with shared memory tiling, morphological post-processing | Demo video recording, notebook finalization, report writing |
