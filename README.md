# Adaptive Gaussian Mixture Model

Real-time background subtraction and video background blur, with multiple
backend implementations for performance comparison.

## Backends

Two families of models share one class contract and one planar state layout
(`means (K,C,H,W)`, `vars (K,H,W)`, `weights (K,H,W)`), and take planar
`(C, H, W)` float32 frames.

**Stauffer-Grimson (1999)** — fixed K, constant-alpha EMA:

| Backend | Class | Description |
|---|---|---|
| NumPy (CPU) | `GMM_CPU` | Vectorized NumPy baseline |
| Numba (CPU) | `GMM_CPU_NUMBA` | JIT-compiled with serial and `prange` parallel modes |
| CuPy vectorized (GPU) | `GMM_CUPY_V0` | CuPy array operations on GPU |
| CuPy RawKernel (GPU) | `GMM_CUPY_V1` | Custom CUDA kernels via CuPy RawKernel |

**MOG2 (Zivkovic 2004)** — a port of OpenCV `BackgroundSubtractorMOG2`:

| Backend | Class | Description |
|---|---|---|
| Python (CPU) | `GMM_CPU_MOG2` | Plain Python reference — the specification the others follow |
| Numba (CPU) | `GMM_CPU_NUMBA_MOG2` | `@njit(parallel=True)`, `prange` over rows |
| Numba CUDA (GPU) | `GMM_CUDA_MOG2` | `@cuda.jit`, model state resident on the GPU |

The MOG2 family adds a per-pixel adaptive mode count, the `k = alpha/w_new`
update, variance clamping, complexity-reduction pruning, separate `Tb`/`Tg`/`TB`
thresholds and shadow detection — which is what makes it reproduce OpenCV.
Because MOG2 fuses classification and update into one traversal, those models
expose `step()` instead of the `predict()` / `update()` pair.

## Installation

```bash
pip install -r requirements.txt
```

For the CuPy backends, install CuPy matching your CUDA version:
```bash
pip install cupy-cuda12x   # CUDA 12.x
# or
pip install cupy-cuda11x   # CUDA 11.x
```

CuPy is optional — only `GMM_CUPY_V0` / `GMM_CUPY_V1` need it. Everything else,
including `GMM_CUDA_MOG2`, imports without it.

Conda alternative:
```bash
conda env create -f environment.yml
conda activate gmm-blur
python tests/test_env.py
```

## Usage

```bash
python main.py
```

Select the backend via `--model` (0=CPU, 1=Numba, 2=CuPy V0, 3=CuPy V1, 4=MOG2 CPU, 5=MOG2 Numba,
6=MOG2 CUDA); see `python main.py --help`.

Stauffer-Grimson defaults: `K=7`, `match_threshold=3.5`, `bg_threshold=0.7`,
`alpha=0.01`. MOG2 defaults match OpenCV and live in `settings.py`.

The full blur pipeline —
`frame → MOG2 step (mask) → morphological open → separable blur ⨝ composite`:

```python
from gmm import GMM_CPU_NUMBA_MOG2          # or GMM_CUDA_MOG2
from pipeline import make_pipeline

p = make_pipeline(GMM_CPU_NUMBA_MOG2, first_frame, n_components=5)
out, mask, timings = p.process(frame_bgr)
```

The model on its own:

```python
from gmm.mog2_common import to_planar
model = GMM_CPU_NUMBA_MOG2(first_frame, n_components=5)
mask, seconds = model.step(to_planar(frame_bgr))   # planar (C, H, W) float32
```

`step(frame, match_threshold, update_alpha, weight_threshold) -> (mask, seconds)`
is the same signature every other model uses, so `main.py` and `benchmark.py`
drive them all identically. MOG2 accepts and ignores `match_threshold` /
`weight_threshold` — its `Tb` / `Tg` / `TB` come from `settings.MOG2_*`,
calibrated to reproduce OpenCV; override them via the constructor.

Knobs: `color=False` (1-channel grayscale model instead of 3-channel),
`detect_shadows`, `var_threshold`, `background_ratio`, and `update_alpha` on
`step` (negative = OpenCV's `1/min(2*nframes, history)` ramp).

On the GPU, `CUDAPipeline.process_stream(frames)` is the pipelined variant — the
upload of frame *i* overlaps the compute of frame *i-1*.

## Running Tests

```bash
pytest tests/ -v
```

GPU tests are automatically skipped when CuPy is not available.

The MOG2 suite can also be run on its own:

```bash
python tests/test_mog2_correctness.py
```

Without an NVIDIA GPU the CUDA kernels still run — on the CPU — via Numba's
simulator:

```bash
NUMBA_ENABLE_CUDASIM=1 python tests/test_mog2_correctness.py
```

Against OpenCV MOG2: bit-exact on synthetic sequences, grayscale and colour,
including `getBackgroundImage`. On real 8-bit video it is also exact on x86-64
Linux (0/2,304,000 pixels differ on Colab); on macOS arm64 ~0.002% of pixels
differ (IoU 0.99), because that OpenCV build contracts `acc += d*d` into an FMA
and rounds once where we round twice.

The three MOG2 models agree with each other on masks exactly. CPU vs CUDA state
arrays differ by at most 8e-6 on a T4, from the same FMA contraction in the GPU
kernel — well inside the 1e-5 tolerance the suite asserts.

## Project Structure

```
.
├── gmm/
│   ├── mog2_common.py                # MOG2 state, params, background image, cv2 reference
│   ├── cpu/
│   │   ├── GMM_cpu.py                # NumPy vectorized
│   │   ├── GMM_cpu_numba.py          # Numba JIT (serial + parallel)
│   │   ├── GMM_cpu_mog2.py           # MOG2 reference implementation
│   │   └── GMM_cpu_numba_mog2.py     # MOG2, prange over rows
│   └── gpu/
│       ├── GMM_cupy_v0.py            # CuPy array ops
│       ├── GMM_cupy_v1.py            # CuPy RawKernel
│       ├── GMM_cuda_mog2.py          # MOG2 on Numba CUDA
│       └── kernels/                  # CUDA .cu kernel files
├── utils/
│   ├── post_processing.py            # Morphological refinement, background blur (OpenCV)
│   ├── blur_numba.py                 # Separable blur + morphology (CPU)
│   ├── blur_cuda.py                  # Same kernels, shared-memory tiled
│   ├── benchmark.py                  # Timing and mask-quality helpers
│   └── timer.py                      # CPU/GPU timing utilities
├── tests/
│   ├── conftest.py                   # pytest fixtures
│   ├── test_correctness.py           # Cross-backend tests (Stauffer-Grimson)
│   ├── test_mog2_correctness.py      # OpenCV parity, model agreement, stability
│   └── test_env.py                   # Environment smoke test
├── notebooks/                        # Deliverable notebook
├── pipeline.py                       # Pipeline / CUDAPipeline (pinned buffers, 2 streams)
├── main.py                           # Webcam/video demo entry point
├── settings.py                       # Model constants for both families
└── requirements.txt
```

## Behaviour note: foreground persistence

With MOG2, a person who stops moving stays foreground for
`ln(TB) / ln(1 - alpha)` frames, because the *old* background mode has to decay
below the background ratio `TB = 0.9`. At the default `alpha = 1/500` that is
~53 frames (~1.8 s at 30 FPS). It is **not** `TB × history`. Lower the learning
rate to keep still subjects sharp for longer; the test suite checks this against
OpenCV at three learning rates.

## Measured results

Google Colab, **Tesla T4** + 2 vCPU, `input.mp4`, full pipeline
(model + morphology + separable blur + composite), median over 30 frames:

| Resolution | `GMM_CPU_MOG2` | `GMM_CPU_NUMBA_MOG2` | `GMM_CUDA_MOG2` | `GMM_CUDA_MOG2` streamed |
|---|---|---|---|---|
| 480p  | 0.30 FPS | 13.3 FPS | **371 FPS** | 410 FPS |
| 720p  | 0.11 FPS |  4.3 FPS | **114 FPS** | 125 FPS |
| 1080p | 0.05 FPS |  1.9 FPS | **40.2 FPS** | 37.6 FPS |

Speedup over the sequential Python reference at 1080p: 41× (Numba, 2 vCPU) and
**877×** (CUDA). Streaming buys ~10% at 480p/720p but costs ~6% at 1080p, where
compute already dominates and the extra pinned staging buffer does not pay for
itself.

Blur at 1080p — separable + shared-memory tiled vs the naive 2D convolution:
**22.6× faster on the T4** (1.56 ms vs 35.2 ms), 7.5× on CPU.

## Target

Full HD (1920×1080) at >30 FPS on an NVIDIA T4 — met, at 40.2 FPS.

## Authors

- Duc Tin (22127415) - [@Sn-cpp](https://github.com/Sn-cpp)
- Hai Duong Huynh Le (22127081) - [@haiduonghuynhle](https://github.com/haiduonghuynhle)
