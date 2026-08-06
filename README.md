# Adaptive Gaussian Mixture Model

Real-time background subtraction and video background blur, with multiple
backend implementations for performance comparison.

## Backends

Every model is a from-scratch port of OpenCV `BackgroundSubtractorMOG2`
(Zivkovic 2004). They share one class contract and one planar state layout
(`means (K,C,H,W)`, `vars (K,H,W)`, `weights (K,H,W)`), and take planar
`(C, H, W)` float32 frames.

| Backend | Class | Description |
|---|---|---|
| Python (CPU) | `GMM_CPU` | Plain Python reference — the specification the others follow |
| Numba (CPU) | `GMM_CPU_NUMBA` | `@njit(parallel=True)`, `prange` over rows |
| Numba CUDA (GPU) | `GMM_CUDA` | `@cuda.jit`, model state resident on the GPU |
| CuPy RawKernel (GPU) | `GMM_CUPY` | Same algorithm through a hand-written `.cu` kernel |

MOG2 keeps a per-pixel adaptive mode count, the `k = alpha/w_new` update,
variance clamping, complexity-reduction pruning, separate `Tb`/`Tg`/`TB`
thresholds and shadow detection — which is what makes it reproduce OpenCV.
Classification and update are fused into one traversal, so every model exposes a
single `step()`. `GMM_CUDA` and `GMM_CUPY` implement the same kernel through two
independent toolchains and are pinned to each other by the test suite.

An earlier Stauffer-Grimson (1999) family (`GMM_CPU`/`GMM_CPU_NUMBA`/
`GMM_CUPY_V0`/`GMM_CUPY_V1` before commit `a6b1d0f`) was retired: its NumPy
baseline was vectorized (not a sequential reference), and its backends had
drifted apart algorithmically. The git history keeps it.

## Installation

```bash
pip install -r requirements.txt
```

For the CuPy backend, install CuPy matching your CUDA version:
```bash
pip install cupy-cuda12x   # CUDA 12.x
# or
pip install cupy-cuda11x   # CUDA 11.x
```

CuPy is optional — only `GMM_CUPY` needs it. Everything else, including
`GMM_CUDA`, imports without it.

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

Select the backend via `--model` (0=CPU, 1=Numba, 2=CUDA, 3=CuPy RawKernel);
see `python main.py --help`. `--colorspace ycrcb` runs the model in YCrCb while
the display stays BGR — on CDnet `highway` this lifts mask F1 from 0.73 to 0.86
(shadow detection on) by separating luma from chroma.

Model defaults match OpenCV and live in `settings.py`.

The full blur pipeline —
`frame → MOG2 step (mask) → morphological open → separable blur ⨝ composite`:

```python
from gmm import GMM_CPU_NUMBA          # or GMM_CUDA
from pipeline import make_pipeline

p = make_pipeline(GMM_CPU_NUMBA, first_frame, n_components=5)
out, mask, timings = p.process(frame_bgr)
```

The model on its own:

```python
from gmm.mog2_common import to_planar
model = GMM_CPU_NUMBA(first_frame, n_components=5)
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
│   │   ├── GMM_cpu.py                # Plain-Python reference implementation
│   │   └── GMM_cpu_numba.py          # Numba JIT, prange over rows
│   └── gpu/
│       ├── GMM_cuda.py               # Numba CUDA, state resident on GPU
│       ├── GMM_cupy.py               # CuPy RawKernel
│       └── kernels/                  # CUDA .cu kernel files
├── utils/
│   ├── post_processing.py            # Morphological refinement, background blur (OpenCV)
│   ├── blur_numba.py                 # Separable blur + morphology (CPU)
│   ├── blur_cuda.py                  # Same kernels, shared-memory tiled
│   ├── benchmark.py                  # Timing and mask-quality helpers
│   └── timer.py                      # CPU/GPU timing utilities
├── tests/
│   ├── conftest.py                   # pytest fixtures + cupy mock
│   ├── test_mog2_correctness.py      # OpenCV parity, model agreement, stability
│   ├── test_smoke_models.py          # Every model through main.py's call shape
│   ├── test_post_processing.py       # Mask refinement + blur composite
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
(model + morphology + separable blur + composite) over 30 frames. A Colab T4 is
shared hardware and the spread is large: across 11 separate 1080p measurements
the CUDA path ranged **30.2 to 45.0 FPS**. Every figure below is therefore the
**median of repeated runs**, never a single one — treat any single benchmark
from this notebook as ±20%.

| Resolution | `GMM_CPU` | `GMM_CPU_NUMBA` | `GMM_CUDA` | `GMM_CUDA` streamed |
|---|---|---|---|---|
| 480p  | 0.31 FPS | 13.2 FPS | 358 FPS | **376 FPS** |
| 720p  | 0.10 FPS |  4.4 FPS | 103 FPS | **126 FPS** |
| 1080p | 0.047 FPS |  1.9 FPS | 34.1 FPS | **37.0 FPS** |

Speedup over the sequential Python reference: 41× (Numba, 2 vCPU) and **732×**
(CUDA) at 1080p; up to 1150× at 480p. Streaming helps at every resolution — most
at 720p (+23%) — and it is also markedly more stable: across 5 repeats at 1080p
the synchronous path ranged 30.2–38.1 FPS while the streamed path stayed within
37.3–38.6.

### Where the GPU frame actually goes

Per-stage GPU time, CUDA events between stages, 20 repeats (ms/frame):

| Resolution | H→D upload | GMM update | morphology | blur + composite | D→H readback | total |
|---|---|---|---|---|---|---|
| 480p  | 0.632 | 0.120 | 0.044 | 0.150 | 0.108 | 1.055 |
| 720p  | 1.412 | 0.336 | 0.117 | 0.419 | 0.293 | 2.577 |
| 1080p | 2.879 | 0.748 | 0.256 | 0.927 | 0.643 | 5.453 |

The upload alone is **53–60%** of GPU time and the readback another 10–12%, so
about two thirds of the frame is PCIe traffic and only a third is the three
kernels. That is the measurement the streaming stretch goal rests on:
`process_stream` overlaps exactly those rows with the previous frame's compute.

It also shows the pipeline is **host-bound end to end**. At 1080p the stages
above sum to ~5.5 ms (~180 FPS), while `process` measures ~42 FPS (~24 ms). The
missing ~18 ms is host-side — the `cv2` colour conversion and the copies into the
pinned staging buffers, on 2 Colab vCPUs. Moving the colour conversion into a
kernel is the obvious next optimisation; it is kept on the host deliberately so
every backend sees bit-identical model input and the CPU/GPU parity tests stay
meaningful.

Blur at 1080p — separable + shared-memory tiled vs the naive 2D convolution:
**14.2× faster on the T4** (1.39 ms vs 20.4 ms, median of 5; observed 13.3–24.4×
across sessions — the separable pass is stable at ~1.4 ms, the naive 2D kernel is
what fluctuates), 7.5× on CPU.

## Target

Full HD (1920×1080) at >30 FPS on an NVIDIA T4 — met. Median 34 FPS
synchronous, 37 FPS streamed; the slowest of 11 measured runs was 30.2 FPS.

## Authors

- Duc Tin (22127415) - [@Sn-cpp](https://github.com/Sn-cpp)
- Hai Duong Huynh Le (22127081) - [@haiduonghuynhle](https://github.com/haiduonghuynhle)
