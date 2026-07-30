# Adaptive Gaussian Mixture Model

Real-time background subtraction using the Stauffer-Grimson adaptive Gaussian Mixture Model (1999), with multiple backend implementations for performance comparison.

## Backends

| Backend | Class | Description |
|---|---|---|
| NumPy (CPU) | `GMM_CPU` | Vectorized NumPy baseline |
| Numba (CPU) | `GMM_CPU_NUMBA` | JIT-compiled with serial and `prange` parallel modes |
| CuPy vectorized (GPU) | `GMM_CUPY_V0` | CuPy array operations on GPU |
| CuPy RawKernel (GPU) | `GMM_CUPY_V1` | Custom CUDA kernels via CuPy RawKernel |

## Installation

```bash
pip install -r requirements.txt
```

For GPU backends, install CuPy matching your CUDA version:
```bash
pip install cupy-cuda12x   # CUDA 12.x
# or
pip install cupy-cuda11x   # CUDA 11.x
```

## Usage

```bash
python main.py
```

Edit `model_choice` in `main.py` to select backend (0=CPU, 1=Numba, 2=CuPy V0, 3=CuPy V1).

Default parameters: `K=7`, `match_threshold=3.5`, `bg_threshold=0.7`, `alpha=0.01`.

## Running Tests

```bash
pytest tests/ -v
```

GPU tests are automatically skipped when CuPy is not available.

## Project Structure

```
.
├── gmm/
│   ├── cpu/
│   │   ├── GMM_cpu.py            # NumPy vectorized
│   │   └── GMM_cpu_numba.py      # Numba JIT (serial + parallel)
│   └── gpu/
│       ├── GMM_cupy_v0.py        # CuPy array ops
│       ├── GMM_cupy_v1.py        # CuPy RawKernel
│       └── kernels/              # CUDA .cu kernel files
├── utils/
│   ├── post_processing.py        # Morphological refinement, background blur
│   └── timer.py                  # CPU/GPU timing utilities
├── tests/
│   ├── conftest.py               # pytest fixtures
│   └── test_correctness.py       # Cross-backend correctness tests
├── main.py                       # Webcam/video demo entry point
├── settings.py                   # INIT_VAR, REINIT_WEIGHT constants
└── requirements.txt
```

## Authors

- Duc Tin (22127415) - [@Sn-cpp](https://github.com/Sn-cpp)
- Hai Duong Huynh Le (22127081) - [@haiduonghuynhle](https://github.com/haiduonghuynhle)
