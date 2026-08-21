# Adaptive Gaussian Mixture Model — foreground segmentation and selective blur

CSC14116 Applied Parallel Programming, HCMUS. Background subtraction with MOG2
(Zivkovic 2004) and background blur on traffic video, from a sequential Python
specification through Numba to CUDA.

Scored on **CDnet `baseline/highway`**, frames 470–1700, CDnet protocol
(don't-care pixels excluded). Best chain: **F1 0.9843, IoU 0.9691** —
**re-measured at the current HEAD** from the team's dataset mirror, reproducing
the original `b2523ba` figures digit for digit. Raw output:
[`benchmarks/records/eval_highway_full.txt`](benchmarks/records/eval_highway_full.txt).

Measured on a Colab T4: **88.8 FPS at 1080p**, 4.89× over the v0 baseline and
10.25× over Numba CPU, with all four backends producing byte-identical masks and
composites over 1.49 billion synthetic pixel positions. On the 1231 real CDnet
frames, 2 pixels of 94.5 million differ — both proven float32 threshold-boundary
flips ([benchmarks/records/t4_final_run.txt](benchmarks/records/t4_final_run.txt)).
Full numbers and method in [RESULTS-T4.md](RESULTS-T4.md).

## Quick start

```bash
pip install -r requirements.txt
python main.py --input_path LTSSUD-Test.mp4 --model numba
python main.py --input_path LTSSUD-Test.mp4 --model cuda_v2 --no-display
```

`--model` is one of `cpu numba cuda cuda_v1 cuda_v2`. The GPU backends
fall back to `None` and say so when no device is present.

## Layout

```
main.py                     the pipeline, end to end
settings.py                 the model and post-processing hyper-parameters
eval_highway.py             F1/IoU against CDnet, and the parity gate
bench_post.py               per-stage timing for v0/v1/v2, and the equivalence gate
bench_t4.py                 the isolated Kernel 0/Kernel 2 and baseline measurements
bench_fill.py               flood fill vs data-parallel reconstruction
notebook.ipynb              the report: recomputes what it can, labels what it quotes
utils/post_processing.py    the host chain — the specification the kernels match
gmm_mask/
  cpu/gmm_mask_cpu.py       sequential Python, the readable transliteration
  cpu/gmm_mask_numba.py     @njit + prange
  gpu/gmm_mask_cuda.py      v0 — model kernel only
  gpu/gmm_mask_cuda_v1.py   v1 — post-processing and blur on the device
  gpu/gmm_mask_cuda_v2.py   v2 — threshold fused, median and blur tiled
  gpu/post_kernels.py       threshold, median 5x5 (naive and tiled)
  gpu/blur_kernels.py       Kernel 2 — colour convert, separable blur, composite
tests/                      parity, blur, post chain
```

## The three GPU versions

| | what runs where | bus traffic @1080p |
|---|---|---|
| **v0** | mask on GPU; everything else on the host with OpenCV | 35.25 MB/frame |
| **v1** | colour convert, threshold, median, blur, composite as kernels; flood fill on host | 16.59 MB/frame |
| **v2** | threshold fused into the model kernel's epilogue; median and blur shared-memory tiled | 16.59 MB/frame |

## The course's optimization ladder, climbed

| Course ladder (Project Description §1.2) | In this project |
|---|---|
| Level 0 — CPU | `GMM_Mask_CPU` (pure NumPy spec; `src/cpu_baseline.py` is the template entry point), plus Numba CPU as the honest compiled baseline |
| Level 1 — naive GPU port, "correct, not fast" | v0: the model kernel, one thread per pixel |
| Level 2 — memory | planar coalesced state (`means[k][c][y][x]`); 3 B/px ingest; shared-memory tiling for median and blur; bus 35.25 → 16.59 MB/frame. (The Q8 integer blur itself is the correctness story — it ships in v1 and v2 alike.) |
| Level 3 — compute | threshold fused into the model kernel's epilogue |
| Level 4 — optional | CUDA streams: identified by the per-stage profile as the next step, **not claimed** |

`fill_holes` stays on the host deliberately — the course's own Partial GPU
Principle: *"identifying the one or two operations that dominate runtime and
accelerating those — not rewriting an entire application."* OpenCV implements it
as a scan-line flood fill; the data-parallel formulation (morphological
reconstruction) needs one dilate per pixel of propagation distance. `bench_fill.py` implements both, checks
they agree pixel-for-pixel, and times them: at 1080p that is **593 full-frame
passes**, and the wall-clock ratio runs 112–287× depending on the host. The pass
count is identical on every machine tried, which is the durable half of the
result — each pass is a grid-wide barrier, and a CUDA dilate a hundred times
faster still needs 593 of them in sequence. Knowing which stage not to move is a
result.

## Correctness

Every equivalence is a test, and all but one run without a GPU:

```bash
pytest tests/ -q                              # the host-side claims
NUMBA_ENABLE_CUDASIM=1 pytest tests/ -q       # + every CUDA kernel
```

- `tests/test_parity.py` — sequential == Numba == v0 == v1 == v2, and agreement
  with `cv2.createBackgroundSubtractorMOG2`.
- `tests/test_blur.py` — the Q8 blur, the colour conversion, the borders and the
  composite against OpenCV with **zero tolerance**.
- `tests/test_post_chain.py` — the threshold and median kernels, plus a check that
  every kernel compiles on real hardware (skipped under CUDASIM by design).
- `tests/test_scoring.py` — the CDnet don't-care protocol, on a hand-checkable fixture.

On a real GPU the hard gate is per-frame equality over the whole scored window,
not an unchanged F1:

```bash
python eval_highway.py --model cuda_v2 --parity-vs numba
```

### Grading access to the dataset

CDnet's original hosts went offline mid-project, so the team mirrors the sequence on
Hugging Face — anonymous HTTPS, CDnet citation on the card, sha256 recorded in
`benchmarks/records/eval_highway_full.txt`:

```bash
curl -LO https://huggingface.co/datasets/haiduonghuynhle/changedetection-2012-highway/resolve/main/highway.zip
unzip highway.zip && HIGHWAY_DIR=highway python eval_highway.py --colorspace both
```

The notebook downloads it from the same mirror automatically. `tests/test_scoring.py` pins the
scoring protocol on a hand-checkable fixture, and every pipeline still runs end-to-end on
labelled synthetic frames when the network is absent.

### What "bit-exact with OpenCV" means here, precisely

- **The blur is exact, unconditionally.** `cv2.GaussianBlur` on uint8 is not a
  float convolution: it quantises the kernel to Q8 with a *cumulative*
  quantiser and runs integer arithmetic. We reproduce that, so the equality
  holds independently of GPU architecture and compiler flags. (A *correct*
  float64 Gaussian disagrees with OpenCV on ~15% of pixels by one grey level —
  that gap is OpenCV's, and reproducing OpenCV means reproducing it.)
- **`cv2.cvtColor(BGR2YCrCb)` is exact** for the same reason, which is what lets
  the conversion move onto the device and the upload drop from 12 to 3 bytes
  per pixel.
- **The model is exact on synthetic input and 0.0014% off on video.** MOG2 is
  float32; OpenCV accumulates in a different order and contracts its own FMAs,
  so a pixel within an ulp of `Tb·σ²` can fall on either side. 22 pixels of
  1 536 000, all in one frame. Both numbers are reported.

## Development environment

macOS on Apple Silicon has no CUDA. Develop and verify on the CPU, run the
kernels under `NUMBA_ENABLE_CUDASIM=1` for correctness, measure on a Colab T4.
Python 3.12 (numba does not build on 3.14 yet).

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
```
