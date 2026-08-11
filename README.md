# grabcut_numba

A lightweight Numba-accelerated reimplementation of GrabCut. It parallelizes beta and neighbor-weight computation, uses OpenCV kmeans for GMM initialization and PyMaxflow for the graph cut stage.

Usage:

1. Copy an `input.jpg` into this folder.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the demo:

```bash
python demo.py
```

Files:
- `grabcut_numba.py` — main module.
- `demo.py` — simple demo that writes `mask_result.png`.

Notes:
- This is an approximation intended to show Numba-parallelized parts. For production use, further optimization and validation are recommended.
