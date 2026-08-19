"""Is the flood fill worth moving to the GPU? Measure it, do not assert it.

The repository has claimed "344 ms against 2.2 ms at 1080p" in three places for
some time, with no code that produces either number. This file is that code.

`fill_holes` marks every background region fully enclosed by foreground. OpenCV
does it with a scan-line flood fill, which is sequential. The data-parallel
equivalent is **morphological reconstruction by dilation**: seed the background
at the image border, then dilate-and-mask against the complement of the mask
until nothing changes. Every iteration propagates the frontier by one pixel, so
the iteration count is the longest path a flood has to travel — hundreds of
steps at 1080p, each one a full-frame pass.

That is the shape of the argument, and this file checks both halves of it:

1. **the two agree** — a reconstruction that gives a different answer is not a
   slower alternative, it is a wrong one, and comparing their speed would be
   meaningless;
2. **how much slower**, on this machine, with the iteration count printed so the
   reader can see where the cost comes from.

    python bench_fill.py --sizes 240 480 1080

**What this does and does not measure.** Both timings are CPU: OpenCV's
`floodFill` against a Python loop over `cv2.dilate` plus a convergence check.
No CUDA reconstruction was implemented or measured, so read the millisecond
column as an order of magnitude, not as the number a tuned GPU implementation
would produce. The **pass count** is the durable result — it is a property of
this formulation and of the image, not of the hardware, and a CUDA dilate that
is a hundred times faster per pass still needs one grid-wide synchronisation
between each of them.

Nor is this the only conceivable parallel algorithm; it is the standard one.
The claim being supported is "this formulation costs hundreds of dependent
full-frame passes", not "no parallel flood fill can ever be worthwhile".

Connectivity is the one detail that would rig the comparison. `cv2.floodFill`
defaults to 4-connectivity, so the structuring element here is `MORPH_CROSS`,
which is also 4-connected. `MORPH_RECT` would propagate diagonally, converge in
fewer passes, and compute a *different* answer — which is why the equality
check below is not a formality.
"""
import argparse
import time

import cv2
import numpy as np

from utils.post_processing import fill_holes

SIZES = {240: (320, 240), 480: (854, 480), 720: (1280, 720), 1080: (1920, 1080)}
REPEATS = 3


def blobby(h, w, seed=0, thresh=0.52, sigma=31):
    """A mask shaped like MOG2 output: blobs with holes in them."""
    rng = np.random.default_rng(seed)
    field = cv2.GaussianBlur(rng.random((h, w)).astype(np.float32),
                             (sigma, sigma), 0)
    return ((field > thresh).astype(np.uint8)) * 255


def fill_by_reconstruction(mask, max_iter=4096):
    """Morphological reconstruction of the background from the border.

    The data-parallel formulation: `marker` starts as the image border and grows
    by one dilate per iteration, constrained to stay inside the background. What
    it never reaches is a hole. Each iteration is embarrassingly parallel; the
    *sequence* of them is not, and that is the whole problem.

    Returns (filled, iterations).
    """
    bg = mask == 0
    marker = np.zeros_like(bg)
    marker[0, :] = bg[0, :]
    marker[-1, :] = bg[-1, :]
    marker[:, 0] = bg[:, 0]
    marker[:, -1] = bg[:, -1]

    el = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    m = marker.astype(np.uint8)
    bgu = bg.astype(np.uint8)
    for i in range(1, max_iter + 1):
        grown = cv2.dilate(m, el) & bgu
        if np.array_equal(grown, m):
            return np.where(bgu & ~m, np.uint8(255), mask), i
        m = grown
    raise RuntimeError(f"reconstruction did not converge in {max_iter} passes")


def timeit(fn, arg, repeats=5):
    best = []
    for _ in range(repeats):
        t = time.perf_counter()
        fn(arg)
        best.append((time.perf_counter() - t) * 1000)
    return float(np.median(best))


def run(size, seed=0):
    w, h = size
    mask = blobby(h, w, seed)

    a = fill_holes(mask)
    b, iters = fill_by_reconstruction(mask)
    agree = np.array_equal(a, b)

    # Equal repeats, or the two medians are not comparable.
    t_seq = timeit(fill_holes, mask, repeats=REPEATS)
    t_rec = timeit(lambda m: fill_by_reconstruction(m)[0], mask, repeats=REPEATS)

    holes = int(cv2.connectedComponents(((a != mask)).astype(np.uint8))[0]) - 1
    print(f"  {w}x{h:<6} {t_seq:9.2f} {t_rec:12.2f} {t_rec / t_seq:9.1f}x "
          f"{iters:9d} {holes:8d}  {'yes' if agree else 'NO — DIFFERENT RESULT'}")
    return agree


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sizes", nargs="+", type=int, default=[240, 480, 720, 1080])
    args = ap.parse_args()

    print("fill_holes: sequential flood fill vs data-parallel reconstruction\n")
    print(f"  {'size':<12} {'floodFill':>9} {'reconstruct':>12} {'ratio':>10} "
          f"{'passes':>9} {'holes':>8}  identical")
    ok = True
    for s in args.sizes:
        ok &= run(SIZES[s])
    print("\n  'passes' is the number of full-frame dilate steps the parallel")
    print("  version needs. Each one is a grid-wide barrier, and no amount of")
    print("  GPU makes the sequence of them shorter.")
    if not ok:
        raise SystemExit("reconstruction disagreed with floodFill — the timing "
                         "comparison above is meaningless")
