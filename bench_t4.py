"""The measurements in RESULTS-T4.md that bench_post.py does not produce.

Every table in that file should be reproducible from committed code, and four
of them were originally produced by one-off cells typed into a Colab notebook.
This is those cells, kept.

    python bench_t4.py                  # all of it
    python bench_t4.py --only blur      # blur | ingest | baseline | equivalence

  blur         Kernel 2 in isolation: host cv2 vs naive vs tiled, no transfers.
  ingest       Kernel 0 decomposed, so the win is attributed honestly -- part of
               it is the conversion kernel and part of it is simply not
               allocating a device array every frame, which is what v0 does.
  baseline     v2 against Numba CPU. `bench_post.py --with-sequential` gives the
               >20x-over-sequential-Python number the proposal asks for, but
               sequential Python is a per-pixel interpreter loop and that ratio
               flatters. Numba is the baseline worth quoting.
  equivalence  120 frames at 1080p: Numba == v0 == v1 == v2, mask and composite,
               every frame. bench_post.py's own check is 24 frames and excludes
               the Numba host reference.

Needs a real GPU.
"""
import argparse
import time
import warnings

import cv2
import numpy as np

import bench_post as bp
from gmm_mask import (GMM_Mask_CUDA, GMM_Mask_CUDA_v1, GMM_Mask_CUDA_v2,
                      GMM_Mask_Numba)
from gmm_mask.gpu import blur_kernels as bk
from settings import BLUR_KSIZE, BLUR_SIGMA
from utils.post_processing import background_blur

SIZES = [((854, 480), "480p"), ((1280, 720), "720p"), ((1920, 1080), "1080p")]


def _median(fn, rounds, inner):
    from numba import cuda
    out = []
    for _ in range(rounds):
        cuda.synchronize()
        t = time.perf_counter()
        for _ in range(inner):
            fn()
        cuda.synchronize()
        out.append((time.perf_counter() - t) / inner * 1000)
    return float(np.median(out))


def blur(rounds=7, inner=20):
    from numba import cuda
    print("Kernel 2 -- blur + composite only, no transfers")
    print(f"  {'size':>7} {'host cv2':>10} {'GPU naive':>10} {'GPU tiled':>10} "
          f"{'host/tiled':>11} {'naive/tiled':>12}")
    d_kq = cuda.to_device(bk.gaussian_kernel_q8())
    block = (bk.BLUR_TILE_X, bk.BLUR_TILE_Y)
    for (w, h), lbl in SIZES:
        rng = np.random.default_rng(0)
        img = np.ascontiguousarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8))
        field = cv2.GaussianBlur(rng.random((h, w)).astype(np.float32), (31, 31), 0)
        msk = np.ascontiguousarray(((field > 0.55).astype(np.uint8)) * 255)
        grid = bk.blur_grid_for(h, w)
        d_src, d_msk = cuda.to_device(img), cuda.to_device(msk)
        d_hor = cuda.device_array((h, w, 3), np.uint16)
        d_out = cuda.device_array((h, w, 3), np.uint8)

        def gpu(tiled):
            kh = bk.blur_h_tiled_kernel if tiled else bk.blur_h_kernel
            kv = (bk.blur_v_composite_tiled_kernel if tiled
                  else bk.blur_v_composite_kernel)
            kh[grid, block](d_src, d_hor, d_kq)
            kv[grid, block](d_hor, d_src, d_msk, d_out, d_kq)

        host = lambda: background_blur(img, msk, BLUR_KSIZE, BLUR_SIGMA)
        for _ in range(5):
            gpu(False); gpu(True); host()
        # Interleaved, not three consecutive batches: on a shared T4 a batch
        # measures whatever the machine was doing during that batch.
        acc = {"host": [], "naive": [], "tiled": []}
        for _ in range(rounds):
            acc["host"].append(_median(host, 1, inner))
            acc["naive"].append(_median(lambda: gpu(False), 1, inner))
            acc["tiled"].append(_median(lambda: gpu(True), 1, inner))
        th, tn, tt = (float(np.median(acc[k])) for k in ("host", "naive", "tiled"))
        print(f"  {lbl:>7} {th:9.3f}ms {tn:9.3f}ms {tt:9.3f}ms "
              f"{th/tt:10.1f}x {tn/tt:11.2f}x")


def ingest(rounds=7, inner=20, size=(1920, 1080)):
    from numba import cuda
    w, h = size
    rng = np.random.default_rng(1)
    bgr = np.ascontiguousarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8))
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    planar = np.ascontiguousarray(ycc.transpose(2, 0, 1).astype(np.float32))
    grid, block = bk.blur_grid_for(h, w), (bk.BLUR_TILE_X, bk.BLUR_TILE_Y)
    d_bgr = cuda.device_array((h, w, 3), np.uint8)
    d_pl = cuda.device_array((3, h, w), np.float32)
    d_pl_pre = cuda.device_array((3, h, w), np.float32)

    steps = {
        "A host cvtColor":         lambda: cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb),
        "B host f32+transpose":    lambda: np.ascontiguousarray(ycc.transpose(2, 0, 1).astype(np.float32)),
        "C H2D 12B/px (alloc)":    lambda: cuda.to_device(planar),
        "D H2D 12B/px (prealloc)": lambda: d_pl_pre.copy_to_device(planar),
        "E H2D  3B/px (prealloc)": lambda: d_bgr.copy_to_device(bgr),
        "F cvt kernel on device":  lambda: bk.bgr2ycrcb_planar_kernel[grid, block](d_bgr, d_pl, True),
    }
    for _ in range(5):
        for f in steps.values():
            f()
    acc = {k: [] for k in steps}
    for _ in range(rounds):                 # interleaved across stages
        for k, f in steps.items():
            acc[k].append(_median(f, 1, inner))
    r = {k: float(np.median(v)) for k, v in acc.items()}

    print(f"\nKernel 0 -- frame ingest at {w}x{h}, decomposed")
    for k, v in r.items():
        print(f"  {k:26s} {v:7.3f} ms")
    old = r["A host cvtColor"] + r["B host f32+transpose"] + r["C H2D 12B/px (alloc)"]
    new = r["E H2D  3B/px (prealloc)"] + r["F cvt kernel on device"]
    print(f"\n  OLD  A+B+C = {old:6.2f} ms      NEW  E+F = {new:5.2f} ms   -> {old/new:.1f}x")
    print(f"\n  attribution of the {old-new:.1f} ms saved:")
    print(f"    host convert + transpose removed : {r['A host cvtColor']+r['B host f32+transpose']:6.2f} ms")
    print(f"    12 B/px -> 3 B/px upload         : {r['D H2D 12B/px (prealloc)']-r['E H2D  3B/px (prealloc)']:6.2f} ms")
    print(f"    device allocation removed        : {r['C H2D 12B/px (alloc)']-r['D H2D 12B/px (prealloc)']:6.2f} ms")
    print(f"    conversion kernel added back     : {-r['F cvt kernel on device']:6.2f} ms")
    print("\n  The conversion kernel is nearly free, and that is what made the")
    print("  3 B/px upload possible -- the conversion had to happen somewhere.")


def baseline(n=40, warm=8):
    print("\nv2 against Numba CPU -- the honest baseline")
    print(f"  {'size':>7} {'Numba CPU':>11} {'v2 GPU':>9} {'speedup':>9} "
          f"{'Numba FPS':>10} {'v2 FPS':>8}")
    for (w, h), lbl in SIZES:
        frames = bp.make_frames(n, (w, h))
        tn = bp.one_pass(lambda: GMM_Mask_Numba(h, w), bp.run_v0, frames, warm=warm)
        tg = bp.one_pass(lambda: GMM_Mask_CUDA_v2(h, w), bp.run_gpu, frames, warm=warm)
        print(f"  {lbl:>7} {tn:9.2f}ms {tg:7.2f}ms {tn/tg:8.2f}x "
              f"{1000/tn:10.1f} {1000/tg:8.1f}")
    print("\n  Single passes, not medians. Same 40 frames and 8 warmups as")
    print("  bench_post.py so the two are comparable, but expect these to read")
    print("  slower -- quote bench_post.py's medians; this is for the ratio.")


def equivalence(n=120, size=(1920, 1080)):
    w, h = size
    frames = bp.make_frames(n, (w, h))
    models = {"numba (host, reference)": (GMM_Mask_Numba(h, w), bp.run_v0),
              "v0 CUDA + host post":     (GMM_Mask_CUDA(h, w), bp.run_v0),
              "v1 GPU post+blur":        (GMM_Mask_CUDA_v1(h, w), bp.run_gpu),
              "v2 fused + tiled":        (GMM_Mask_CUDA_v2(h, w), bp.run_gpu)}
    ref_key = "numba (host, reference)"
    bad_m = {k: 0 for k in models}
    bad_c = {k: 0 for k in models}
    first = {k: None for k in models}
    fg = 0
    for i, f in enumerate(frames):
        outs = {k: fn(m, f) for k, (m, fn) in models.items()}
        rm, rc = outs[ref_key]
        fg += bool((rm == 255).any())
        for k in models:
            m, c = outs[k]
            dm, dc = int((m != rm).sum()), int((c != rc).sum())
            if (dm or dc) and first[k] is None:
                first[k] = i
            bad_m[k] += dm
            bad_c[k] += dc

    px = n * h * w
    print(f"\nEquivalence: {w}x{h}, {n} frames, {px:,} pixels per comparison")
    print(f"  frames containing foreground: {fg}/{n}\n")
    print(f"  {'backend':26} {'mask diff':>11} {'composite diff':>15} {'verdict':>12}")
    ok = fg > 0
    for k in models:
        v = "IDENTICAL" if not (bad_m[k] or bad_c[k]) else f"DIVERGED@{first[k]}"
        ok &= not (bad_m[k] or bad_c[k])
        print(f"  {k:26} {bad_m[k]:>11} {bad_c[k]:>15} {v:>12}")
    print(f"\n  total pixels compared: {px*len(models)*2:,}")
    if not ok:
        raise SystemExit("backends disagree — every timing number is void")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", choices=["blur", "ingest", "baseline", "equivalence"])
    args = ap.parse_args()
    if GMM_Mask_CUDA is None:
        raise SystemExit("no CUDA device — this benchmark needs a real GPU")
    try:
        from numba.core.errors import NumbaPerformanceWarning
        warnings.simplefilter("ignore", NumbaPerformanceWarning)
    except Exception:
        pass

    bp.environment()
    for name, fn in (("blur", blur), ("ingest", ingest),
                     ("baseline", baseline), ("equivalence", equivalence)):
        if args.only in (None, name):
            fn()
