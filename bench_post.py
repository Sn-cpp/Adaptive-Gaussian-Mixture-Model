"""Per-stage timing for the GPU pipeline versions. Needs a real GPU.

    python bench_post.py --sizes 480 720 1080
    python bench_post.py --sizes 480 --with-sequential   # the >20x baseline

    v0  mask on the device; threshold, median, fill, blur, composite on the host
    v1  threshold + median + blur + composite as CUDA kernels; fill on the host
    v2  threshold fused into the model kernel's epilogue; median and blur tiled

The number that matters is not the kernel time — it is what crosses the bus. v0
uploads a planar float32 frame (12 bytes a pixel) and copies the confidence map
back (4 bytes a pixel) so the host can threshold it. v1 and v2 upload the BGR
frame (3 bytes a pixel), convert colour on the device, and return one byte a
pixel of mask plus the finished composite. The bytes column below is computed by
`bytes_per_frame()` from the array shapes; do not quote a figure from this
docstring, which is precisely how the number here came to be wrong once already
— it omitted v0's bg_prob copy-back and read 26.96 MB instead of 35.25.

**Correctness is measured separately from speed, and first.** The previous
version of this file interleaved them: it timed three models, then compared a
single frame of output after 312 state updates. That misses a divergence that
appears at frame 40 and heals by frame 312, and it reports a coincidence as a
proof. Here every frame of every version is compared against the host chain,
on models that are driven once, and only then does anything get timed.

Timings are medians of interleaved repeats, each on a freshly built model so
that no repeat measures a more converged mixture than the one before it.
Interleave if you re-measure: a single cold pass on a shared T4 has reported
swings of 30% and more, larger than every effect this file tries to measure.
"""
import argparse
import platform
import sys
import time

import cv2
import numpy as np

from gmm_mask import (GMM_Mask_CPU, GMM_Mask_CUDA, GMM_Mask_CUDA_v1,
                      GMM_Mask_CUDA_v2)
from settings import BLUR_KSIZE, BLUR_SIGMA
from utils.post_processing import background_blur, fill_holes, refine_mask

SIZES = {480: (854, 480), 720: (1280, 720), 1080: (1920, 1080)}
ROUNDS = 5      # interleaved passes per version


def environment():
    """Every claim here is build-sensitive; record what produced it."""
    import numba
    rows = [("python", platform.python_version()), ("numpy", np.__version__),
            ("opencv", cv2.__version__), ("numba", numba.__version__),
            ("platform", platform.platform())]
    try:
        from numba import cuda
        d = cuda.get_current_device()
        # numba returns device.name as bytes on some versions and str on
        # others (0.60 on Colab gives str, 0.61 locally gives bytes). Found by
        # running this on the T4, which is the point of running it on the T4.
        name = d.name.decode() if isinstance(d.name, bytes) else str(d.name)
        rows.append(("gpu", f"{name} cc{d.compute_capability}"))
        rows.append(("cuda driver", str(cuda.cudadrv.driver.driver.get_version())))
    except Exception as e:                          # pragma: no cover
        rows.append(("gpu", f"unavailable ({e})"))
    print("environment")
    for k, v in rows:
        print(f"  {k:14s} {v}")


def make_frames(n, size, seed=0):
    """Synthetic traffic as a decoder would hand it over: uint8 BGR.

    Not float32 planar. The old version handed the models a preprocessed array,
    which quietly excluded the host's colour conversion and transpose from v0's
    measurement — the exact cost v1 removes. Measuring the thing you optimised
    away as if it were free makes the speedup look smaller and the reason for
    it invisible.
    """
    w, h = size
    rng = np.random.default_rng(seed)
    road = cv2.GaussianBlur(
        rng.integers(60, 110, (h, w, 3), dtype=np.uint8), (7, 7), 0)
    out = []
    for i in range(n):
        f = road.astype(np.float32)
        for k in range(4):
            cx = int((i * 7 + k * w // 4) % w)
            cy = h // 2 + (k - 2) * h // 8
            cv2.rectangle(f, (cx, cy), (cx + w // 12, cy + h // 14),
                          (200.0, 195.0, 205.0), -1)
        out.append(np.ascontiguousarray(
            np.clip(f + rng.normal(0, 2.0, f.shape), 0, 255), dtype=np.uint8))
    return out


# ── the three pipelines, each stated end to end from a BGR frame ─────────────

def run_v0(model, frame_bgr):
    """Host everything except the model kernel — the baseline being beaten."""
    # uint8 in: to_planar() inside apply() does the transpose and the cast
    # together, so casting here would only add a redundant full-frame copy.
    mask, bg_prob, _ = model.apply(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb))
    refined = refine_mask(np.asarray(mask), bg_prob=np.asarray(bg_prob))
    return refined, background_blur(frame_bgr, refined, BLUR_KSIZE, BLUR_SIGMA)


def run_gpu(model, frame_bgr):
    """v1/v2 — only the flood fill happens on the host."""
    refined = fill_holes(model.mask_from_bgr(frame_bgr))
    return refined, model.composite(refined)


PIPELINES = {
    "v0 host post+blur": (GMM_Mask_CUDA, run_v0),
    "v1 GPU post+blur": (GMM_Mask_CUDA_v1, run_gpu),
    "v2 fused + tiled": (GMM_Mask_CUDA_v2, run_gpu),
}


def bytes_per_frame(h, w):
    """Bus traffic derived from the array shapes, not from prose."""
    px = h * w
    v0 = px * 3 * 4 + px * 1 + px * 4          # planar f32 up, mask + bg_prob down
    gpu = px * 3 + px * 1 + px * 1 + px * 3    # BGR up, mask down, mask up, composite down
    return v0 / 1e6, gpu / 1e6


# ── correctness, once, on models nothing has timed ───────────────────────────

def check_equivalence(size, n=24):
    w, h = size
    frames = make_frames(n, size)
    models = {k: cls(h, w) for k, (cls, _) in PIPELINES.items()}
    runs = {k: fn for k, (_, fn) in PIPELINES.items()}

    bad_mask = {k: 0 for k in PIPELINES}
    bad_comp = {k: 0 for k in PIPELINES}
    ref_key = "v0 host post+blur"
    saw_fg = False

    for f in frames:
        outs = {k: runs[k](models[k], f) for k in PIPELINES}
        ref_m, ref_c = outs[ref_key]
        saw_fg |= bool((ref_m == 255).any())
        for k in PIPELINES:
            m, c = outs[k]
            bad_mask[k] += int((m != ref_m).sum())
            bad_comp[k] += int((c != ref_c).sum())

    print(f"\nequivalence at {w}x{h} over {n} frames, vs the host chain")
    ok = saw_fg
    if not saw_fg:
        print("  DEGENERATE: no foreground in any frame — this proves nothing")
    for k in PIPELINES:
        verdict = "identical" if not (bad_mask[k] or bad_comp[k]) else "DIVERGED"
        ok &= not (bad_mask[k] or bad_comp[k])
        print(f"  {k:20s} mask {bad_mask[k]:>10d} px   "
              f"composite {bad_comp[k]:>10d} px   {verdict}")
    if not ok:
        print("  A speedup that changes the output is not a speedup.")
    return ok


# ── timing ────────────────────────────────────────────────────────────────────

def one_pass(build, fn, frames, warm=8):
    """One timed pass over the frames, on a model built for this pass alone.

    Rebuilding matters. Replaying the same frames into one model makes the
    second pass measure a converged mixture and the first a converging one —
    genuinely different branch behaviour in the same kernel, reported as
    run-to-run noise.

    This returns a single measurement, not a median. The interleaving and the
    median belong to the caller, because a median taken here would be a median
    of consecutive runs and the whole point is that consecutive runs on a
    shared T4 are correlated.
    """
    from numba import cuda
    m = build()
    for f in frames[:warm]:
        fn(m, f)
    cuda.synchronize()
    t0 = time.perf_counter()
    for f in frames[warm:]:
        fn(m, f)
    cuda.synchronize()
    return (time.perf_counter() - t0) / len(frames[warm:]) * 1000


def stage_breakdown(size, n=24):
    """Where the frame actually goes, for each of v0, v1 and v2.

    Boundaries are sync-bounded wall clock: `time.perf_counter()` with every
    device stage synchronised before the clock is read (`mask_from_bgr` and
    `composite` sync internally; v0's `apply` copies back, which syncs). These
    are NOT CUDA events — a stage that says "H2D+kernels+D2H" is the wall time
    of that whole span, not a kernel-only figure, and the labels say so.

    This is the table `proposal.md` §6 promises. The earlier version measured
    only v2, which could not answer the obvious referee question — of the
    end-to-end speedup, how much comes from fusing, how much from tiling, and
    how much from moving the blur at all. Three columns per version answers it
    by inspection: v0's host stages shrink to device stages in v1, and v1's
    device stages shrink again in v2.
    """
    from numba import cuda
    w, h = size
    frames = make_frames(n, size)
    warm = 8

    def timed_stages(build, stages):
        """Drive one fresh model; return mean ms per named stage."""
        m = build()
        for f in frames[:warm]:
            for _, fn in stages:
                fn(m, f)
        cuda.synchronize()
        acc = {name: 0.0 for name, _ in stages}
        for f in frames[warm:]:
            for name, fn in stages:
                t = time.perf_counter()
                fn(m, f)
                acc[name] += time.perf_counter() - t
        k = len(frames) - warm
        return {name: v / k * 1000 for name, v in acc.items()}

    # Stage chains mirror run_v0/run_gpu exactly — same calls, same order,
    # state threaded through a scratch dict so each stage feeds the next.
    scratch = {}

    def v0_stages():
        def cvt(m, f):
            scratch["ycc"] = cv2.cvtColor(f, cv2.COLOR_BGR2YCrCb)

        def model_step(m, f):
            mask, bg_prob, _ = m.apply(scratch["ycc"])
            scratch["mask"], scratch["bg"] = np.asarray(mask), np.asarray(bg_prob)

        def thr_med(m, f):
            scratch["refined"] = refine_mask(scratch["mask"],
                                             bg_prob=scratch["bg"], do_fill=False)

        def fill(m, f):
            scratch["filled"] = fill_holes(scratch["refined"])

        def blur(m, f):
            background_blur(f, scratch["filled"], BLUR_KSIZE, BLUR_SIGMA)

        return [("host cvtColor", cvt),
                ("f32 H2D + model + mask/bg D2H", model_step),
                ("host threshold + median", thr_med),
                ("host fill_holes", fill),
                ("host blur + composite", blur)]

    def gpu_stages():
        def ingest(m, f):
            scratch["refined"] = m.mask_from_bgr(f)

        def fill(m, f):
            scratch["filled"] = fill_holes(scratch["refined"])

        def comp(m, f):
            m.composite(scratch["filled"])

        return [("ingest (H2D+cvt+model+post+D2H)", ingest),
                ("host fill_holes", fill),
                ("composite (H2D+blur+D2H)", comp)]

    versions = [("v0", lambda: GMM_Mask_CUDA(h, w), v0_stages()),
                ("v1", lambda: GMM_Mask_CUDA_v1(h, w), gpu_stages()),
                ("v2", lambda: GMM_Mask_CUDA_v2(h, w), gpu_stages())]

    print(f"\nper-stage at {w}x{h} — sync-bounded wall clock, {n - warm} frames")
    for name, build, stages in versions:
        r = timed_stages(build, stages)
        total = sum(r.values())
        print(f"  {name}  (total {total:7.3f} ms)")
        for stage, ms in r.items():
            print(f"    {stage:34s} {ms:8.3f} ms  {ms / max(total, 1e-9):6.1%}")


def bench(size, n=40, with_sequential=False):
    w, h = size
    frames = make_frames(n, size)
    v0_mb, gpu_mb = bytes_per_frame(h, w)

    # Genuinely interleaved: v0, v1, v2, v0, v1, v2, ... so thermal drift and
    # host contention land on all three equally. The reported figure is the
    # median of all ROUNDS passes, not a median of per-version medians — that
    # would average away the very correlation the interleaving exists to break.
    rows = {name: [] for name in PIPELINES}
    for _ in range(ROUNDS):
        for name, (cls, fn) in PIPELINES.items():
            rows[name].append(one_pass(lambda c=cls: c(h, w), fn, frames))

    print(f"\n{w}x{h}, {n - 8} timed frames, median of {ROUNDS} interleaved passes")
    print(f"  {'version':20s} {'ms/frame':>9s} {'FPS':>7s} {'vs v0':>8s} "
          f"{'MB/frame':>9s}")
    base = np.median(rows["v0 host post+blur"])
    for name in PIPELINES:
        ms = np.median(rows[name])
        mb = v0_mb if name.startswith("v0") else gpu_mb
        print(f"  {name:20s} {ms:9.2f} {1000 / ms:7.1f} {base / ms:7.2f}x "
              f"{mb:9.2f}")

    if with_sequential:
        # The 100% target is ">20x over sequential Python". GMM_Mask_CPU is a
        # per-pixel Python loop, so this is minutes per frame at 1080p — run it
        # at 480p and say so rather than quoting an extrapolation as a measurement.
        seq = GMM_Mask_CPU(h, w)
        t = time.perf_counter()
        for f in frames[:2]:
            run_v0(seq, f)
        seq_ms = (time.perf_counter() - t) / 2 * 1000
        best = min(float(np.median(rows[k])) for k in PIPELINES)
        print(f"  {'sequential Python':20s} {seq_ms:9.2f} {1000 / seq_ms:7.1f} "
              f"{'-':>7s}  (2 frames only)")
        print(f"  -> best GPU version is {seq_ms / best:.1f}x sequential "
              f"at {w}x{h}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sizes", nargs="+", type=int, default=[480, 720, 1080])
    ap.add_argument("--with-sequential", action="store_true",
                    help="add the pure-Python baseline (slow; use at 480p)")
    ap.add_argument("--skip-equivalence", action="store_true")
    args = ap.parse_args()
    if GMM_Mask_CUDA is None:
        raise SystemExit("no CUDA device — this benchmark needs a real GPU")

    environment()
    for s in args.sizes:
        if not args.skip_equivalence and not check_equivalence(SIZES[s]):
            raise SystemExit(
                f"versions disagree at {s}p — timing them would be meaningless")
        bench(SIZES[s], with_sequential=args.with_sequential)
        stage_breakdown(SIZES[s])
