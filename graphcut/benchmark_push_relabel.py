"""Push-Relabel: correctness against Boykov-Kolmogorov, and how far it scales.

Correctness is the point of the BK column. A wrong max-flow still produces a
mask that looks like a mask, so the only way to know the cut is minimum is to
weigh it against an independent solver.

    NUMBA_NUM_THREADS=4 python -m graphcut.benchmark_push_relabel

Measured, Apple M-series (8 performance + 2 efficiency cores), 240x320:

    threads     1       2       4       8
    ms        196.5   152.8   219.8   336.5

and at 480x640:

    threads     1       2       4       8
    ms        990.2   836.4   617.7  1041.1

The peak moves right as the image grows — 2 threads at 240x320, 4 at 480x640 —
and then it degrades. Each outer iteration launches five prange passes (four
checkerboard push classes plus a relabel), and a push pass only touches a
quarter of the pixels, so past a certain thread count the fork/join and the
contention on the shared residual arrays cost more than the work they split.
Best measured speed-up is 1.60x at 480x640 on 4 threads.

On the Colab reference machine (2 vCPU) it beats serial BK outright:

    size        BK ms    PR ms   speed-up
    60x80        25.4     12.1     2.1x
    120x160     158.5    148.1     1.07x
    240x320     723.1    390.0     1.85x
    480x640    2410.6   1977.7     1.22x

That is the case for a GPU port rather than against it: the algorithm has the
right shape — per-pixel local work, no augmenting-path bookkeeping — but a
handful of CPU threads cannot amortise five kernel launches per iteration.
Thousands of resident GPU threads can.
"""
import time

import cv2
import numpy as np

from graphcut.push_relabel_numba import push_relabel

SIZES = ((60, 80), (120, 160), (240, 320), (480, 640))


def synthetic_graph(H, W, seed=1):
    """Capacities shaped like a real graph cut: t-links from a probability
    field, n-links from image gradient."""
    rng = np.random.default_rng(seed)
    img = cv2.GaussianBlur(rng.random((H, W)).astype(np.float32), (9, 9), 0)
    prob = cv2.GaussianBlur(rng.random((H, W)).astype(np.float32), (31, 31), 0)

    cap_src = (-np.log(np.clip(1 - prob, 1e-3, 1))).astype(np.float32).ravel() * 5
    cap_snk = (-np.log(np.clip(prob, 1e-3, 1))).astype(np.float32).ravel() * 5

    gx = np.zeros((H, W), np.float32); gx[:, :-1] = np.abs(np.diff(img, axis=1))
    gy = np.zeros((H, W), np.float32); gy[:-1, :] = np.abs(np.diff(img, axis=0))
    beta = 1.0 / (2 * max(float((gx ** 2).mean() + (gy ** 2).mean()), 1e-8))

    cap_right = (50 * np.exp(-beta * gx ** 2)).astype(np.float32); cap_right[:, -1] = 0
    cap_down = (50 * np.exp(-beta * gy ** 2)).astype(np.float32); cap_down[-1, :] = 0
    return cap_src, cap_snk, cap_right.ravel(), cap_down.ravel()


def cut_value(labeling, cap_src, cap_snk, cap_right, cap_down, H, W):
    total = float(np.where(labeling == 1, cap_src, cap_snk).sum())
    lab = labeling.reshape(H, W)
    total += float(cap_right.reshape(H, W)[:, :-1][lab[:, :-1] != lab[:, 1:]].sum())
    total += float(cap_down.reshape(H, W)[:-1, :][lab[:-1, :] != lab[1:, :]].sum())
    return total


def reference(cap_src, cap_snk, cap_right, cap_down, H, W):
    import maxflow
    g = maxflow.Graph[float]()
    nodes = g.add_nodes(H * W)
    for n in range(H * W):
        g.add_tedge(nodes[n], float(cap_src[n]), float(cap_snk[n]))
    for y in range(H):
        for x in range(W):
            n = y * W + x
            if x + 1 < W and cap_right[n] > 0:
                g.add_edge(nodes[n], nodes[n + 1], float(cap_right[n]), float(cap_right[n]))
            if y + 1 < H and cap_down[n] > 0:
                g.add_edge(nodes[n], nodes[n + W], float(cap_down[n]), float(cap_down[n]))
    flow = g.maxflow()
    seg = np.array([g.get_segment(nodes[n]) for n in range(H * W)], np.int32)
    return flow, seg


def main():
    import numba
    try:
        import maxflow  # noqa: F401
        have_ref = True
    except ImportError:
        have_ref = False
        print("PyMaxflow not installed — timing only, correctness unchecked\n")

    print(f"numba threads = {numba.get_num_threads()} "
          f"(set NUMBA_NUM_THREADS to sweep)\n")
    head = f"{'size':>11s} {'PR ms':>9s}"
    if have_ref:
        head += f" {'BK ms':>9s} {'speed-up':>9s} {'cut':>10s} {'labels':>8s}"
    print(head)

    for H, W in SIZES:
        caps = synthetic_graph(H, W)
        push_relabel(*caps, np.int32(H), np.int32(W), 5, 10)   # JIT warm-up

        times = []
        for _ in range(3):
            t = time.perf_counter()
            labeling = push_relabel(*caps, np.int32(H), np.int32(W), 20000, 20)
            times.append(time.perf_counter() - t)
        pr_ms = np.median(times) * 1000

        row = f"{H:5d}x{W:<5d} {pr_ms:9.1f}"
        if have_ref:
            t = time.perf_counter()
            flow, seg = reference(*caps, H, W)
            bk_ms = (time.perf_counter() - t) * 1000
            ours = cut_value(labeling, *caps, H, W)
            exact = abs(ours - flow) < 1e-3 * max(1.0, abs(flow))
            row += (f" {bk_ms:9.1f} {bk_ms / pr_ms:8.2f}x "
                    f"{'exact' if exact else 'ABOVE MIN':>10s} "
                    f"{(labeling == seg).mean() * 100:7.2f}%")
        print(row)


if __name__ == "__main__":
    main()
