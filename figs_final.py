"""Figures for the final seminar deck (`Project/final-seminar.tex`).

Numbers are transcribed from RESULTS-T4.md, measured on a Colab Tesla T4 at
commit a6c0285. They are *quoted* here rather than measured: the harnesses that
produce them are `bench_post.py`, `bench_t4.py` and `bench_fill.py`, and this
file only draws. If a number here disagrees with RESULTS-T4.md, that file wins.

The visual idiom follows the reference deck the slides are styled after: white
background, blue bars with exactly one red bar for the version being argued
for, value labels above the bars, bold axis labels, dashed gridlines.

    python figs_final.py            # writes into ../figs/
"""
import os
import sys

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(REPO), "figs")
sys.path.insert(0, REPO)
os.makedirs(OUT, exist_ok=True)

# Palette validated for colour-vision safety (dataviz six-checks): the old
# grey/blue failed the chroma floor. Fixed identity: v0 blue, v1 amber, v2 red.
BLUE, RED, GREY = "#2a78d6", "#c0392b", "#eda100"  # GREY slot now amber (v1)
SIZES = ["480p", "720p", "1080p"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.grid": True, "grid.linestyle": "--", "grid.alpha": 0.4,
    "axes.axisbelow": True, "figure.facecolor": "white",
})


def _save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {name}")


def _label(ax, bars, fmt="{:.2f}", dy=0.01):
    top = max(b.get_height() for b in bars)
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + top * dy,
                fmt.format(b.get_height()), ha="center", va="bottom",
                fontsize=8, fontweight="bold")


# ── 1. end-to-end throughput ─────────────────────────────────────────────────

MS = {"v0": [10.22, 20.82, 55.03],
      "v1": [3.74, 6.98, 13.75],
      "v2": [3.26, 6.02, 11.26]}
FPS = {"v0": [97.9, 48.0, 18.2],
       "v1": [267.3, 143.3, 72.8],
       "v2": [307.1, 166.2, 88.8]}


def fig_speedup():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.6))
    x = np.arange(3)
    w = 0.26
    for i, (k, c) in enumerate((("v0", BLUE), ("v1", GREY), ("v2", RED))):
        b = a1.bar(x + (i - 1) * w, MS[k], w, label=k, color=c)
        _label(a1, b)
    a1.set_xticks(x); a1.set_xticklabels(SIZES)
    a1.set_ylabel("ms / frame", fontweight="bold")
    a1.set_title("Time per frame (lower is better)", fontweight="bold", fontsize=10)
    a1.legend(frameon=False, fontsize=9)

    for i, (k, c) in enumerate((("v0", BLUE), ("v1", GREY), ("v2", RED))):
        b = a2.bar(x + (i - 1) * w, FPS[k], w, label=k, color=c)
        _label(a2, b, "{:.0f}")
    a2.axhline(30, color="black", lw=1, ls=":")
    a2.text(2.45, 34, "30 FPS target", ha="right", fontsize=8, style="italic")
    a2.set_xticks(x); a2.set_xticklabels(SIZES)
    a2.set_ylabel("frames / second", fontweight="bold")
    a2.set_title("Throughput (higher is better)", fontweight="bold", fontsize=10)
    _save(fig, "fig_final_speedup.png")


# ── 2. where the frame goes ──────────────────────────────────────────────────

# Per-stage at commit bc55214, sync-bounded wall clock (RESULTS-T4.md §4).
# For the chart, v0's five stages are folded into the same three buckets
# v1/v2 report — mask production (cvt+model+threshold+median), host fill,
# blur+composite — so the three bars stack comparably; the full five-way
# split is in the table.
PERSTAGE = {  # ms per frame: [480p, 720p, 1080p]
    "v0": {"mask production": [6.278, 12.302, 35.924],
           "host fill_holes": [0.965, 2.155, 4.622],
           "blur + composite": [3.385, 6.333, 14.383]},
    "v1": {"mask production": [1.175, 1.940, 3.651],
           "host fill_holes": [0.877, 1.938, 4.441],
           "blur + composite": [1.616, 3.083, 6.057]},
    "v2": {"mask production": [1.149, 1.984, 3.510],
           "host fill_holes": [0.882, 1.972, 4.719],
           "blur + composite": [1.108, 2.044, 3.665]},
}
STAGE_COLORS = {"mask production": BLUE, "host fill_holes": RED,
                "blur + composite": GREY}


def fig_perstage():
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4), sharey=False)
    for ax, (si, lbl) in zip(axes, enumerate(SIZES)):
        x = np.arange(3)
        bottom = np.zeros(3)
        for stage, c in STAGE_COLORS.items():
            vals = [PERSTAGE[v][stage][si] for v in ("v0", "v1", "v2")]
            ax.bar(x, vals, 0.55, bottom=bottom, color=c,
                   label=stage if si == 0 else None)
            for xi, (v, b0) in enumerate(zip(vals, bottom)):
                if v > 0.55 * (1 + 3 * si):
                    ax.text(xi, b0 + v / 2, f"{v:.1f}", ha="center", va="center",
                            fontsize=7, fontweight="bold",
                            color="white" if c != GREY else "black")
            bottom += np.array(vals)
        ax.set_xticks(x); ax.set_xticklabels(["v0", "v1", "v2"])
        ax.set_title(lbl, fontweight="bold", fontsize=10)
        if si == 0:
            ax.set_ylabel("ms / frame", fontweight="bold")
    fig.legend(loc="upper center", ncol=3, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 1.06))
    fig.suptitle("Per stage, all three versions — at 1080p the host flood fill "
                 "is v2's largest stage (39.7%)", fontweight="bold",
                 fontsize=10, y=1.16)
    _save(fig, "fig_final_perstage.png")


# ── 3. bus traffic ───────────────────────────────────────────────────────────

def fig_bus():
    v0 = [6.97, 15.67, 35.25]
    gpu = [3.28, 7.37, 16.59]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    x = np.arange(3)
    b1 = ax.bar(x - 0.19, v0, 0.38, label="v0", color=BLUE)
    b2 = ax.bar(x + 0.19, gpu, 0.38, label="v1 / v2", color=RED)
    _label(ax, b1); _label(ax, b2)
    ax.set_xticks(x); ax.set_xticklabels(SIZES)
    ax.set_ylabel("MB across the bus / frame", fontweight="bold")
    ax.set_title("Host–device traffic, from the array shapes",
                 fontweight="bold", fontsize=10)
    ax.legend(frameon=False, fontsize=9)
    _save(fig, "fig_final_bus.png")


# ── 4. the blur in isolation: tiling was predicted not to matter ─────────────

def fig_blur():
    host = [3.300, 6.397, 13.740]
    naive = [0.876, 1.874, 3.690]
    tiled = [0.370, 0.793, 1.554]
    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    x = np.arange(3)
    w = 0.26
    for i, (vals, lab, c) in enumerate(((host, "host cv2", GREY),
                                        (naive, "GPU naive", BLUE),
                                        (tiled, "GPU tiled", RED))):
        b = ax.bar(x + (i - 1) * w, vals, w, label=lab, color=c)
        _label(ax, b, "{:.2f}")
    for xi, (n, t) in enumerate(zip(naive, tiled)):
        ax.annotate(f"{n/t:.2f}×", xy=(xi + 0.4, t + 0.6), fontsize=9,
                    fontweight="bold", color=RED, ha="center")
    ax.set_xticks(x); ax.set_xticklabels(SIZES)
    ax.set_ylabel("ms / frame", fontweight="bold")
    ax.set_title("Blur + composite, kernels only — shared-memory tiling "
                 "wins 2.36×", fontweight="bold", fontsize=10)
    ax.legend(frameon=False, fontsize=9)
    _save(fig, "fig_final_blur.png")


# ── 5 & 6. pictures, from the real pipeline ─────────────────────────────────

def _sequence(n, size=(320, 240)):
    """Frames for the mechanism figures: CDnet if present, else synthetic.

    Synthetic frames are honest here as long as they are labelled: they show
    what each stage does to a moving object, which is all these two figures
    claim. Quantitative claims never rest on them -- those come from
    `bench_post.py` and, for quality, from CDnet.
    """
    hw = os.environ.get("HIGHWAY_DIR", os.path.join(REPO, "highway"))
    if not os.path.isdir(os.path.join(hw, "input")):
        # the team's Hugging Face mirror; anonymous HTTPS, ~27 MB
        try:
            import urllib.request, zipfile, socket
            socket.setdefaulttimeout(30)
            url = ("https://huggingface.co/datasets/haiduonghuynhle/"
                   "changedetection-2012-highway/resolve/main/highway.zip")
            zp = os.path.join(REPO, "highway.zip")
            urllib.request.urlretrieve(url, zp)
            with zipfile.ZipFile(zp) as z:
                z.extractall(REPO)
        except Exception:
            pass
    if os.path.isdir(os.path.join(hw, "input")):
        out = [cv2.imread(os.path.join(hw, "input", f"in{i:06d}.jpg"))
               for i in range(1, n + 1)]
        out = [f for f in out if f is not None]
        if out:
            return out, "CDnet highway"

    # The two .mp4 clips beside the repo are deliberately NOT used here. They
    # are classroom footage of a nearly-stationary person, which a background
    # subtractor correctly absorbs into the background -- the mask comes back
    # as fragments and the composite blurs the subject. That is expected
    # behaviour and it is exactly why the project retargeted from webcam to
    # traffic, but as a slide it reads as broken code. Synthetic frames show
    # the mechanism honestly; the real highway imagery is on the dataset
    # figure, which is where real data belongs.

    import bench_post
    return bench_post.make_frames(n, size), "synthetic traffic (illustrative only)"


def fig_dataset():
    """The task itself, on real CDnet highway.

    Cropped out of figs/fig_s3_dataset.png, which was rendered while the CDnet
    download still worked. The source frames are gone; this is the only real
    highway imagery left in the repository, so it is reused rather than
    regenerated.
    """
    src = os.path.join(OUT, "fig_s3_dataset.png")
    if not os.path.exists(src):
        print("  (skipped fig_final_dataset.png — fig_s3_dataset.png missing)")
        return
    img = cv2.cvtColor(cv2.imread(src), cv2.COLOR_BGR2RGB)
    img = img[18:]                      # drop the burned-in yellow captions
    h, w = img.shape[:2]
    third = w // 3
    fig, ax = plt.subplots(1, 2, figsize=(9, 2.6))
    _panel(ax[0], img[:, :third], "input frame")
    _panel(ax[1], img[:, 2 * third:], "hand-labelled ground truth")
    fig.suptitle("CDnet 2014 baseline/highway", fontsize=7, y=0.02,
                 color="#666666")
    _save(fig, "fig_final_dataset.png")


def _panel(ax, img, title, gray=False):
    ax.imshow(img, cmap="gray" if gray else None)
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def fig_chain():
    from gmm_mask import GMM_Mask_Numba
    from utils.post_processing import fill_holes, threshold_bg_prob

    frames, src = _sequence(560)
    h, w = frames[0].shape[:2]
    m = GMM_Mask_Numba(h, w)
    for bgr in frames:
        mask, bg_prob, _ = m.apply(cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb))

    raw = np.asarray(mask)
    thr = threshold_bg_prob(np.asarray(bg_prob))
    med = cv2.medianBlur(thr, 5)
    fil = fill_holes(med)

    stages = [("input", cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), False),
              ("raw MOG2", raw, True), ("threshold bg_prob < 0.5", thr, True),
              ("median 5", med, True), ("+ fill_holes", fil, True)]
    fig, ax = plt.subplots(1, 5, figsize=(12, 2.3))
    for a, (t, im, g) in zip(ax, stages):
        _panel(a, im, t, g)
    fig.suptitle(f"source: {src}", fontsize=7, y=0.03, color="#666666")
    _save(fig, "fig_final_chain.png")


def fig_result():
    from gmm_mask import GMM_Mask_Numba
    from settings import BLUR_KSIZE, BLUR_SIGMA
    from utils.post_processing import background_blur, refine_mask

    frames, src = _sequence(600)
    h, w = frames[0].shape[:2]
    m = GMM_Mask_Numba(h, w)
    for bgr in frames:
        mask, bg_prob, _ = m.apply(cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb))

    refined = refine_mask(np.asarray(mask), bg_prob=np.asarray(bg_prob))
    out = background_blur(bgr, refined, BLUR_KSIZE, BLUR_SIGMA)
    # the model was fed YCrCb, so its learned means are YCrCb
    bg = cv2.cvtColor(m.background_image(), cv2.COLOR_YCrCb2RGB)

    fig, ax = plt.subplots(1, 4, figsize=(11, 2.4))
    for a, (t, im, g) in zip(ax, [
            ("original", cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), False),
            ("foreground mask", refined, True),
            ("learned background", bg, False),
            ("composite — sharp vehicles, blurred road",
             cv2.cvtColor(out, cv2.COLOR_BGR2RGB), False)]):
        _panel(a, im, t, g)
    fig.suptitle(f"source: {src}", fontsize=7, y=0.03, color="#666666")
    _save(fig, "fig_final_result.png")


def fig_q8():
    """The discovery, drawn: OpenCV's quantiser is cumulative, not per-tap."""
    import cv2 as _cv
    k = _cv.getGaussianKernel(15, 5.0).ravel()
    cum = np.diff(np.concatenate(([0.0], np.round(256 * np.cumsum(k))))).astype(int)
    per = np.rint(k * 256).astype(int)
    x = np.arange(15)
    diff = np.flatnonzero(cum != per)

    fig, ax = plt.subplots(figsize=(7.4, 3.1))
    ax.bar(x - 0.2, cum, 0.4, label="cumulative  (what OpenCV does)", color=BLUE)
    ax.bar(x + 0.2, per, 0.4, label="per-tap  (the obvious guess)", color=GREY)
    for i in diff:
        ax.annotate("", xy=(i, max(cum[i], per[i]) + 2.6), xytext=(i, max(cum[i], per[i]) + 0.4),
                    arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.4))
        ax.text(i, max(cum[i], per[i]) + 3.0, f"{cum[i]}/{per[i]}", ha="center",
                fontsize=7.5, fontweight="bold", color=RED)
    ax.set_xticks(x)
    ax.set_xlabel("tap", fontweight="bold")
    ax.set_ylabel("weight  (denominator 256)", fontweight="bold")
    ax.set_title("Both sum to 256. They differ at 4 of 15 taps — and only one "
                 "matches cv2.", fontweight="bold", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_ylim(0, 33)
    _save(fig, "fig_final_q8.png")


def fig_fill():
    """Why the flood fill stays on the CPU: the pass count, not the clock."""
    passes = [132, 266, 399, 593]
    labels = ["240p", "480p", "720p", "1080p"]
    fig, ax = plt.subplots(figsize=(6.4, 3.1))
    b = ax.bar(labels, passes, 0.5, color=[GREY, GREY, GREY, RED])
    _label(ax, b, "{:.0f}")
    ax.set_ylabel("dependent full-frame passes", fontweight="bold")
    ax.set_title("Morphological reconstruction needs one grid-wide dilate per "
                 "pixel of travel", fontweight="bold", fontsize=9.5)
    ax.set_ylim(0, 700)
    ax.text(3, 292, "same count on\nboth machines\nwe measured",
            ha="center", fontsize=8, style="italic", color=RED)
    ax.text(0.5, -0.32, "One seeded synthetic mask per size. CPU reconstruction — "
            "no CUDA implementation was measured.",
            transform=ax.transAxes, ha="center", fontsize=7, color="#767676")
    _save(fig, "fig_final_fill.png")


if __name__ == "__main__":
    print(f"writing to {OUT}")
    for fn in (fig_speedup, fig_perstage, fig_bus, fig_blur,
               fig_dataset, fig_chain, fig_result, fig_q8, fig_fill):
        fn()
