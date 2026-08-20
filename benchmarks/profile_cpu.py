"""Profile the CPU baseline — the course's Part 5 requirement, kept runnable.

    python benchmarks/profile_cpu.py

The project's design decisions all cite a profile ("81.8% of the frame was on
the host" is what motivated v1), but that number came from a one-off session
that no committed script reproduces. This file is the reproducible version:
cProfile over the sequential pipeline on synthetic frames, top functions by
cumulative time, and the one-line conclusion the proposal's bottleneck
analysis is built on.

What it shows, on any machine: the interpreted model update dominates the
sequential pipeline outright (~99.8% of cumulative time here). That is the
profile behind the *first* kernel — the model. The later kernels (threshold,
median, colour conversion, blur) were justified by a different measurement:
the per-stage pipeline profile in RESULTS-T4.md §4, taken after the model had
already moved and the bottleneck had shifted to the host stages.
"""
import cProfile
import io
import os
import pstats
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.cpu_baseline import load_data, run_cpu

if __name__ == "__main__":
    frames, _ = load_data(n_frames=6)

    profiler = cProfile.Profile()
    profiler.enable()
    run_cpu(frames)
    profiler.disable()

    buf = io.StringIO()
    stats = pstats.Stats(profiler, stream=buf)
    stats.sort_stats("cumulative")
    stats.print_stats(20)
    print(buf.getvalue())

    # The conclusion the proposal's bottleneck analysis rests on, computed
    # from this very run rather than asserted.
    total = stats.total_tt
    model_t = max((ct for (f, l, n), (cc, nc, tt, ct, cal) in stats.stats.items()
                   if n == "mog2_step"), default=0.0)
    print(f"model update (mog2_step) share of total: {model_t / total:.1%} — "
          "this is the stage every GPU version accelerates first")
