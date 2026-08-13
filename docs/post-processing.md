# Post-processing: what the mask needed, measured

Everything here is CDnet `highway`, frames 470-1700 (the full `temporalROI`,
1231 scored frames), YCrCb input, `GMM_CPU_NUMBA`, conservative update **off**,
scoring only ground-truth 0/255 pixels inside `ROI.bmp`. Reproduce the ablation
with the scripts named at the bottom.

## The table

| stage | F1 | IoU | P | R | empty |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw MOG2 | 0.8607 | 0.7554 | 0.9032 | 0.8220 | 0 |
| median 5 only | 0.9248 | 0.8602 | 0.9872 | 0.8699 | 0 |
| median + fill_holes *(previous shipping path)* | 0.9338 | 0.8758 | 0.9873 | 0.8858 | 0 |
| median + OPEN 5 | 0.9182 | 0.8487 | 0.9919 | 0.8546 | **6** |
| median + OPEN + CLOSE 15 x2 | 0.9255 | 0.8614 | 0.9311 | 0.9200 | **6** |
| old chain (OPEN + CLOSE x2 + dilate) | 0.8748 | 0.7775 | 0.8121 | 0.9480 | **6** |
| median + CLOSE 11 + fill | 0.9511 | 0.9067 | 0.9805 | 0.9234 | 0 |
| median + CLOSE 15 + fill | 0.9542 | 0.9123 | 0.9709 | 0.9379 | 0 |
| median + CLOSE 19 + fill | 0.9496 | 0.9041 | 0.9517 | 0.9475 | 0 |
| **bg_prob < 0.5 + median + fill** *(shipping)* | **0.9633** | **0.9292** | 0.9418 | 0.9857 | 0 |
| bg_prob < 0.3 + median + fill | 0.9636 | 0.9297 | 0.9434 | 0.9847 | 0 |
| bg_prob < 0.5 + median + CLOSE 15 + fill | 0.9316 | 0.8719 | 0.8750 | 0.9959 | 0 |
| GrabCut refinement (`graphcut/`) | 0.9552 | 0.9142 | 0.9804 | 0.9312 | 5 |

## bg_prob beats MOG2's own decision by 3.0 F1, for free

MOG2 classifies a pixel as background if the first mode it matches lies inside
the background set — *any* match, however little weight that mode carries. In
the kernel, `background` is exactly `bg_prob > 0`, where `bg_prob` is the summed
weight of every background mode that matched.

Thresholding that sum at 0.5 instead asks a stricter question: do the modes this
pixel matches actually carry half the background weight? Matches against
spurious low-weight modes stop counting, recall goes 0.886 -> 0.986, precision
0.987 -> 0.942, and F1 0.9338 -> 0.9633.

Three things make this more than a lucky threshold:

- It is **flat** between 0.3 and 0.5 (0.9636 against 0.9633), so it is not tuned
  to a knife edge. It only collapses past 0.9, where it starts eating real
  background.
- It is **free**. Every backend already computed `bg_prob` and discarded it; the
  change is one comparison per pixel instead of one.
- It **beats the graph cut**. The GrabCut refinement in `graphcut/` — dual
  full-covariance colour GMMs and a parallel push-relabel max-flow, verified
  exact against Boykov-Kolmogorov — scores 0.9552 at 31 ms/frame at 240p and
  ~900 ms at 1080p, with 5 entirely empty masks. This scores higher, never
  empties the mask, and costs nothing measurable.

The credit for looking at `bg_prob` as an output rather than an intermediate
belongs to Đức Tín's `main_contour.py` on the `push_relabel_remade` branch,
which proposed using it as a soft alpha for the composite.

## The OPEN was the problem, not the CLOSE

The old `OPEN + CLOSE x2 + dilate` chain was rejected for producing 6 entirely
empty masks and dropping precision to 0.81, and the CLOSE was blamed. That was
wrong. Split apart:

- `median + OPEN` alone scores 0.9182 and produces **all 6** empty frames. An
  erode-first pass deletes any structure thinner than the kernel, and a small
  car or a thin limb is exactly that.
- The final `dilate` is what takes precision from 0.9311 to 0.8121.
- A CLOSE on its own is the **second-best** refinement measured here.

A binary closing is extensive — its output always contains its input — so it
could never have emptied a mask, and the claim should not have survived a
moment's thought. `pipeline.py` and `CUDAPipeline` were running that same
erode-first OPEN; both now dilate first. Same two kernels, same cost, opposite
order.

## When to use the CLOSE

The CLOSE kernel has to be small against the **object**, and `close_ksize_for`
can only scale it against the **frame**. Those coincide on a webcam and do not
on a traffic camera:

- `LTSSUD-Test.mp4` at 480x270, conservative on, subject truly 25-30% of frame:
  `bg_prob` alone gives 19.4% coverage, 29 of 310 frames essentially empty, 55
  connected components. Adding CLOSE 17: **27.8% coverage, 10 empty frames, 10
  components**.
- `highway`: a car is ~20 px in a 240 px frame, so a 15 px kernel is most of a
  car. F1 drops 0.9633 -> 0.9316.

So `main.py` turns it on — it is a webcam application — and the CDnet scoring
path leaves it off. That is a stated assumption about the application, not a
tuned constant, and it is the honest way to present it.

**The failure mode to keep in view:** a CLOSE wider than the gap between two
objects merges them, and F1 will not tell you. A 15x15 CLOSE across a 10-pixel
aisle between two people fills 98.6% of the aisle while F1 stays near 0.96.
`CLOSE_KSIZE_MAX` exists for that reason.

## RECT, not ELLIPSE

| resolution | k | ELLIPSE | RECT |
| --- | ---: | ---: | ---: |
| 320x240 | 15 | 0.34 ms | 0.04 ms |
| 854x480 | 27 | 6.26 ms | 0.36 ms |
| 1920x1080 | 61 | 22.27 ms | 3.59 ms |

A rectangle is separable into a 1xk and a kx1 pass and OpenCV exploits that; an
ellipse is not. 22 ms at 1080p would eat most of the 27 ms frame budget behind
37 FPS. RECT costs 0.0026 F1 (0.9516 against 0.9542) and 6x less time. It is the
same separability argument as the Gaussian blur elsewhere in this project.

The visible cost is that a 61-wide square leaves staircase edges on the mask
boundary. In the blur composite they are largely hidden by the blur transition
itself, but an octagonal approximation (horizontal, vertical and two diagonal
passes) would fix it properly, and a van Herk / Gil-Werman kernel would make the
cost independent of `k` altogether. Neither is implemented.

## Reproducing

    python -m pytest tests/test_post_processing.py     # the properties, not the numbers

The ablation scripts live outside the repo (they need the CDnet `highway`
sequence, which is not vendored): `morph_ablation.py`, `bgprob_sweep.py`,
`compare_all.py`, `webcam_all.py`. Point them at a `highway/` directory holding
`input/`, `groundtruth/`, `ROI.bmp`.
