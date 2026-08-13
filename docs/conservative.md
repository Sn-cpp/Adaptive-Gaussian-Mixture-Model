# The stationary subject, and what actually fixes it

MOG2 detects **motion**, not people. On CDnet `highway` that distinction never
comes up — cars are always moving — and the model scores F1 0.9338. On a webcam
clip of somebody sitting at a desk it is the whole story: the subject is folded
into the background within a second or two and the mask goes empty.

This is not a tuning problem, and it is not a post-processing problem. Post
-processing refines a mask; it cannot put back a person the mask no longer
contains.

## The two mechanisms, separated

Zivkovic's MOG2 updates **every** pixel every frame. A subject who holds still
therefore teaches the background model that they *are* the background. ViBe
(Barnich & Van Droogenbroeck, 2011) answers this with a **conservative update**:
a pixel only feeds the model if it was classified *background*. That is the
`conservative=True` option on every model here — `alpha = 0` and `prune = 0`
wherever the previous frame said foreground, so the Gaussians at that pixel are
frozen entirely.

But protection only protects what was **detected**. A subject who was already
sitting in front of the camera when the program started is inside mode 0 before
anything runs, is never classified foreground, and so is never protected. For
that, the model has to see the room empty first — a **clean plate**.

Both are necessary. Neither is sufficient.

## Measurement

Composite sequence, exact ground truth: real `highway` frames as the background
(static camera, real sensor noise), a real crop of the person from
`LTSSUD-Test.mp4` pasted through an ellipse, with per-frame Gaussian noise at
2.5 grey levels on the paste. The subject enters at frame 60 and then never
moves again. 200 scored frames; IoU is measured on the subject only, because
the highway cars are genuine moving foreground and detecting them is correct.

Reproduce with `python docs/experiment_cleanplate.py`.

| configuration                  |   IoU |  first 20 frames | last 100 frames | frames lost (IoU<0.2) |
| ------------------------------ | ----: | ---------------: | --------------: | --------------------: |
| present from frame 0, plain    | 0.000 |            0.000 |           0.000 |               200/200 |
| present from frame 0, conserv. | 0.000 |            0.000 |           0.000 |               200/200 |
| clean plate, plain MOG2        | 0.070 |            0.696 |           0.000 |               186/200 |
| **clean plate + conservative** | **0.965** |        **0.990** |       **0.953** |             **0/200** |

![clean plate](conservative_cleanplate.png)

Row 2 is the `LTSSUD-Test.mp4` situation: the subject box is empty at every
frame. Row 3 shows plain MOG2 detecting the subject cleanly on entry (frame 62)
and losing it by frame 90 — absorbed in about one second. Row 4 holds it for
the full 190 frames.

## Cost on the moving-object case

Conservative update is not free. CDnet `highway`, frames 470-1700, YCrCb,
through `mask_refiner`:

| | F1 | IoU | P | R |
| --- | ---: | ---: | ---: | ---: |
| conservative off | 0.9338 | 0.8758 | 0.9873 | 0.8858 |
| conservative on  | 0.9283 | 0.8661 | 0.8816 | 0.9801 |

It costs **nothing measurable in time**. Tesla T4, `input.mp4` upscaled to
1080p, `GMM_CUDA`, five interleaved A/B repeats of 30 timed frames each:

    plain          9.74  9.50  9.32  9.21 10.08   median 9.50 ms
    conservative   9.52  9.29  9.22 10.10  9.84   median 9.52 ms

+0.2%, well inside the spread of a shared T4. It is one comparison and three
register writes per thread, and it launches no extra kernel and allocates
nothing — the previous frame's decision is already in `mask`, and the thread
that reads it is the thread that wrote it. Interleave the A/B repeats if you
re-measure this: a single cold pass reported +37% purely from host contention.

What it does cost is 0.6 F1 on highway: a car's trailing edge stays protected
longer than the ground truth says it should, so recall rises to 0.98 and
precision falls. That is the correct trade for a webcam and the wrong one for
traffic, which is why the option is **off by default** —
`settings.MOG2_CONSERVATIVE_UPDATE` — and turned on only by `main.py`. With it
off the models remain bit-exact against `cv2.BackgroundSubtractorMOG2`, which
is what `tests/` asserts.

All four backends were checked against each other on a real T4, with the option
both off and on: the sequential Python spec, the Numba CPU kernel, the Numba
CUDA kernel and the CuPy `RawKernel` produce **identical masks, zero pixels
differing**, with `bg_prob` agreeing to 2.09e-07 — float32 rounding. Re-run at
each HEAD that touches a kernel; an earlier version of this file quoted a check
made *before* `f2bdaf4` restored the pruning `GMM_CUPY` was skipping, which is
exactly the window in which the claim was false.

## The trap in "release by classification"

Freezing a pixel is the easy half. Releasing it is where this goes wrong, and
the obvious design — release when the frozen model classifies the pixel as
background again — is a trap.

It works for the case it was designed for: the subject walks away, the pixel
returns to the colour it was frozen at, it matches, it resumes updating. It
fails absolutely for a *global* appearance change. A webcam auto-exposing turns
most of the frame foreground in a single frame; every one of those pixels then
freezes at the **old** exposure; and none of them can ever track the new one,
because release would need `dist2 < Tb * var` with `var <= var_max`, which at
the defaults a uniform shift beyond about 20 levels per channel can never
satisfy — at any learning rate, for ever. Measured on `highway` with a uniform
+25 grey-level step at frame 150, raw mask coverage:

| | +50 frames | +150 | +400 | +700 |
| --- | ---: | ---: | ---: | ---: |
| plain MOG2 | 0.652 | 0.127 | 0.005 | 0.139 |
| release-by-classification | 0.904 | 0.855 | 0.845 | **0.826** |

Plain MOG2 reabsorbs the step within about 150 frames. The naive protected
version is still 83% foreground 700 frames later — a frozen screen, on a demo
whose input device auto-exposes.

Two mechanisms fix it, and both are needed:

- **`MOG2_PROTECT_EXIT`** — a second, looser threshold. Protection is held only
  while the pixel is still *far* from its dominant frozen mode. A global
  photometric shift is a moderate distance; a person against a wall is a large
  one.
- **`MOG2Base.CONSERVATIVE_MAX_COVERAGE`** — a frame-wide backstop, because the
  per-pixel rule measures distance against each mode's *own* variance and a
  well-converged background pixel has almost none, so a big step still looks
  far everywhere at once. A person is not most of the frame; an exposure step
  is. Above this fraction, protection is dropped for one frame.

Swept together against everything they trade off (plain MOG2 scores 0.028 on
the latch column):

| `Te` | latch, +25 step | clean-plate IoU | webcam coverage / frames lost |
| ---: | ---: | ---: | --- |
| 0 (backstop only) | 0.199 | 0.961 | 31.2% / 9 |
| 36 | 0.115 | 0.959 | 30.0% / 9 |
| **64** *(shipping)* | **0.083** | **0.953** | 29.5% / 9 |
| 100 | 0.047 | 0.936 | 29.2% / 9 |
| 144 | 0.030 | 0.888 | 29.1% / 9 |

64 keeps the capability the feature exists for and brings the latch within a
small multiple of plain MOG2. `test_a_global_exposure_change_does_not_latch_the_frame`
pins it, and asserts the failure still reproduces with both mechanisms disabled
so the guard cannot quietly become vacuous.

What remains: a *local* background change while protected — a chair pushed
aside — where the new appearance sits beyond the exit threshold. That pixel
stays foreground until something moves through it. ViBe answers this with
random spatial propagation, not implemented here.

## What it does *not* fix

On `LTSSUD-Test.mp4` itself — subject present from frame 0, no clean plate
available — conservative update roughly doubles coverage and cuts the frames
where the subject has vanished from 137 to 35 out of 310:

| configuration | coverage | <5% | >60% | blobs |
| --- | ---: | ---: | ---: | ---: |
| plain, alpha ramp | 8.9% | 137 | 1 | 36.6 |
| plain, alpha 0.002 | 12.9% | 60 | 1 | 52.8 |
| conservative, alpha ramp | 18.1% | 35 | 1 | 57.3 |
| conservative, alpha 0.002 | 20.6% | 21 | 1 | 48.4 |

but look at what those extra pixels are:

![webcam](conservative_webcam.png)

It is an **outline**, not a silhouette. Small head movements reveal the edges
of the face, those get protected and stay; the interior of a cheek or a shirt
never changes appearance when it shifts a few pixels, so it is never detected,
never protected, and stays in the background model where frame 0 put it.

Two things close most of that gap, and both are in `mask_refiner` now —
`bg_prob` thresholding and a scaled CLOSE. See
[post-processing.md](post-processing.md); the combination takes coverage on
this clip from 8.9% to 27.8% against a subject that occupies 25-30%, and cuts
the frames where the subject has effectively vanished from 137 to 10 of 310.
What it cannot do is invent the interior when the outline itself is missing,
which is why the clean plate still matters.

> **Correction.** An earlier version of this file said a large CLOSE "costs
> precision 0.90 -> 0.81 on highway and empties the mask on 6 frames". That was
> wrong, and the wrongness mattered: those costs belong to the `OPEN` and the
> final `dilate` of the old chain, not to the CLOSE. Measured separately,
> `median + OPEN` scores F1 0.9182 and produces all 6 empty frames on its own,
> while `median + CLOSE15 + fill` scores 0.9542 with none. A binary closing is
> extensive — it cannot remove foreground — so it could never have emptied a
> mask, and the claim should not have survived a moment's thought.

## Practical consequence

`main.py --clean-plate 48` trains on the first 48 frames (2 s at 24 fps) before
showing anything; step out of shot while it does. There is nothing special
about those frames inside the model — they are ordinary steps that happen to
see an empty room, so mode 0 ends up holding the real background. Conservative
update, on by default in `main.py`, is what then keeps it intact.

Without a clean plate the problem is underdetermined, and it is worth saying so
plainly rather than staging a demo where the subject has to keep waving: a
stationary person is statistically indistinguishable from persistent background
under the MOG2 observation model. Recovering them needs either an empty-scene
observation or a semantic prior. Neither is a defect in the implementation.

## References

- Barnich, O. & Van Droogenbroeck, M. (2011). *ViBe: A Universal Background
  Subtraction Algorithm for Video Sequences.* IEEE TIP 20(6). — conservative
  update, and the random spatial propagation that this implementation leaves out.
- Zivkovic, Z. (2004). *Improved Adaptive Gaussian Mixture Model for Background
  Subtraction.* ICPR. — the model being modified.
- Braham, M., Piérard, S. & Van Droogenbroeck, M. (2017). *Semantic Background
  Subtraction.* ICIP. — the other way out: a semantic mask gating the update.
