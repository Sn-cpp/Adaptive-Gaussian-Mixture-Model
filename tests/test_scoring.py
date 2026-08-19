"""The CDnet scoring protocol, on a fixture whose answer is known by hand.

`eval_highway.py` produces the only quality numbers in this project, and until
now nothing tested it. Equivalence tests protect against a backend changing the
mask; they say nothing about whether the *metric* is computed correctly, and a
scorer that quietly counts don't-care pixels will report a confident, wrong,
reproducible F1 forever.

CDnet labels shadows 50 and object boundaries 170 and defines both as
*don't care*, and it ships an ROI mask. Counting either is, in the proposal's
own words, the easiest way to publish a wrong number. These tests build a 6x6
frame where every category appears exactly once and the true TP/FP/FN are
countable by eye.

No GPU, no dataset — this is the part of the quality claim that can be checked
without CDnet, which matters because the CDnet download is offline.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np


def confusion(pred, gt, roi):
    """The arithmetic eval_highway.py performs, isolated.

    Mirrors eval_highway.score(): `valid` keeps only ground truth that is
    exactly 0 or 255 inside the ROI, and TP/FP/FN are counted over that subset.
    """
    valid = roi & ((gt == 255) | (gt == 0))
    g = (gt == 255) & valid
    p = (pred == 255) & valid
    return int((p & g).sum()), int((p & ~g).sum()), int((~p & g).sum())


def f1_iou(tp, fp, fn):
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return 2 * p * r / max(p + r, 1e-9), tp / max(tp + fp + fn, 1)


def fixture():
    """A 3x4 grid of labelled cells; the expected counts are in the test."""
    gt = np.array([
        [255, 255,   0,   0],      # 2 true fg, 2 true bg
        [ 50,  50, 170, 170],      # shadow + unknown: all don't care
        [255,   0, 255,   0],      # inside/outside ROI, set below
    ], np.uint8)
    roi = np.ones((3, 4), bool)
    roi[2, 2:] = False             # last two cells of row 2 are outside the ROI
    return gt, roi


def test_dont_care_labels_are_excluded_from_every_count():
    gt, roi = fixture()
    pred = np.full((3, 4), 255, np.uint8)      # call everything foreground

    tp, fp, fn = confusion(pred, gt, roi)
    # Scorable cells: row 0 (4 cells) + row 2 columns 0-1 (2 cells) = 6.
    # Of those, foreground truth is (0,0), (0,1), (2,0) = 3.
    assert tp == 3, "true positives miscounted"
    assert fp == 3, "false positives miscounted"
    assert fn == 0
    assert tp + fp + fn == 6, (
        "the shadow (50), unknown (170) and out-of-ROI cells were scored; "
        "CDnet defines all three as don't care")


def test_out_of_roi_pixels_are_excluded_even_when_the_label_is_0_or_255():
    """The ROI is a second, independent mask. A pixel can carry a perfectly
    good 0/255 label and still be outside the scored region."""
    gt, roi = fixture()
    pred = np.zeros((3, 4), np.uint8)
    pred[2, 2] = 255                    # a foreground guess outside the ROI

    tp, fp, fn = confusion(pred, gt, roi)
    assert fp == 0, "a pixel outside the ROI was counted as a false positive"


def test_a_perfect_prediction_scores_exactly_one():
    gt, roi = fixture()
    pred = np.where(gt == 255, np.uint8(255), np.uint8(0))
    f1, iou = f1_iou(*confusion(pred, gt, roi))
    assert f1 == 1.0 and iou == 1.0


def test_an_empty_prediction_scores_zero_rather_than_dividing_by_zero():
    gt, roi = fixture()
    f1, iou = f1_iou(*confusion(np.zeros((3, 4), np.uint8), gt, roi))
    assert f1 == 0.0 and iou == 0.0


def test_counting_dont_care_pixels_would_change_the_score():
    """The guard is only meaningful if getting it wrong is visible.

    If excluding shadows made no difference, the exclusion would be untested in
    practice however carefully it was written.
    """
    gt, roi = fixture()
    pred = np.full((3, 4), 255, np.uint8)
    correct = f1_iou(*confusion(pred, gt, roi))[0]

    naive_valid = np.ones_like(roi)                       # score everything
    g = (gt == 255) & naive_valid
    p = (pred == 255) & naive_valid
    wrong = f1_iou(int((p & g).sum()), int((p & ~g).sum()), int((~p & g).sum()))[0]

    assert abs(correct - wrong) > 0.05, (
        "this fixture cannot distinguish correct scoring from naive scoring")


def test_eval_highway_uses_the_same_rule_this_file_tests():
    """Pin the fixture to the real scorer rather than to a copy of it.

    A test that re-implements the thing under test drifts away from it. This
    reads the module and asserts the two defining expressions are still there,
    so a change to the protocol in eval_highway.py fails here.
    """
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', 'eval_highway.py')).read()
    assert "roi & ((gt == 255) | (gt == 0))" in src, (
        "eval_highway.py no longer builds `valid` the way this fixture assumes")
    assert "T0, T1 = 470, 1700" in src, (
        "the temporal window changed; CDnet's own temporalROI for highway is "
        "470-1700 and a different window moves F1 by up to 5 points")
