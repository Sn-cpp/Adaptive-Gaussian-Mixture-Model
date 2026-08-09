"""Post-processing helpers: mask refinement and the blur composite."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import cv2
import numpy as np


class TestPostProcessing:

    def test_mask_refiner_preserves_shape(self, small_dims):
        from utils.post_processing import mask_refiner
        H, W = small_dims
        mask = np.random.randint(0, 2, (H, W), dtype=np.uint8) * 255
        refined = mask_refiner(mask)
        assert refined.shape == mask.shape
        assert refined.dtype == np.uint8

    def test_fill_holes_closes_an_enclosed_gap(self, small_dims):
        from utils.post_processing import fill_holes
        H, W = small_dims
        mask = np.zeros((H, W), np.uint8)
        mask[8:H - 8, 8:W - 8] = 255          # solid block...
        mask[H // 2 - 3:H // 2 + 3, W // 2 - 3:W // 2 + 3] = 0   # ...with a hole

        filled = fill_holes(mask)

        assert filled[H // 2, W // 2] == 255, "enclosed hole was not filled"
        # and the silhouette did not grow: every pixel outside the block is background
        outside = np.ones((H, W), bool)
        outside[8:H - 8, 8:W - 8] = False
        assert not filled[outside].any(), "fill_holes inflated the mask"

    def test_fill_holes_leaves_background_connected_to_the_border(self, small_dims):
        """A concave notch open to the outside is background, not a hole."""
        from utils.post_processing import fill_holes
        H, W = small_dims
        mask = np.zeros((H, W), np.uint8)
        mask[8:H - 8, 8:W - 8] = 255
        mask[8:H // 2, 8:12] = 0              # notch cut in from the left edge of the block
        # carve a channel from the notch to the image border so it stays connected
        mask[8:H // 2, 0:12] = 0

        filled = fill_holes(mask)

        assert filled[H // 4, 9] == 0, "background reachable from the border was filled in"

    def test_mask_refiner_removes_holes_without_inflating(self, small_dims):
        """The property the old CLOSE+dilate recipe failed: no holes, no growth.

        On CDnet highway the old chain scored F1 0.7971 against 0.8929 for this
        one, because it traded precision away to close holes by brute force.
        """
        from utils.post_processing import mask_refiner
        H, W = small_dims
        mask = np.zeros((H, W), np.uint8)
        mask[10:H - 10, 10:W - 10] = 255
        mask[H // 2 - 4:H // 2 + 4, W // 2 - 4:W // 2 + 4] = 0

        out = mask_refiner(mask)

        assert out[H // 2, W // 2] == 255, "hole survived"
        grown = int(((out > 0) & (mask == 0)).sum())
        hole_area = 8 * 8
        assert grown <= hole_area, f"mask grew by {grown}px beyond filling the hole"

    def test_background_subtractor_preserves_shape(self, small_dims):
        from utils.post_processing import background_subtractor
        H, W = small_dims
        frame = np.random.randint(0, 256, (H, W, 3), dtype=np.uint8)
        mask = np.random.randint(0, 2, (H, W), dtype=np.uint8) * 255
        result = background_subtractor(frame, mask)
        assert result.shape == frame.shape
        assert result.dtype == np.uint8

    def test_background_subtractor_keeps_dark_foreground_pixels(self, small_dims):
        """Foreground must survive untouched even where a channel is 0.

        Using the foreground image as the mask (instead of the mask itself)
        silently blurs those channels, which is invisible on a bright test
        frame but wrecks dark hair and clothing on real footage.
        """
        from utils.post_processing import background_subtractor
        H, W = small_dims
        frame = np.full((H, W, 3), 200, dtype=np.uint8)
        mask = np.zeros((H, W), dtype=np.uint8)
        mask[H // 4:3 * H // 4, W // 4:3 * W // 4] = 255
        # Subject is dark enough that two of its channels are exactly 0.
        frame[mask > 0] = (0, 0, 30)

        result = background_subtractor(frame, mask)

        assert np.array_equal(result[mask > 0], frame[mask > 0])
        # ...and the background really did get blurred (guards against a
        # trivially-passing implementation that just returns the frame).
        assert not np.array_equal(result[mask == 0], frame[mask == 0])


class TestPostProcessingPackage:
    """The post_processing/ backends against OpenCV and against utils/.

    Same arrangement as the GMM models: post_processing/cpu is the plain-Python
    specification, and it has to agree with the fast path in utils/.
    """

    def test_erode_matches_opencv(self):
        from post_processing.cpu.post_processing_cpu import erode2d
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)).astype(np.float64)
        mask = np.zeros((24, 24), np.uint8)
        mask[6:18, 6:18] = 255
        assert np.array_equal(erode2d(mask, k),
                              cv2.erode(mask, k.astype(np.uint8)))

    def test_dilate_matches_opencv(self):
        from post_processing.cpu.post_processing_cpu import dilate2d
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)).astype(np.float64)
        mask = np.zeros((24, 24), np.uint8)
        mask[6:18, 6:18] = 255
        assert np.array_equal(dilate2d(mask, k),
                              cv2.dilate(mask, k.astype(np.uint8)))

    def test_reference_fill_holes_matches_the_fast_one(self):
        from post_processing.cpu.post_processing_cpu import fill_holes as slow
        from utils.post_processing import fill_holes as fast
        mask = np.zeros((24, 24), np.uint8)
        mask[6:18, 6:18] = 255
        mask[11:14, 11:14] = 0
        assert np.array_equal(slow(mask), fast(mask))

    def test_apply_runs_and_keeps_the_foreground(self):
        from post_processing import PostProcessingCPU
        rng = np.random.default_rng(0)
        frame = rng.integers(0, 256, (24, 24, 3), dtype=np.uint8)
        mask = np.zeros((24, 24), np.uint8)
        mask[6:18, 6:18] = 255
        mask[11:14, 11:14] = 0            # a hole, to exercise fill_holes

        pp = PostProcessingCPU()
        out, seconds = pp.apply(frame, mask)

        assert out.shape == frame.shape and out.dtype == frame.dtype
        assert seconds >= 0
        assert pp.refined_mask[12, 12] == 255, "hole was not filled"
        fg = pp.refined_mask > 0
        assert np.array_equal(out[fg], frame[fg]), "foreground was blurred"
        assert not np.array_equal(out[~fg], frame[~fg]), "background was not blurred"
