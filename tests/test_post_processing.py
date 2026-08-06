"""Post-processing helpers: mask refinement and the blur composite."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np


class TestPostProcessing:

    def test_mask_refiner_preserves_shape(self, small_dims):
        from utils.post_processing import mask_refiner
        H, W = small_dims
        mask = np.random.randint(0, 2, (H, W), dtype=np.uint8) * 255
        refined = mask_refiner(mask)
        assert refined.shape == mask.shape
        assert refined.dtype == np.uint8

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
