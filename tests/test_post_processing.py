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

    def test_fill_holes_when_the_subject_touches_the_top_left_corner(self, small_dims):
        """Seeding the flood at pixel (0, 0) only works while that corner is
        background. A shoulder in the top-left of a webcam frame put foreground
        there, the flood could not start, nothing was reachable, and the
        complement was the whole image — every pixel declared foreground, so
        nothing got blurred at all."""
        from utils.post_processing import fill_holes
        H, W = small_dims
        mask = np.zeros((H, W), np.uint8)
        mask[0:H // 2, 0:W // 2] = 255            # block anchored at (0, 0)

        filled = fill_holes(mask)

        assert int((filled > 0).sum()) == int((mask > 0).sum()), (
            f"grew from {int((mask > 0).sum())} to {int((filled > 0).sum())} px "
            "— the flood never started")
        assert filled[H - 1, W - 1] == 0, "far corner must stay background"

    def test_fill_holes_matches_the_plain_python_spec_on_edge_cases(self):
        """post_processing/cpu seeds from every border pixel; utils must agree
        with it, including on the cases that used to diverge."""
        from utils.post_processing import fill_holes as fast
        from post_processing.cpu.post_processing_cpu import fill_holes as spec
        cases = {}
        m = np.zeros((16, 16), np.uint8); m[0:8, 0:8] = 255
        cases["corner is foreground"] = m
        cases["all foreground"] = np.full((16, 16), 255, np.uint8)
        cases["all background"] = np.zeros((16, 16), np.uint8)
        m = np.zeros((16, 16), np.uint8); m[2:14, 2:14] = 255; m[5:11, 5:11] = 0
        cases["ring with a hole"] = m
        for name, mask in cases.items():
            assert np.array_equal(fast(mask), spec(mask)), name

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

        On CDnet highway (frames 470-1700) the old chain scored F1 0.8748
        against 0.9344 for this one, because it traded precision away to close
        holes by brute force.
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

    def test_mask_refiner_treats_shadow_as_background(self, small_dims):
        """MOG2 writes 127 for shadow; every downstream step reads non-zero as
        foreground, so the shadow has to be dropped before any of them run."""
        from utils.post_processing import mask_refiner
        H, W = small_dims
        mask = np.zeros((H, W), np.uint8)
        mask[10:H - 10, 10:W // 2] = 255      # subject
        mask[10:H - 10, W // 2:W - 10] = 127  # its shadow, adjacent

        out = mask_refiner(mask)

        assert out[H // 2, W // 2 + 4] == 0, "shadow was kept as foreground"
        assert out[H // 2, W // 4] == 255, "subject was dropped"

    def test_blur_kernel_scales_with_resolution(self):
        """A fixed 15x15 is strong at 240p and nearly invisible at 1080p.

        Measured on LTSSUD-Test.mp4, residual sharpness after a fixed 15x15:
        0.7% at 240p against 8.1% at 1080p. Scaling by frame height keeps the
        effect constant.
        """
        from utils.post_processing import blur_ksize_for
        assert blur_ksize_for(np.zeros((240, 320, 3), np.uint8)) == 15
        assert blur_ksize_for(np.zeros((1080, 1920, 3), np.uint8)) == 68 | 1
        for h in (120, 240, 480, 720, 1080):
            k = blur_ksize_for(np.zeros((h, h * 2, 3), np.uint8))
            assert k % 2 == 1 and k >= 3, f"{h}p gave an unusable kernel {k}"

    def test_background_subtractor_blurs_harder_at_higher_resolution(self):
        from utils.post_processing import background_subtractor
        rng = np.random.default_rng(0)
        residual = {}
        for h, w in ((240, 320), (1080, 1440)):
            frame = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
            mask = np.zeros((h, w), np.uint8)          # blur the whole frame
            out = background_subtractor(frame, mask)
            sharp = lambda im: cv2.Laplacian(
                cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
            residual[h] = sharp(out) / sharp(frame)
        assert residual[1080] < residual[240] * 4, (
            f"1080p keeps {residual[1080]:.3%} sharp against {residual[240]:.3%} "
            "at 240p — the kernel is not scaling")


class TestPipelineGuards:

    def test_pipeline_rejects_a_differently_sized_frame(self):
        """Every buffer is allocated once against the first frame. A frame of
        another size used to pass straight through and return a mask of the
        original shape, silently — which is what a webcam renegotiating its
        resolution mid-stream produces."""
        import pytest
        from gmm import GMM_CPU_NUMBA
        from pipeline import make_pipeline
        rng = np.random.default_rng(0)
        first = rng.integers(0, 256, (32, 40, 3), dtype=np.uint8)
        p = make_pipeline(GMM_CPU_NUMBA, first, n_components=5)

        out, mask, _ = p.process(rng.integers(0, 256, (32, 40, 3), dtype=np.uint8))
        assert mask.shape == (32, 40)

        with pytest.raises(ValueError, match="64x48|48x64"):
            p.process(rng.integers(0, 256, (48, 64, 3), dtype=np.uint8))


class TestBgProbDecision:
    """`bg_prob` replacing MOG2's binary decision — worth +3.0 F1 on highway."""

    def test_bg_prob_threshold_is_stricter_than_the_binary_decision(self):
        """MOG2 calls a pixel background on *any* match inside the background
        set, however little weight that mode carries — `background` in the
        kernel is exactly `bg_prob > 0`. Thresholding at 0.5 demands the
        matched modes carry half the weight, so it can only ever mark more
        pixels foreground, never fewer."""
        from utils.post_processing import mask_refiner
        from settings import MOG2_BG_PROB_THRESHOLD
        rng = np.random.default_rng(0)
        H, W = 40, 50
        mask = np.where(rng.random((H, W)) < 0.3, np.uint8(255), np.uint8(0))
        # bg_prob must agree with the mask on which pixels matched nothing
        bg_prob = np.where(mask == 255, 0.0, rng.random((H, W))).astype(np.float32)

        binary = mask_refiner(mask)
        graded = mask_refiner(mask, bg_prob=bg_prob)
        assert (graded[binary == 255] == 255).all(), (
            "a pixel MOG2 called foreground must stay foreground")
        assert float(MOG2_BG_PROB_THRESHOLD) > 0.0

    def test_bg_prob_recovers_a_weakly_matched_pixel(self):
        """The whole point: a pixel that matched a background mode carrying 10%
        of the weight is background to MOG2 and foreground here."""
        from utils.post_processing import mask_refiner
        H, W = 40, 50
        mask = np.zeros((H, W), np.uint8)          # MOG2: all background
        bg_prob = np.ones((H, W), np.float32)
        bg_prob[10:30, 15:35] = 0.1                # ...but weakly so, in a block

        assert not (mask_refiner(mask) == 255).any()
        graded = mask_refiner(mask, bg_prob=bg_prob)
        assert (graded[15:25, 20:30] == 255).all()

    def test_close_bridges_a_gap_the_hole_fill_cannot(self):
        """fill_holes only fills *enclosed* background. An open notch in the
        silhouette — the gap MOG2 leaves along a low-contrast edge — needs the
        CLOSE."""
        from utils.post_processing import mask_refiner
        H, W = 60, 60
        mask = np.zeros((H, W), np.uint8)
        mask[15:45, 15:45] = 255
        mask[15:30, 28:32] = 0                     # a notch open to the top

        assert mask_refiner(mask)[20, 30] == 0, "not a hole; fill cannot see it"
        assert mask_refiner(mask, close_ksize=9)[20, 30] == 255

    def test_close_ksize_scales_with_the_frame_and_is_capped(self):
        from utils.post_processing import close_ksize_for
        from settings import CLOSE_KSIZE_MAX
        assert close_ksize_for(np.zeros((240, 320))) == 15
        assert close_ksize_for(np.zeros((1080, 1920))) <= CLOSE_KSIZE_MAX
        for h in (120, 240, 480, 720, 1080, 4320):
            k = close_ksize_for(np.zeros((h, h)))
            assert k % 2 == 1 and 3 <= k <= CLOSE_KSIZE_MAX

    def test_close_is_extensive_and_cannot_empty_a_mask(self):
        """The property that makes CLOSE safe where OPEN is not. `median+OPEN`
        produced 6 entirely empty masks on highway; a CLOSE never can."""
        from utils.post_processing import mask_refiner
        rng = np.random.default_rng(3)
        mask = np.zeros((60, 60), np.uint8)
        mask[28:32, 28:32] = 255                   # one small blob
        for k in (0, 5, 9, 15):
            assert (mask_refiner(mask, close_ksize=k) == 255).any(), (
                f"CLOSE {k} erased the only foreground")


class TestPipelineMorphology:

    def test_pipeline_morphology_is_a_close_not_an_open(self):
        """An erode-first pass deletes anything thinner than the kernel. A
        one-pixel-wide arm is exactly that, and the pipelines used to run
        erode first."""
        from utils import blur_numba
        blur_numba.warmup()
        mask = np.zeros((40, 40), np.uint8)
        mask[20, 5:35] = 255                       # a one-pixel-wide limb
        out = blur_numba.morph_close(mask, mask.copy(), mask.copy())
        assert (out == 255).sum() >= (mask == 255).sum(), (
            "morphology erased a thin structure — that is an OPEN")
