import numpy as np


# ------------------------------------------------------------------------------
# MOG2 model parameters (Zivkovic 2004).
# These mirror the OpenCV `BackgroundSubtractorMOG2` defaults exactly, so the
# models can be compared 1:1 against cv2.

MOG2_COLOR = True
MOG2_N_COMPONENTS = 5           # nmixtures
MOG2_HISTORY = 250              # learning rate falls back to 1/history

MOG2_VAR_THRESHOLD = np.float32(16.0)       # Tb — background / foreground decision
MOG2_VAR_THRESHOLD_GEN = np.float32(9.0)    # Tg — "does the pixel fit this mode" (Tb > Tg)
MOG2_BACKGROUND_RATIO = np.float32(0.9)     # TB — cumulative weight forming the background

MOG2_VAR_INIT = np.float32(15.0)
MOG2_VAR_MIN = np.float32(4.0)
MOG2_VAR_MAX = np.float32(75.0)

MOG2_CT = np.float32(0.05)                  # complexity reduction: prune = -alpha * CT

MOG2_SHADOW_TAU = np.float32(0.5)
MOG2_SHADOW_VALUE = np.uint8(127)
MOG2_DETECT_SHADOWS = False

FLT_EPSILON = np.float32(1.1920929e-07)

# ------------------------------------------------------------------------------
# Post-processing
#
# A pixel counts as background only if the background modes it matched carry at
# least this much of the weight. MOG2's own rule is the degenerate case of this
# one, `bg_prob > 0` — any match at all, however weak the mode. Requiring real
# weight rejects matches against spurious low-weight modes.
MOG2_BG_PROB_THRESHOLD = np.float32(0.5)

# Morphological CLOSE applied to the binary mask before the hole fill, as a
# fraction of frame height. Capped: a kernel wider than the gap between two
# objects merges them into one blob, and F1 will not tell you.
CLOSE_KSIZE_FRACTION = 0.0625
CLOSE_KSIZE_MAX = 61

# Background blur
BLUR_KSIZE = 15
BLUR_SIGMA = 5.0
