import numpy as np

MAX_COMPONENTS = 20

# ------------------------------------------------------------------------------
# MOG2 model parameters (Zivkovic 2004).
# These mirror the OpenCV `BackgroundSubtractorMOG2` defaults exactly, so the
# models can be compared 1:1 against cv2.

MOG2_N_COMPONENTS = 5           # nmixtures
MOG2_HISTORY = 500              # learning rate falls back to 1/history

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

# Conservative (foreground-protected) update — ViBe's rule, Barnich & Van
# Droogenbroeck 2011: a pixel only feeds the background model if it was
# classified *background*. Zivkovic's MOG2 updates every pixel unconditionally,
# so a subject that holds still is absorbed into the background within a few
# seconds and stops being detected. That is the single biggest quality problem
# on webcam footage, and no amount of post-processing repairs it — the mask it
# would refine no longer contains the person.
#
# Off by default: with it on the models no longer reproduce cv2's
# BackgroundSubtractorMOG2, and that parity is what the correctness tests
# assert. `main.py` turns it on, because there the target is a person rather
# than a moving car. See `MOG2Base.__init__` for the exact semantics.
MOG2_CONSERVATIVE_UPDATE = False

FLT_EPSILON = np.float32(1.1920929e-07)

# Background blur
BLUR_KSIZE = 15
BLUR_SIGMA = 5.0
