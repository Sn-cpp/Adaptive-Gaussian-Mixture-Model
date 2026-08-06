import numpy as np

INIT_VAR = np.float32(15.0)
REINIT_WEIGHT = np.float32(0.01)
MAX_COMPONENTS = 20

# Legacy Stauffer-Grimson knob, kept because debug.py still passes it and every
# model's step() accepts-and-ignores it.
COMP_GEN_THRESHOLD = np.float32(9.0)

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
MOG2_DETECT_SHADOWS = True

FLT_EPSILON = np.float32(1.1920929e-07)

# Background blur
BLUR_KSIZE = 15
BLUR_SIGMA = 5.0
