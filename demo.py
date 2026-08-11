"""Live camera / video-file demo: MOG2 + dual-GMM + Push-Relabel GrabCut.

Mirrors cv2_grabcut2.py output: foreground composited over blurred background.

Usage
-----
  python demo.py                   # camera 0, full resolution
  python demo.py video.mp4         # video file
  python demo.py video.mp4 0.5     # video file at half resolution
  python demo.py 0 0.5             # camera 0 at half resolution

Window layout (after warmup):
  [Original + ROI] | [Our composite] | [Our mask] | [bg_prob heatmap]

Keys
----
  q  — quit
  r  — reset GMM state and restart warmup
  b  — toggle bg_prob panel
"""
import sys
import cv2
import numpy as np
from time import perf_counter

from gmm import GMM_CPU_NUMBA
from gmm.mog2_common import to_planar
from grabcut_numba import GrabCutPipeline
from settings import MOG2_N_COMPONENTS

WARMUP_FRAMES = 30
WINDOW_NAME   = "MOG2 + Push-Relabel GrabCut"
DEFAULT_SCALE = 1.0
BLUR_KSIZE    = 15


def _make_roi(H, W):
    rw = int(W * 0.6); rh = int(H * 0.7)
    rx = (W - rw) // 2; ry = (H - rh) // 2
    return (rx, ry, rw, rh)


def _reset_gmm(gmm):
    gmm.means[:]   = 0
    gmm.vars[:]    = 0
    gmm.weights[:] = 0
    gmm.modes[:]   = 0
    gmm.nframes    = 0


def run(source=0, scale=DEFAULT_SCALE):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source!r}")

    ret, first_raw = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError("Cannot read first frame")

    def resize(f):
        if scale == 1.0:
            return f
        fh, fw = f.shape[:2]
        return cv2.resize(f, (max(1, int(round(fw * scale))),
                              max(1, int(round(fh * scale)))),
                          interpolation=cv2.INTER_LINEAR)

    first_frame = resize(first_raw)
    H, W        = first_frame.shape[:2]
    roi         = _make_roi(H, W)
    rx, ry, rw, rh = roi

    gmm      = GMM_CPU_NUMBA(first_frame, MOG2_N_COMPONENTS)
    pipeline = GrabCutPipeline(gmm, roi, blur_ksize=BLUR_KSIZE)

    frame_count = 0
    fps         = 0.0
    t_prev      = perf_counter()
    show_bgprob = True

    is_video = isinstance(source, str)
    print(f"Source: {'video: ' + source if is_video else f'camera {source}'}  ({W}x{H}  scale={scale})")
    print(f"ROI: {roi}   Warmup: {WARMUP_FRAMES} frames")
    print("Keys: q=quit  r=reset GMM  b=toggle bg_prob panel")

    while True:
        ret, raw = cap.read()
        if not ret:
            print("End of video." if is_video else "Camera read failed.")
            break

        frame = resize(raw)
        frame_count += 1

        if frame_count <= WARMUP_FRAMES:
            gmm.step(to_planar(frame))
            display = frame.copy()
            cv2.rectangle(display, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)
            cv2.putText(display,
                        f"Warming up... {frame_count}/{WARMUP_FRAMES}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
            cv2.imshow(WINDOW_NAME, display)
        else:
            mog2_mask, bg_prob, final_mask, composite, elapsed = pipeline.rqstep(frame)

            now    = perf_counter()
            fps    = 1.0 / max(now - t_prev, 1e-6)
            t_prev = now

            p1 = frame.copy()
            cv2.rectangle(p1, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)
            cv2.putText(p1, f"FPS {fps:.1f}  MOG2 {elapsed*1000:.0f}ms",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(p1, "Original", (10, H - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

            p2 = composite.copy()
            cv2.putText(p2, "Composite (fg sharp / bg blur)", (10, H - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 255, 100), 1)

            p3 = cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGR)
            cv2.putText(p3, "Mask", (10, H - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 255, 100), 1)

            foreground = np.zeros_like(frame)
            cv2.copyTo(frame, p3, foreground)

            cv2.imshow("Original", p1)
            cv2.imshow("Final (Subtracted/Blurred)", p2)
            cv2.imshow("Mask", p3)
            cv2.imshow("Foreground", foreground)


            # panels = [p1, p2, p3]

            if show_bgprob:
                prob_u8 = (np.clip(bg_prob, 0.0, 1.0) * 255.0).astype(np.uint8)
                p4 = cv2.applyColorMap(prob_u8, cv2.COLORMAP_JET)
                cv2.putText(p4, "bg_prob", (10, H - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                cv2.imshow("BG/FG Probabilities", p4)

            # cv2.imshow(WINDOW_NAME, np.hstack(panels))

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            _reset_gmm(gmm)
            frame_count = 0
            print("GMM reset — restarting warmup.")
        elif key == ord('b'):
            show_bgprob = not show_bgprob
            print(f"bg_prob panel: {'ON' if show_bgprob else 'OFF'}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    source = sys.argv[1] if len(sys.argv) > 1 else 0
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    scale = float(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_SCALE
    run(source, scale)
