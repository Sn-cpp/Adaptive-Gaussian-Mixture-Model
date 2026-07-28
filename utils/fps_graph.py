import numpy as np
import cv2

class FPS_Graph:
    def __init__(self, height: int, width: int, max_fps: int = 150):
        self.graph = np.full(shape=(height , width, 3), fill_value=255, dtype=np.uint8)
        self.MAX_FPS = max_fps
        self.height = height
        self.width = width

        self.buffer = [0, 0]
        self.cur_frame = 0

        self.overlay = np.full(shape=(height , width, 3), fill_value=0, dtype=np.uint8)
        # Draw reference lines
        for t in [0, 30, 60, 90, 120]:
            yy = int((1.0 - t / self.MAX_FPS) * (self.height - 1))
            cv2.line(self.overlay, (0, yy), (self.width - 1, yy), (0, 0, 255), 1)
            cv2.putText(self.overlay, f"{t}", (5, yy - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                        (0, 0, 255), 1)

        self.overlay_mask = (self.overlay[:, :, 2] > 0).astype(np.uint8) * 255

    def write_value(self, value):

        # Left-shift the scene
        self.graph[:, :-1] = self.graph[:, 1:]
        self.graph[:, -1] = 255

        # Map FPS value to y-coordinate
        y = int((1.0 - value / self.MAX_FPS) * (self.height - 1))

        # Store to buffer for line drawing
        self.buffer[0] = self.buffer[1]
        self.buffer[1] = y

        # Draw the line connecting two last FPS values
        cv2.line(self.graph, (self.width - 2, self.buffer[0]), (self.width - 1, self.buffer[1]), (0, 0, 0), 1)

    def display(self, window_name: str, fps_value):
        scene = self.graph.copy()
        cv2.copyTo(self.overlay, self.overlay_mask, scene)

        # Current value
        cv2.putText(scene,
                    f"{fps_value}",
                    (self.width - 50, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    1)

        cv2.imshow(window_name, scene)