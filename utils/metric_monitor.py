import numpy as np
import cv2

class Monitor:
    def __init__(self, height: int, width: int, max_value, references, window_name: str, description: str="", org: tuple=None):
        self.height = height
        self.width = width
        self.max_value = max_value
        self.window_name = window_name

        self.graph = np.full(shape=(height , width, 3), fill_value=255, dtype=np.uint8)

        self.overlay = np.full(shape=(height , width, 3), fill_value=255, dtype=np.uint8)
        # Draw reference lines
        for t in references:
            yy = int((1.0 - t / max_value) * (self.height - 1))
            cv2.line(self.overlay, (0, yy), (self.width - 1, yy), (0, 0, 0), 1)
            cv2.putText(self.overlay, f"{t}", (5, yy - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                        (0, 0, 0), 1)

        # Draw description if any
        if description:
            if org is None:
                raise ValueError("Description text requires origin")
            
            cv2.putText(self.overlay, description, org, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
            
        self.overlay_mask = (self.overlay[:, :, 2] <= 0).astype(np.uint8) * 255

    def write_value(self, value, last_value, color: tuple):
        # Left-shift the scene
        self.graph[:, :-1] = self.graph[:, 1:]
        self.graph[:, -1] = 255

        # Map FPS value to y-coordinate
        y0 = int((1.0 - min(last_value, self.max_value) / self.max_value) * (self.height - 1))
        y1 = int((1.0 - min(value, self.max_value) / self.max_value ) * (self.height - 1))


        # Draw the line connecting two last values
        cv2.line(self.graph, (self.width - 2, y0), (self.width - 1, y1), color, 1)

        return value

    def display(self, dynamic_desc: str="", org: tuple=None):
        scene = self.graph.copy()

        cv2.copyTo(self.overlay, self.overlay_mask, scene)

        if dynamic_desc:
            if org is None:
                raise ValueError("Dynamic description requires origin")
            
            cv2.putText(scene, dynamic_desc, org, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
        
        cv2.imshow(self.window_name, scene)

