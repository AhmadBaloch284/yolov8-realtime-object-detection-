# display.py

import cv2
import time
import config

DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720


class FrameRenderer:

    def __init__(self):
        self._prev_time = time.time()  # for FPS calculation
        self.fps        = 0.0

    def _update_fps(self):
        """Compute smoothed FPS using exponential moving average."""
        now            = time.time()
        instant_fps    = 1.0 / max(now - self._prev_time, 1e-6)
        self.fps       = 0.9 * self.fps + 0.1 * instant_fps  # smoothing
        self._prev_time = now

    def draw(self, frame, detections):

        self._update_fps()

        label_pad_v = 10
        label_pad_h = 6

        for det in detections:
            x1, y1, x2, y2 = det["box"]
            cid = det["class_id"]
            color = config.BOX_COLORS[cid % len(config.BOX_COLORS)]
            label = f'{det["class_name"]} {det["confidence"]:.0%}'

            # ── Bounding box 
            cv2.rectangle(
                frame, (x1, y1), (x2, y2),
                color, config.BOX_THICKNESS
            )

            # ── Label background (solid filled rect) 
            (text_w, text_h), baseline = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                config.LABEL_FONT_SCALE,
                config.LABEL_THICKNESS,
            )
            label_h = text_h + baseline + label_pad_v * 2
            label_y1 = max(y1 - label_h - 4, 0)
            label_y2 = label_y1 + label_h

            cv2.rectangle(
                frame,
                (x1, label_y1),
                (x1 + text_w + label_pad_h, label_y2),
                color, cv2.FILLED
            )

            # ── Label text 
            brightness = 0.299 * color[2] + 0.587 * color[1] + 0.114 * color[0]
            text_color = (0, 0, 0) if brightness > 160 else (255, 255, 255)

            cv2.putText(
                frame, label,
                (x1 + 3, label_y1 + label_pad_v + text_h),
                cv2.FONT_HERSHEY_SIMPLEX,
                config.LABEL_FONT_SCALE,
                text_color,
                config.LABEL_THICKNESS,
                cv2.LINE_AA,
            )

        # ── HUD overlays 
        h, w = frame.shape[:2]

        if config.SHOW_FPS:
            cv2.putText(
                frame, f"FPS: {self.fps:.1f}",
                (10, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                (0, 255, 0), 3, cv2.LINE_AA,
            )

        if config.SHOW_COUNT:
            cv2.putText(
                frame, f"Objects: {len(detections)}",
                (10, 78),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                (0, 255, 255), 3, cv2.LINE_AA,
            )

        return frame

    def resize_for_display(self, frame):
        """Scale frame to the display window size (detection uses full resolution)."""
        return cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
