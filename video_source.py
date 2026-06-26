# video_source.py

import cv2
import config


class VideoSource:

    def __init__(self):
        self.source = config.SOURCE
        self.cap    = None

    def __enter__(self):
        self.cap = cv2.VideoCapture(self.source)

        if not self.cap.isOpened():
            raise IOError(
                f"[VideoSource] Cannot open source: '{self.source}'\n"
                f"  → If using webcam, check it's not in use by another app.\n"
                f"  → If using a file, check the path is correct."
            )

        # Read one frame per call; avoid backlog that skips frames on webcam
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        src_type = "webcam" if isinstance(self.source, int) else f"file '{self.source}'"
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS) or "N/A"
        print(f"[VideoSource] Opened {src_type} — {w}×{h} @ {fps:.1f} fps")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cap:
            self.cap.release()
            print("[VideoSource] Source released.")

    def is_open(self):
        return self.cap is not None and self.cap.isOpened()

    def read(self):
        """
        Read the next frame in order (no skipping).

        Returns:
            numpy.ndarray (BGR) if a frame is available, else None.
        """
        if self.cap is None:
            return None
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return None
        return frame

    def get_fps(self):
        """Native FPS of the source (used when saving output video)."""
        return self.cap.get(cv2.CAP_PROP_FPS) or 30.0
