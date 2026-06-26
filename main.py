# main.py

import cv2
from pathlib import Path
import config
from detector     import ObjectDetector
from video_source import VideoSource
from display      import FrameRenderer

_PROJECT_DIR = Path(__file__).resolve().parent


def _resolve_video_path(filename):
    """Return absolute path if the video file exists, else None."""
    path = Path(filename)
    if path.is_file():
        return str(path.resolve())
    project_path = _PROJECT_DIR / filename
    if project_path.is_file():
        return str(project_path)
    return None


def select_input_source():
    """Prompt user for webcam or video file; override config.SOURCE at runtime."""
    while True:
        print("\n    Select Input Source:")
        print("    [1] Webcam")
        print("    [2] Video File")
        choice = input("\nEnter choice (1 or 2): ").strip()

        if choice == "1":
            config.SOURCE = 0
            return
        if choice == "2":
            while True:
                filename = input(
                    "Enter video filename (e.g. Test_Video.mp4): "
                ).strip()
                path = _resolve_video_path(filename)
                if path:
                    config.SOURCE = path
                    return
                print(f"\nError: File not found: '{filename}'")
                print("Please check the name and try again.\n")
        print("Invalid choice. Please enter 1 or 2.\n")


def get_video_writer(source, frame):
    """Create a cv2.VideoWriter to save annotated output."""
    h, w = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    fps    = source.get_fps()
    writer = cv2.VideoWriter(config.OUTPUT_PATH, fourcc, fps, (w, h))
    print(f"[Main] Saving output to '{config.OUTPUT_PATH}' at {fps:.1f} fps")
    return writer


def main():
    select_input_source()

    print("=" * 50)
    print("  YOLOv8 Real-Time Object Detection")
    print("  Press Q or ESC to quit | S to toggle save")
    print("=" * 50)

    # ── Initialise modules 
    detector = ObjectDetector()    # loads YOLO model once
    renderer = FrameRenderer()     # handles all drawing
    writer   = None
    saving   = config.SAVE_OUTPUT

    # ── Main loop 
    with VideoSource() as source:
        while source.is_open():
            # 1. Grab frame from camera / file
            frame = source.read()
            if frame is None:
                print("[Main] End of source — exiting.")
                break

            # 2. Run YOLO inference on the frame
            detections = detector.detect(frame)

            # 3. Draw boxes + labels + HUD onto the frame
            annotated = renderer.draw(frame, detections)

            # 4. Optionally save to file
            if saving:
                if writer is None:
                    writer = get_video_writer(source, annotated)
                writer.write(annotated)

            # 5. Display in window (resize for screen; save/detection stay full size)
            cv2.imshow(config.WINDOW_TITLE, renderer.resize_for_display(annotated))

            # 6. Required every frame so the window updates and keys are read
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):   # Q or ESC
                print("[Main] Quit requested.")
                break
            elif key == ord("s"):       # S = toggle save
                saving = not saving
                state  = "ON" if saving else "OFF"
                print(f"[Main] Save output: {state}")
                if not saving and writer:
                    writer.release()
                    writer = None

    # ── Cleanup 
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("[Main] Done.")


if __name__ == "__main__":
    main()
