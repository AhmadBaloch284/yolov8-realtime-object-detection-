from pathlib import Path

_PROJECT_DIR = Path(__file__).resolve().parent

# Model
MODEL_NAME = "yolov8n.pt"       
CONFIDENCE_THRESHOLD = 0.35     
IOU_THRESHOLD = 0.5             

#Input source 
# SOURCE = 0 for webcam, or a video file in the project folder
TEST_VIDEO = "Test_Video.mp4"
SOURCE = str(_PROJECT_DIR / TEST_VIDEO)                            

#  Display
WINDOW_TITLE = "YOLOv8 Object Detection"
SHOW_FPS     = True            
SHOW_COUNT   = True             

# Box colors 
BOX_COLORS = [
    (0,   255, 0),     # green
    (0,   0,   255),   # red
    (255, 0,   0),     # blue
    (0,   255, 255),   # yellow
    (255, 0,   255),   # magenta
    (0,   165, 255),   # orange
    (255, 255, 0),     # cyan
    (128, 0,   255),   # purple
    (0,   128, 255),   # amber
    (255, 128, 0),     # sky blue
]

BOX_THICKNESS    = 3
LABEL_FONT_SCALE = 0.8
LABEL_THICKNESS  = 2

#  Output 
SAVE_OUTPUT  = False            
OUTPUT_PATH  = "output.avi"
