import argparse
from ultralytics import YOLO
import cv2

if __name__ == "__main__":
    pass
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to the input image")
    parser.add_argument("--selection-color", required=True, help="Color of masks and boxes")
    parser.add_argument("--selection-type", required=True, help="1 for boxes or 0 for masks")
    args = parser.parse_args()

    source = args.image
    selection_type = int(args.selection_type)

    model = YOLO('yolov8x-seg.pt')
    image = cv2.imread(source)