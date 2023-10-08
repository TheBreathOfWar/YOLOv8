import argparse
from ultralytics import YOLO
import cv2
from PIL import ImageColor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to the input image")
    parser.add_argument("--selection_color", required=True, help="Color in HEX format without # (FFFFFF)")
    parser.add_argument("--selection_type", required=True, help="1 for boxes or 0 for masks")
    args = parser.parse_args()

    source = args.image
    selection_type = int(args.selection_type)
    color = args.selection_color

    model = YOLO('yolov8x-seg.pt')
    image = cv2.imread(source)
    h, w, _ = image.shape
    results = model(source=image.copy(), classes=0, retina_masks=True, conf=0.2, verbose=False)

    color = list(ImageColor.getcolor('#'+color, "RGB"))[::-1]
