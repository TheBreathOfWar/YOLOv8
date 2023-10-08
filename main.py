import argparse
from ultralytics import YOLO
from utils import *
from PIL import Image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to the input image")
    parser.add_argument("--selection-color", required=True, help="Color of masks and boxes")
    parser.add_argument("--selection-type", required=True, help="1 for boxes or 0 for masks")
    args = parser.parse_args()

    source = args.image
    selection_type = int(args.selection_type)

    model = YOLO('yolov8x-seg.pt')
    image = cv2.imread(source)
    h, w, _ = image.shape
    results = model(source=image.copy(), classes=0, retina_masks=True, conf=0.2, verbose=False)

    for r in results:
        boxes = r.boxes
        masks = r.masks
        masks = masks.data.cpu()
        for mask, box in zip(masks.data.cpu().numpy(), boxes):
            if selection_type == 1:
                x_min = int(box.data[0][0])
                y_min = int(box.data[0][1])
                x_max = int(box.data[0][2])
                y_max = int(box.data[0][3])
                draw_box(image, [x_min, y_min, x_max, y_max])
            elif selection_type == 0:
                mask = cv2.resize(mask, (w, h))
                image = draw_mask(image, mask)
            else:
                print('Error: incorrect selection_type, expected 1 or 0')

    image = Image.fromarray(image[..., ::-1])
    image.save('output.jpg')
