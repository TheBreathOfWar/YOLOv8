# Import necessary libraries
import argparse  # For parsing command-line arguments
from ultralytics import YOLO  # Import YOLO model from ultralytics
from PIL import ImageColor  # For converting color from HEX to RGB
from utils import *  # Import utility functions
from PIL import Image  # For image processing

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to the input image")
    parser.add_argument("--selection_color", required=True, help="Color in HEX format without # (FFFFFF)")
    parser.add_argument("--selection_type", required=True, help="1 for boxes or 0 for masks")
    args = parser.parse_args()

    # Extract arguments
    source = args.image
    selection_type = int(args.selection_type)
    color = args.selection_color

    # Load YOLO model for object detection
    model = YOLO('yolov8x-seg.pt')

    # Read the input image using OpenCV
    image = cv2.imread(source)
    h, w, _ = image.shape

    # Perform object detection using YOLO
    results = model(source=image.copy(), classes=0, retina_masks=True, conf=0.2, verbose=False)

    # Convert HEX color to BGR format
    color = list(ImageColor.getcolor('#' + color, "RGB"))[::-1]

    # Process the results
    for r in results:
        boxes = r.boxes
        masks = r.masks
        masks = masks.data.cpu()

        # Iterate over detected objects
        for mask, box in zip(masks.data.cpu().numpy(), boxes):
            if selection_type == 1:
                # Draw bounding boxes if selection_type is 1
                x_min = int(box.data[0][0])
                y_min = int(box.data[0][1])
                x_max = int(box.data[0][2])
                y_max = int(box.data[0][3])

                draw_box(image, [x_min, y_min, x_max, y_max], color)

            elif selection_type == 0:
                # Draw masks if selection_type is 0
                mask = cv2.resize(mask, (w, h))
                image = draw_mask(image, mask, color)
            else:
                print('Error: incorrect selection_type, expected 1 or 0')

    # Convert the processed image back to PIL format
    image = Image.fromarray(image[..., ::-1])

    # Save the result as an image
    image.save('output.jpg')
    print('Result saved to output.jpg')
