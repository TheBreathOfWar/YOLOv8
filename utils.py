import cv2
import numpy as np


def draw_box(image, box, color=[56, 56, 255], label='person'):
    c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(image, c1, c2, color, thickness=3, lineType=cv2.LINE_AA)

    t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
    c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
    cv2.rectangle(image, c1, c2, color, -1, cv2.LINE_AA)
    cv2.putText(image, label, (c1[0], c1[1] - 2), 0, 1, [255, 255, 255], thickness=2, lineType=cv2.LINE_AA)


def draw_mask(image, mask, color=[56, 56, 255], alpha=0.4):
    colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
    colored_mask = np.moveaxis(colored_mask, 0, -1)
    masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
    image_overlay = masked.filled()
    image_combined = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)

    return image_combined
