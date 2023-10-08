import cv2

def draw_box(image, box, color=[56, 56, 255], label='person'):
    c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(image, c1, c2, color, thickness=3, lineType=cv2.LINE_AA)

    t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
    c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
    cv2.rectangle(image, c1, c2, color, -1, cv2.LINE_AA)
    cv2.putText(image, label, (c1[0], c1[1] - 2), 0, 1, [255, 255, 255], thickness=2, lineType=cv2.LINE_AA)