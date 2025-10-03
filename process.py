import numpy as np
import cv2

# Class names (update according to your dataset)
class_names = ["cap","nocap","plastic-bottle"]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def xywh2xyxy(xywh):
    # [x, y, w, h] -> [x1, y1, x2, y2]
    x, y, w, h = xywh
    return [x - w/2, y - h/2, x + w/2, y + h/2]

def process_yolov8_output(output, conf_threshold=0.25, iou_threshold=0.45, img_size=640):
    preds = np.squeeze(output).T  # (8400, 7)
    boxes, scores, class_ids = [], [], []

    for pred in preds:
        x, y, w, h, conf, cls0, cls1 = pred

        if conf < conf_threshold:
            continue

        class_scores = [cls0, cls1]
        class_id = int(np.argmax(class_scores))
        score = conf * class_scores[class_id]

        if score > conf_threshold:
            # Scale back to image size
            x1, y1, x2, y2 = xywh2xyxy([x, y, w, h])
            x1, y1, x2, y2 = int(x1 * img_size), int(y1 * img_size), int(x2 * img_size), int(y2 * img_size)

            boxes.append([x1, y1, x2 - x1, y2 - y1])
            scores.append(float(score))
            class_ids.append(class_id)

    # Non-Max Suppression
    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, iou_threshold)

    results = []
    if len(indices) > 0:
        for i in indices.flatten():
            results.append({
                "box": boxes[i],
                "confidence": scores[i],
                "class": class_names[class_ids[i]]
            })
    return results
