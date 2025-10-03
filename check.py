import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

# Class names (edit according to your model)
class_names = ["cap", "nocap", "plastic-bottle"]

# -------------------------
# Helper functions
# -------------------------
def xywh2xyxy(xywh):
    x, y, w, h = xywh
    return [x - w/2, y - h/2, x + w/2, y + h/2]

def process_yolov8_output(output, conf_threshold=0.25, iou_threshold=0.45, img_size=640):
    preds = np.squeeze(output).T  # (8400, 7)
    boxes, scores, class_ids = [], [], []

    for pred in preds:
        x, y, w, h, conf, cls0, cls1, cls2 = pred  # 3 classes
        if conf < conf_threshold:
            continue

        class_scores = [cls0, cls1, cls2]
        class_id = int(np.argmax(class_scores))
        score = conf * class_scores[class_id]

        if score > conf_threshold:
            x1, y1, x2, y2 = xywh2xyxy([x, y, w, h])
            x1, y1, x2, y2 = int(x1 * img_size), int(y1 * img_size), int(x2 * img_size), int(y2 * img_size)

            boxes.append([x1, y1, x2 - x1, y2 - y1])
            scores.append(float(score))
            class_ids.append(class_id)

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

# -------------------------
# Load TFLite model
# -------------------------
interpreter = tflite.Interpreter(model_path="/home/pi/Desktop/test/yolo_model_20251003-184329.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
img_size = input_details[0]['shape'][1]  # 640

# -------------------------
# Start Pi Camera
# -------------------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess
    frame_resized = cv2.resize(frame, (img_size, img_size))
    input_data = np.expand_dims(frame_resized, axis=0).astype(np.float32) / 255.0

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    results = process_yolov8_output(output_data, conf_threshold=0.2)
