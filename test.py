import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
from process import process_yolov8_output  # custom function you wrote

# Load model
interpreter = tflite.Interpreter(model_path="/home/pi/Desktop/test/yolo_model_20251003-184329.tflite")
interpreter.allocate_tensors()

# Input / output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load and preprocess image
image = cv2.imread("/home/pi/Desktop/test/test_cap.JPG")
image_resized = cv2.resize(image, (640, 640))
input_data = np.expand_dims(image_resized, axis=0).astype(np.float32) / 255.0

# Run inference
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])

# Post-process and print results
results = process_yolov8_output(output_data, conf_threshold=0.25)

for r in results:
    print(f"Detected {r['class']} ({r['confidence']:.2f}) at {r['box']}")
