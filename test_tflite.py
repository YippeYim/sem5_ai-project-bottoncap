import tflite_runtime.interpreter as tflite
import numpy as np

# model download
interpreter = tflite.Interpreter(model_path="yolo_model_20251003-184329.tflite")
interpreter.allocate_tensors()

# see input/output data
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Model inputs:", input_details)
print("Model outputs:", output_details)

# random data (dummy data) for test
input_shape = input_details[0]['shape']
dummy_input = np.random.rand(*input_shape).astype(np.float32)

# model run
interpreter.set_tensor(input_details[0]['index'], dummy_input)
interpreter.invoke()

output_data = interpreter.get_tensor(output_details[0]['index'])
print("Inference OK. Output shape:", output_data.shape)