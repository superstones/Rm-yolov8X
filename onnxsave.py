from ultralytics import YOLO

# Load a model
model = YOLO('runs/detect/train84/weights/best.pt')  # load an official model

# Export the model
onnx_file = model.export(format='onnx', dynamic=True)
