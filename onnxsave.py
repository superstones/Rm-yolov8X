from ultralytics import YOLO

# Load a model
model = YOLO('Rm-yolov8X.pt')  # load an official model

# Export the model
onnx_file = model.export(format='onnx', dynamic=True)
