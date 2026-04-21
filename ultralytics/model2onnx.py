from ultralytics import YOLO

# Load the YOLO26 model
model = YOLO(r"ultralytics\best.pt")

# Export the model to ONNX format
model.export(format="onnx")  # creates 'yolo26n.onnx'
