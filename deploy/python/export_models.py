from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("yolo11s-pose.pt")

# Export the model to ONNX format
model.export(format="mnn")  # creates 'yolo11s-pose.onnx'
# Load the exported ONNX model
onnx_model = YOLO("yolo11s-pose.mnn")
# Run inference
results = onnx_model("https://ultralytics.com/images/bus.jpg")

# Export the model to ONNX format
model.export(format="onnx")  # creates 'yolo11s-pose.onnx'
# Load the exported ONNX model
onnx_model = YOLO("yolo11s-pose.onnx")
# Run inference
results = onnx_model("https://ultralytics.com/images/bus.jpg")

# Export the model to NCNN format
model.export(format="ncnn")
# Load the exported NCNN model
ncnn_model = YOLO("./yolo11s-pose_ncnn_model")
# Run inference
results = ncnn_model("https://ultralytics.com/images/bus.jpg")

# Export the model to OpenVINO format
model.export(format="openvino")
# Load the exported OpenVINO model
ov_model = YOLO("yolo11s-pose_openvino_model/")
# Run inference
results = ov_model("https://ultralytics.com/images/bus.jpg")

# Export the model to TensorRT format
model.export(format="engine")
# Load the exported TensorRT model
tensorrt_model = YOLO("yolo11s-pose.engine")
# Run inference
results = tensorrt_model("https://ultralytics.com/images/bus.jpg")

# Export the model to TorchScript format
model.export(format="torchscript")  # creates 'yolo11n.torchscript'
# Load the exported TorchScript model
torchscript_model = YOLO("yolo11s-pose.torchscript")
# Run inference
results = torchscript_model("https://ultralytics.com/images/bus.jpg")