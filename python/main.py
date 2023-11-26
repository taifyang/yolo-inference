from yolov5_onnxruntime import *
from yolov5_opencv import *
from yolov5_openvino import *
from yolov5_tensorrt import *


# yolov5 = YOLOv5_ONNXRuntime(model_path="yolov5n.onnx", device_type=Device_Type.CPU)
# yolov5 = YOLOv5_ONNXRuntime(model_path="yolov5n.onnx", device_type=Device_Type.GPU)
# yolov5 = YOLOv5_OpenCV(model_path="yolov5n.onnx", device_type=Device_Type.CPU)
# yolov5 = YOLOv5_OpenCV(model_path="yolov5n.onnx", device_type=Device_Type.GPU)
# yolov5 = YOLOv5_OpenVINO(model_path="yolov5n_fp32.onnx", device_type=Device_Type.CPU)
# yolov5 = YOLOv5_OpenVINO(model_path="yolov5n_fp32.onnx", device_type=Device_Type.GPU)
# yolov5 = YOLOv5_OpenVINO(model_path="yolov5n_fp16.onnx", device_type=Device_Type.CPU)
# yolov5 = YOLOv5_OpenVINO(model_path="yolov5n_fp16.onnx", device_type=Device_Type.GPU)
# yolov5 = YOLOv5_OpenVINO(model_path="yolov5n_fp32.xml", device_type=Device_Type.CPU)
# yolov5 = YOLOv5_OpenVINO(model_path="yolov5n_fp32.xml", device_type=Device_Type.GPU)
# yolov5 = YOLOv5_OpenVINO(model_path="yolov5n_fp16.xml", device_type=Device_Type.CPU)
# yolov5 = YOLOv5_OpenVINO(model_path="yolov5n_fp16.xml", device_type=Device_Type.GPU)
# yolov5 = YOLOv5_OpenVINO(model_path="yolov5n_int8.xml", device_type=Device_Type.CPU)
yolov5 = YOLOv5_OpenVINO(model_path="yolov5n_int8.xml", device_type=Device_Type.GPU)
#yolov5 = YOLOv5_TensorRT(model_path="yolov5n.engine", device_type=Device_Type.GPU)
yolov5.infer("bus.jpg")