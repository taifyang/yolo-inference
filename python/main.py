from yolov5_onnxruntime import *
from yolov5_opencv import *
from yolov5_openvino import *
from yolov5_tensorrt import *


# yolov5 = YOLOv5_ONNXRuntime(model_path="yolov5n_fp32.onnx", device_type=Device_Type.CPU, model_type=Model_Type.FP32)  #38-57ms
# yolov5 = YOLOv5_ONNXRuntime(model_path="yolov5n_fp32.onnx", device_type=Device_Type.GPU, model_type=Model_Type.FP32)  #23-32ms
# yolov5 = YOLOv5_ONNXRuntime(model_path="yolov5n_fp16.onnx", device_type=Device_Type.CPU, model_type=Model_Type.FP16)  #49-62ms
# yolov5 = YOLOv5_ONNXRuntime(model_path="yolov5n_fp16.onnx", device_type=Device_Type.GPU, model_type=Model_Type.FP16)  #23-35ms
# yolov5 = YOLOv5_ONNXRuntime(model_path="yolov5n_int8.onnx", device_type=Device_Type.CPU, model_type=Model_Type.INT8)  #50-71ms
yolov5 = YOLOv5_ONNXRuntime(model_path="yolov5n_int8.onnx", device_type=Device_Type.GPU, model_type=Model_Type.INT8)    #60-86ms
 
# yolov5 = YOLOv5_OpenCV(model_path="yolov5n_fp32.onnx", device_type=Device_Type.CPU, model_type=Model_Type.FP32)       #105-123ms
# yolov5 = YOLOv5_OpenCV(model_path="yolov5n_fp32.onnx", device_type=Device_Type.GPU, model_type=Model_Type.FP32)

# yolov5 = YOLOv5_OpenVINO(model_path="yolov5n_fp32.xml", device_type=Device_Type.CPU, model_type=Model_Type.FP32)      #24-36ms    
# yolov5 = YOLOv5_OpenVINO(model_path="yolov5n_fp32.xml", device_type=Device_Type.GPU, model_type=Model_Type.FP32)      #86-112ms
# yolov5 = YOLOv5_OpenVINO(model_path="yolov5n_fp16.xml", device_type=Device_Type.CPU, model_type=Model_Type.FP16)      #24-38ms
# yolov5 = YOLOv5_OpenVINO(model_path="yolov5n_fp16.xml", device_type=Device_Type.GPU, model_type=Model_Type.FP16)      #68-105ms
# yolov5 = YOLOv5_OpenVINO(model_path="yolov5n_int8.xml", device_type=Device_Type.CPU, model_type=Model_Type.INT8)      #19-29ms
# yolov5 = YOLOv5_OpenVINO(model_path="yolov5n_int8.xml", device_type=Device_Type.GPU, model_type=Model_Type.INT8)      #67-115ms

# yolov5 = YOLOv5_TensorRT(model_path="yolov5n_fp32.engine", device_type=Device_Type.GPU, model_type=Model_Type.FP32)   #20-30ms
# yolov5 = YOLOv5_TensorRT(model_path="yolov5n_fp16.engine", device_type=Device_Type.GPU, model_type=Model_Type.FP16)   #19-26ms
# yolov5 = YOLOv5_TensorRT(model_path="yolov5n_int8.engine", device_type=Device_Type.GPU, model_type=Model_Type.INT8)   #19-23ms

yolov5.infer("test.mp4")