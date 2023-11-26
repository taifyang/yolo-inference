#include "yolov5_libtorch.h"
#include "yolov5_onnxruntime.h"
#include "yolov5_opencv.h"
#include "yolov5_openvino.h"
#include "yolov5_tensorrt.h"


int main(int argc, char* argv[])
{
	//YOLOv5* yolov5 = new YOLOv5_Libtorch("yolov5n_cpu.torchscript", CPU);
	//YOLOv5* yolov5 = new YOLOv5_Libtorch("yolov5n_gpu.torchscript", CUDA);

	//YOLOv5* yolov5 = new YOLOv5_ONNXRuntime("yolov5n.onnx", CPU);
	//YOLOv5* yolov5 = new YOLOv5_ONNXRuntime("yolov5n.onnx", CUDA);

	//YOLOv5* yolov5 = new YOLOv5_OpenCV("yolov5n.onnx", CPU);
	//YOLOv5* yolov5 = new YOLOv5_OpenCV("yolov5n.onnx", CUDA);

	//YOLOv5* yolov5 = new YOLOv5_OpenVINO("yolov5n_fp32.onnx", CPU);
	//YOLOv5* yolov5 = new YOLOv5_OpenVINO("yolov5n_fp32.onnx", GPU);
	//YOLOv5* yolov5 = new YOLOv5_OpenVINO("yolov5n_fp32.xml", CPU);
	//YOLOv5* yolov5 = new YOLOv5_OpenVINO("yolov5n_fp32.xml", GPU);
	//YOLOv5* yolov5 = new YOLOv5_OpenVINO("yolov5n_fp16.xml", CPU);
	//YOLOv5* yolov5 = new YOLOv5_OpenVINO("yolov5n_fp16.xml", GPU);
	//YOLOv5* yolov5 = new YOLOv5_OpenVINO("yolov5n_int8.xml", CPU);
	//YOLOv5* yolov5 = new YOLOv5_OpenVINO("yolov5n_int8.xml", GPU);

	YOLOv5* yolov5 = new YOLOv5_TensorRT("yolov5n.engine", GPU);

	yolov5->infer("bus.jpg");
	return 0;
}

