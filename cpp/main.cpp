#include "yolov5_libtorch.h"
#include "yolov5_onnxruntime.h"
#include "yolov5_opencv.h"
#include "yolov5_openvino.h"
#include "yolov5_tensorrt.h"


int main(int argc, char* argv[])
{
	//YOLOv5* yolov5 = new YOLOv5_Libtorch("yolov5n_cpu_fp32.torchscript", CPU, FP32);	//65-90ms
	//YOLOv5* yolov5 = new YOLOv5_Libtorch("yolov5n_gpu_fp32.torchscript", GPU, FP32);	//12-19ms
	//YOLOv5* yolov5 = new YOLOv5_Libtorch("yolov5n_gpu_fp16.torchscript", GPU, FP16);	//10-16ms

	//YOLOv5* yolov5 = new YOLOv5_ONNXRuntime("yolov5n_fp32.onnx", CPU, FP32);	//21-25ms
	//YOLOv5* yolov5 = new YOLOv5_ONNXRuntime("yolov5n_fp32.onnx", GPU, FP32);	//13-16ms
	//YOLOv5* yolov5 = new YOLOv5_ONNXRuntime("yolov5n_fp16.onnx", CPU, FP16);	//37-46ms
	//YOLOv5* yolov5 = new YOLOv5_ONNXRuntime("yolov5n_fp16.onnx", GPU, FP16);	//17-20ms

	//YOLOv5* yolov5 = new YOLOv5_OpenCV("yolov5n_fp32.onnx", CPU, FP32);	//51-55ms
	//YOLOv5* yolov5 = new YOLOv5_OpenCV("yolov5n_fp32.onnx", GPU, FP32);	//11-13ms

	//YOLOv5* yolov5 = new YOLOv5_OpenVINO("yolov5n_fp32.xml", CPU, FP32);	//17-22ms
	//YOLOv5* yolov5 = new YOLOv5_OpenVINO("yolov5n_fp32.xml", GPU, FP32);	//75-91ms
	//YOLOv5* yolov5 = new YOLOv5_OpenVINO("yolov5n_fp16.xml", CPU, FP16);	//15-23ms
	//YOLOv5* yolov5 = new YOLOv5_OpenVINO("yolov5n_fp16.xml", GPU, FP16);	//60-75ms
	//YOLOv5* yolov5 = new YOLOv5_OpenVINO("yolov5n_int8.xml", CPU, INT8);	//12-18ms
	//YOLOv5* yolov5 = new YOLOv5_OpenVINO("yolov5n_int8.xml", GPU, INT8);	//47-61ms

	//YOLOv5* yolov5 = new YOLOv5_TensorRT("yolov5n_fp32.engine", GPU, FP32);	//7-11ms
	//YOLOv5* yolov5 = new YOLOv5_TensorRT("yolov5n_fp16.engine", GPU, FP16);	//10-12ms
	YOLOv5* yolov5 = new YOLOv5_TensorRT("yolov5n_int8.engine", GPU, INT8);		//6-7ms

	yolov5->infer("test.mp4");
	return 0;
}

