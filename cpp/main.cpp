#include "yolov5.h"


int main(int argc, char* argv[])
{
	std::cout << "yolov5n_cpu_fp32.torchscript" << std::endl;
	{
		std::unique_ptr<YOLOv5> yolov5 = AlgoFactory::instance().create(Algo_Type::Libtorch);
		yolov5->init("yolov5n_cpu_fp32.torchscript", CPU, FP32);
		yolov5->infer("test.mp4");
		yolov5->release();
	}

	std::cout << "yolov5n_gpu_fp32.torchscript" << std::endl;
	{
		std::unique_ptr<YOLOv5> yolov5 = AlgoFactory::instance().create(Algo_Type::Libtorch);
		yolov5->init("yolov5n_gpu_fp32.torchscript", GPU, FP32);
		yolov5->infer("test.mp4");
		yolov5->release();
	}

	std::cout << "yolov5n_gpu_fp16.torchscript" << std::endl;
	{
	 	std::unique_ptr<YOLOv5> yolov5 = AlgoFactory::instance().create(Algo_Type::Libtorch);
	 	yolov5->init("yolov5n_gpu_fp16.torchscript", GPU, FP16);
	 	yolov5->infer("test.mp4");
	 	yolov5->release();
	}

	std::cout << "yolov5n_fp32.onnx cpu" << std::endl;
	 {
	 	std::unique_ptr<YOLOv5> yolov5 = AlgoFactory::instance().create(Algo_Type::ONNXRuntime);
	 	yolov5->init("yolov5n_fp32.onnx", CPU, FP32);
	 	yolov5->infer("test.mp4");
	 	yolov5->release();
	 }

	 std::cout << "yolov5n_fp32.onnx gpu" << std::endl;
	 {
	 	std::unique_ptr<YOLOv5> yolov5 = AlgoFactory::instance().create(Algo_Type::ONNXRuntime);
	 	yolov5->init("yolov5n_fp32.onnx", GPU, FP32);
	 	yolov5->infer("test.mp4");
	 	yolov5->release();
	 }

	 std::cout << "yolov5n_fp16.onnx cpu" << std::endl;
	 {
	 	std::unique_ptr<YOLOv5> yolov5 = AlgoFactory::instance().create(Algo_Type::ONNXRuntime);
	 	yolov5->init("yolov5n_fp16.onnx", CPU, FP16);
	 	yolov5->infer("test.mp4");
	 	yolov5->release();
	 }

	 std::cout << "yolov5n_fp16.onnx gpu" << std::endl;
	 {
	 	std::unique_ptr<YOLOv5> yolov5 = AlgoFactory::instance().create(Algo_Type::ONNXRuntime);
	 	yolov5->init("yolov5n_fp16.onnx", GPU, FP16);
	 	yolov5->infer("test.mp4");
	 	yolov5->release();
	 }

	 std::cout << "yolov5n_int8.onnx cpu" << std::endl;
	 {
	 	std::unique_ptr<YOLOv5> yolov5 = AlgoFactory::instance().create(Algo_Type::ONNXRuntime);
	 	yolov5->init("yolov5n_int8.onnx", CPU, INT8);
	 	yolov5->infer("test.mp4");
	 	yolov5->release();
	 }

	 std::cout << "yolov5n_int8.onnx gpu" << std::endl;
	 {
	 	std::unique_ptr<YOLOv5> yolov5 = AlgoFactory::instance().create(Algo_Type::ONNXRuntime);
	 	yolov5->init("yolov5n_int8.onnx", GPU, INT8);
	 	yolov5->infer("test.mp4");
	 	yolov5->release();
	 }

	 std::cout << "yolov5n_fp32.onnx cpu" << std::endl;
	 {
	 	std::unique_ptr<YOLOv5> yolov5 = AlgoFactory::instance().create(Algo_Type::OpenCV);
	 	yolov5->init("yolov5n_fp32.onnx", CPU, FP32);
	 	yolov5->infer("test.mp4");
	 	yolov5->release();
	 }

	 std::cout << "yolov5n_fp32.onnx gpu" << std::endl;
	 {
	 	std::unique_ptr<YOLOv5> yolov5 = AlgoFactory::instance().create(Algo_Type::OpenCV);
	 	yolov5->init("yolov5n_fp32.onnx", GPU, FP32);
	 	yolov5->infer("test.mp4");
	 	yolov5->release();
	 }

	 std::cout << "yolov5n_fp32.xml cpu" << std::endl;
	 {
	 	std::unique_ptr<YOLOv5> yolov5 = AlgoFactory::instance().create(Algo_Type::OpenVINO);
	 	yolov5->init("yolov5n_fp32.xml", CPU, FP32);
	 	yolov5->infer("test.mp4");
	 	yolov5->release();
	 }

	 std::cout << "yolov5n_fp32.xml gpu" << std::endl;

	 {
	 	std::unique_ptr<YOLOv5> yolov5 = AlgoFactory::instance().create(Algo_Type::OpenVINO);
	 	yolov5->init("yolov5n_fp32.xml", GPU, FP32);
	 	yolov5->infer("test.mp4");
	 	yolov5->release();
	 }

	 std::cout << "yolov5n_fp16.xml cpu" << std::endl;
	 {
	 	std::unique_ptr<YOLOv5> yolov5 = AlgoFactory::instance().create(Algo_Type::OpenVINO);
	 	yolov5->init("yolov5n_fp16.xml", CPU, FP16);
	 	yolov5->infer("test.mp4");
	 	yolov5->release();
	 }

	 std::cout << "yolov5n_fp16.xml gpu" << std::endl;
	 {
	 	std::unique_ptr<YOLOv5> yolov5 = AlgoFactory::instance().create(Algo_Type::OpenVINO);
	 	yolov5->init("yolov5n_fp16.xml", GPU, FP16);
	 	yolov5->infer("test.mp4");
	 	yolov5->release();
	 }

	 std::cout << "yolov5n_int8.xml cpu" << std::endl;
	 {
	 	std::unique_ptr<YOLOv5> yolov5 = AlgoFactory::instance().create(Algo_Type::OpenVINO);
	 	yolov5->init("yolov5n_int8.xml", CPU, INT8);
	 	yolov5->infer("test.mp4");
	 	yolov5->release();
	 }

	 std::cout << "yolov5n_int8.xml gpu" << std::endl;
	 {
	 	std::unique_ptr<YOLOv5> yolov5 = AlgoFactory::instance().create(Algo_Type::OpenVINO);
	 	yolov5->init("yolov5n_int8.xml", GPU, INT8);
	 	yolov5->infer("test.mp4");
	 	yolov5->release();
	 }

	 std::cout << "yolov5n_fp32.engine" << std::endl;
	 {
	 	std::unique_ptr<YOLOv5> yolov5 = AlgoFactory::instance().create(Algo_Type::TensorRT);
	 	yolov5->init("yolov5n_fp32.engine", GPU, FP32);
	 	yolov5->infer("test.mp4");
	 	yolov5->release();
	 }

	 std::cout << "yolov5n_fp16.engine" << std::endl;
	 {
	 	std::unique_ptr<YOLOv5> yolov5 = AlgoFactory::instance().create(Algo_Type::TensorRT);
	 	yolov5->init("yolov5n_fp16.engine", GPU, FP16);
	 	yolov5->infer("test.mp4");
	 	yolov5->release();
	 }

	 std::cout << "yolov5n_int8.engine" << std::endl;
	 {
	 	std::unique_ptr<YOLOv5> yolov5 = AlgoFactory::instance().create(Algo_Type::TensorRT);
	 	yolov5->init("yolov5n_int8.engine", GPU, INT8);
	 	yolov5->infer("test.mp4");
	 	yolov5->release();
	 }

	return 0;
}

