#pragma once

//#define _YOLO_LIBTORCH
//#define _YOLO_ONNXRUNTIME
//#define _YOLO_OPENCV
//#define _YOLO_OPENVINO
//#define _YOLO_TENSORRT

#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>

enum Backend_Type
{
	Libtorch,
	ONNXRuntime,
	OpenCV,
	OpenVINO,
	TensorRT,
};

enum Task_Type
{
	Classify,
	Detect,
	Segment,
};

enum Algo_Type
{
	YOLOv5,
	YOLOv8,
};

enum Device_Type
{
	CPU,
	GPU,
};

enum Model_Type
{
	FP32,
	FP16,
	INT8,
};

class YOLO
{
public:
	virtual ~YOLO() {};	//这句代码保证虚拟继承析构的时候不会内存泄漏

	virtual void init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path) = 0;

	void infer(const std::string file_path, char* argv[], bool save_result = true, bool show_result = true);

	virtual void release() {};

protected:
	virtual void pre_process() = 0;

	virtual void process() = 0;

	virtual void post_process() = 0;

	cv::Mat m_image;

	cv::Mat m_result;

	int m_input_width = 640;

	int m_input_height = 640;

	int m_input_numel = 1 * 3 * m_input_width * m_input_height;
};

class  CreateFactory
{
public:
	typedef std::unique_ptr<YOLO>(*CreateFunction)();

	static CreateFactory& instance();

	void register_class(const Backend_Type& backend_type, const Task_Type& task_type, CreateFunction create_function);

	std::unique_ptr<YOLO> create(const Backend_Type& backend_type, const Task_Type& task_type);

private:
	CreateFactory();

	//std::map<Backend_Type, CreateFunction> m_backend_registry;
	std::vector<std::vector<CreateFunction>> m_create_registry;
};

