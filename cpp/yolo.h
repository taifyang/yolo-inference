#pragma once


#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>


const std::vector<std::string> class_names = {
	"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
	"fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
	"elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
	"skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
	"tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
	"sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
	"potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
	"microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
	"hair drier", "toothbrush" };			

const int input_width = 640;

const int input_height = 640;

const float score_threshold = 0.2;

const float nms_threshold = 0.5;

const float confidence_threshold = 0.2;

const int input_numel = 1 * 3 * input_width * input_height;

const int num_classes = class_names.size();


enum Algo_Type
{
	YOLOv5,
	YOLOv8,
};

enum Backend_Type
{
	Libtorch,
	ONNXRuntime,
	OpenCV,
	OpenVINO,
	TensorRT,
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
	virtual void init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path) = 0;

	void infer(const std::string input_path, char* argv[]);

	virtual void release() {};
	
protected:
	virtual void pre_process() = 0;	

	virtual void process() = 0;

	virtual void post_process() = 0;

	cv::Mat m_image;

	cv::Mat m_result;

	float* m_outputs_host;

	int m_output_numprob;

	int m_output_numbox;

	int m_output_numel;
};

class  BackendFactory
{
public:
	typedef std::unique_ptr<YOLO>(*CreateFunction)();

	static BackendFactory& instance();

	void register_backend(const Backend_Type& backend_type, CreateFunction create_function);

	std::unique_ptr<YOLO> create(const Backend_Type& backend_type);

private:
	BackendFactory();

	std::map<Backend_Type, CreateFunction> m_backend_registry;
};

