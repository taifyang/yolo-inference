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
	"hair drier", "toothbrush" };			//Àà±ðÃû³Æ

const int input_width = 640;

const int input_height = 640;

const float score_threshold = 0.2;

const float nms_threshold = 0.5;

const float confidence_threshold = 0.2;

const int input_numel = 1 * 3 * input_width * input_height;

const int num_classes = class_names.size();

const int output_numprob = 5 + num_classes;

const int output_numbox = 3 * (input_width / 8 * input_height / 8 + input_width / 16 * input_height / 16 + input_width / 32 * input_height / 32);

const int output_numel = 1 * output_numprob * output_numbox;

enum Device_Type
{
	CPU,
	GPU,
};


class YOLOv5
{
public:
	void infer(const std::string image_path)
	{
		m_image = cv::imread(image_path);
		m_result = m_image.clone();
		pre_process();
		process();
		post_process();
		cv::imwrite("result.jpg", m_result);
		cv::imshow("result", m_result);
		cv::waitKey(0);
	}

	cv::Mat m_image;

	cv::Mat m_result;
	
private:
	virtual void pre_process() = 0;

	virtual void process() = 0;

	virtual void post_process() = 0;
};