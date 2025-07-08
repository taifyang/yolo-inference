/*
 * @Author: taifyang 
 * @Date: 2024-06-12 09:26:41
 * @LastEditors: taifyang 58515915+taifyang@users.noreply.github.com
 * @LastEditTime: 2024-11-22 22:58:34
 * @FilePath: \cpp\yolo.h
 * @Description: header file for YOLO algorithm
 */

#pragma once

#include <iostream>
#include <fstream>
#include <filesystem>
#include <opencv2/opencv.hpp>

/**
 * @description: backend type
 */
enum Backend_Type
{
	Libtorch,
	ONNXRuntime,
	OpenCV,
	OpenVINO,
	TensorRT,
};

/**
 * @description: task type
 */
enum Task_Type
{
	Classify,
	Detect,
	Segment,
};

/**
 * @description: algorithm type
 */
enum Algo_Type
{
	YOLOv5,
	YOLOv6,
	YOLOv7,
	YOLOv8,
	YOLOv9,
	YOLOv10,
	YOLOv11,
	YOLOv12,
	YOLOv13,
};

/**
 * @description: device type
 */
enum Device_Type
{
	CPU,
	GPU,
};

/**
 * @description: model type
 */
enum Model_Type
{
	FP32,
	FP16,
	INT8,
};

/**
 * @description: interface class for YOLO algorithm
 */
class YOLO
{
public:
	/**
	 * @description: destructor
	 * @return {*}
	 */	
	virtual ~YOLO() {};	

	/**
	 * @description: 					initialization interface
	 * @param {Algo_Type} algo_type		algorithm type
	 * @param {Device_Type} device_type	device type
	 * @param {Model_Type} model_type	model type
	 * @param {string} model_path		model path
	 * @return {*}
	 */
	virtual void init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path) = 0;

	/**
	 * @description: 				inference interface
	 * @param {string} file_path	file path		
	 * @param {bool} save_result	save result
	 * @param {bool} show_result	show result
	 * @param {char*} argv			argv
	 * @return {*}
	 */	
	void infer(const std::string file_path, bool save_result = true, bool show_result = true, char* argv[] = {}); 

	/**
	 * @description: release interface
	 * @return {*}
	 */
	virtual void release() {};

protected:
	/**
	 * @description: model pre-process interface
	 * @return {*}
	 */
	virtual void pre_process() = 0;

	/**
	 * @description: model inference interface
	 * @return {*}
	 */
	virtual void process() = 0;

	/**
	 * @description: model post-process interface
	 * @return {*}
	 */
	virtual void post_process() = 0;

	/**
	 * @description: input image
	 */
	cv::Mat m_image;

	/**
	 * @description: result
	 */
	cv::Mat m_result;

	/**
	 * @description: model input image width
	 */
	int m_input_width = 640;

	/**
	 * @description: model input image height
	 */
	int m_input_height = 640;

	/**
	 * @description: model input image size
	 */
	int m_input_numel = 1 * 3 * m_input_width * m_input_height;

	/**
	 * @description: algorithm type
	 */
	Algo_Type m_algo_type;

	/**
	 * @description: model type
	 */
	Model_Type m_model_type;

	/**
	 * @description: draw result
	 */
	bool m_draw_result;
};

/**
 * @description: abstract Factory Class
 */
class CreateFactory
{
public:
	/**
	 * @description: algorithm class creates function pointer alias
	 */
	typedef std::unique_ptr<YOLO>(*CreateFunction)();

	/**
	 * @description: algorithm Class Instance
	 */
	static CreateFactory& instance();

	/**
	 * @description: 							register class
	 * @param {Backend_Type&} backend_type		backend type
	 * @param {Task_Type&} task_type			task type
	 * @param {CreateFunction} create_function	pointer to algorithm class
	 * @return {*}
	 */
	void register_class(const Backend_Type& backend_type, const Task_Type& task_type, CreateFunction create_function);

	/**
	 * @description: 						create algorithm class
	 * @param {Backend_Type&} backend_type	backend type
	 * @param {Task_Type&} task_type		task type
	 * @return {*}							algorithm class
	 */
	std::unique_ptr<YOLO> create(const Backend_Type& backend_type, const Task_Type& task_type);

private:
	/**
	 * @description: constructor
	 * @return {*}
	 */
	CreateFactory();

	/**
	 * @description: registry for algorithm class 
	 */
	std::vector<std::vector<CreateFunction>> m_create_registry;
};
