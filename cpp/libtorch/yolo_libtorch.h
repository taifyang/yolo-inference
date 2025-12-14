 /* @Date: 2024-06-12 09:26:41
 * @LastEditors: taifyang
 * @LastEditTime: 2024-10-30 21:17:38
 * @FilePath: \cpp\libtorch\yolo_libtorch.h
 * @Description: libtorch inference header file for YOLO algorithm
 */

#pragma once

#include "yolo_classify.h"
#include "yolo_detect.h"
#include "yolo_segment.h"
#include "utils.h"
#include <torch/script.h>
#include <torch/torch.h>

/**
 * @description: libtorch inference class for YOLO algorithm
 */
class YOLO_Libtorch : virtual public YOLO
{	
public:
	/**
	 * @description: 					initialization interface
	 * @param {Algo_Type} algo_type		algorithm type
	 * @param {Device_Type} device_type	device type
	 * @param {Model_Type} model_type	model type
	 * @param {string} model_path		model path
	 * @return {*}
	 */
	void init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path);

protected:
	/**
	 * @description: device type
	 */
	torch::DeviceType m_device;

	/**
	 * @description: inference module
	 */
	torch::jit::script::Module m_module;

	/**
	 * @description: model input
	 */
	std::vector<torch::jit::IValue> m_input;

	/**
	 * @description: model output
	 */
	torch::jit::IValue m_output;
};

/**
 * @description: libtorch inference class for the yolo classification algorithm
 */
class YOLO_Libtorch_Classify : public YOLO_Libtorch, public YOLO_Classify
{
public:
	/**
	 * @description: 					initialization interface
	 * @param {Algo_Type} algo_type		algorithm type
	 * @param {Device_Type} device_type	device type
	 * @param {Model_Type} model_type	model type
	 * @param {string} model_path		model path
	 * @return {*}
	 */
	void init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path);

private:
	/**
	 * @description: model pre-process
	 * @return {*}
	 */
	void pre_process();

	/**
	 * @description: model inference
	 * @return {*}
	 */
	void process();

	/**
	 * @description: model post-process
	 * @return {*}
	 */
	void post_process();
};

/**
 * @description: libtorch inference class for the yolo detection algorithm
 */
class YOLO_Libtorch_Detect : public YOLO_Libtorch, public YOLO_Detect
{
public:
	/**
	 * @description: 					initialization interface
	 * @param {Algo_Type} algo_type		algorithm type
	 * @param {Device_Type} device_type	device type
	 * @param {Model_Type} model_type	model type
	 * @param {string} model_path		model path
	 * @return {*}
	 */
	void init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path);

private:
	/**
	 * @description: model pre-process
	 * @return {*}
	 */
	void pre_process();

	/**
	 * @description: model inference
	 * @return {*}
	 */
	void process();

	/**
	 * @description: model post-process
	 * @return {*}
	 */
	void post_process();
};

/**
 * @description: libtorch inference class for the yolo segmentation algorithm
 */
class YOLO_Libtorch_Segment : public YOLO_Libtorch, public YOLO_Segment
{
public:
	/**
	 * @description: 					initialization interface
	 * @param {Algo_Type} algo_type		algorithm type
	 * @param {Device_Type} device_type	device type
	 * @param {Model_Type} model_type	model type
	 * @param {string} model_path		model path
	 * @return {*}
	 */
	void init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path);

private:
	/**
	 * @description: model pre-process
	 * @return {*}
	 */
	void pre_process();

	/**
	 * @description: model inference
	 * @return {*}
	 */
	void process();

	/**
	 * @description: model post-process
	 * @return {*}
	 */
	void post_process();
};