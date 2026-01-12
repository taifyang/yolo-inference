/* 
 * @Author: taifyang
 * @Date: 2024-06-12 09:26:41
 * @LastEditTime: 2026-01-08 23:30:54
 * @Description: header file for YOLO onnxruntime inference
 */

#pragma once

#include "yolo_classify.h"
#include "yolo_detect.h"
#include "yolo_segment.h"
#include "yolo_pose.h"
#include "yolo_obb.h"
#include "utils.h"
#include <onnxruntime_cxx_api.h>

/**
 * @description: class for YOLO onnxruntime inference
 * @return {*}
 */
class YOLO_ONNXRuntime : virtual public YOLO
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

	/**
	 * @description: release interface
	 * @return {*}
	 */
	void release();

protected:
	/**
	 * @description: infenence environment
	 */
	Ort::Env m_env;

	/**
	 * @description: infenence session
	 */
	Ort::Session* m_session;

	/**
	 * @description: options allocator
	 */
	Ort::AllocatorWithDefaultOptions m_allocator;

	/**
	 * @description: input data
	 */
	std::vector<float> m_input;

	/**
	 * @description: input fp16 data 
	 */
	std::vector<uint16_t> m_input_fp16;
	
	/**
	 * @description: output0 data
	 */
	std::vector<float> m_output0;
		
	/**
	 * @description: output1 data
	 */
	std::vector<float> m_output1;

	/**
	 * @description: output0 fp16 data 
	 */
	std::vector<uint16_t> m_output0_fp16;

	/**
	 * @description: output1 fp16 data 
	 */
	std::vector<uint16_t> m_output1_fp16;

	/**
	 * @description: input node names
	 */
	std::vector<const char*> m_input_names;

	/**
	 * @description: output node names
	 */
	std::vector<const char*> m_output_names;
};

/**
 * @description: class for the yolo onnxruntime classification inference
 */
class YOLO_ONNXRuntime_Classify : public YOLO_ONNXRuntime, public YOLO_Classify
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
 * @description: class for the yolo onnxruntime detection inference
 */
class YOLO_ONNXRuntime_Detect : public YOLO_ONNXRuntime, virtual public YOLO_Detect
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
 * @description: class for the yolo onnxruntime segmentation inference
 */
class YOLO_ONNXRuntime_Segment : public YOLO_ONNXRuntime_Detect, public YOLO_Segment
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
 * @description: class for the yolo onnxruntime pose inference
 */
class YOLO_ONNXRuntime_Pose : public YOLO_ONNXRuntime_Detect, public YOLO_Pose
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
 * @description: class for the yolo onnxruntime obb inference
 */
class YOLO_ONNXRuntime_OBB : public YOLO_ONNXRuntime_Detect, public YOLO_OBB
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
