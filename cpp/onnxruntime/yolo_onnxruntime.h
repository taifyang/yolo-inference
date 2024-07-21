/*
 * @Author: taifyang 
 * @Date: 2024-06-12 09:26:41
 * @LastEditors: taifyang 
 * @LastEditTime: 2024-06-17 22:52:49
 * @FilePath: \cpp\onnxruntime\yolo_onnxruntime.h
 * @Description: yolo算法的onnxruntime推理框架头文件
 */

#pragma once

#include "yolo_classify.h"
#include "yolo_detect.h"
#include "yolo_segment.h"
#include "utils.h"
#include <onnxruntime_cxx_api.h>

/**
 * @description: yolo算法的onnxruntime推理框架抽象类
 * @return {*}
 */
class YOLO_ONNXRuntime : virtual public YOLO
{
public:
	/**
	 * @description: 					初始化接口
	 * @param {Algo_Type} algo_type		算法类型
	 * @param {Device_Type} device_type	推理设备
	 * @param {Model_Type} model_type	模型精度
	 * @param {string} model_path		模型路径
	 * @return {*}
	 */
	void init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path);

	/**
	 * @description: 资源释放接口
	 * @return {*}
	 */
	void release();

protected:
	/**
	 * @description: ort推理环境
	 */
	Ort::Env m_env;

	/**
	 * @description: ort推理会话
	 */
	Ort::Session* m_session;

	/**
	 * @description: 输入数据
	 */
	std::vector<float> m_input;

	/**
	 * @description: 输入数据fp16
	 */
	std::vector<uint16_t> m_input_fp16;

	/**
	 * @description: 输出数据fp16
	 */
	std::vector<uint16_t> m_output_fp16;

	/**
	 * @description: 网络输入节点名
	 */
	std::vector<const char*> m_input_names;

	/**
	 * @description: 网络输出节点名
	 */
	std::vector<const char*> m_output_names;
};

/**
 * @description: yolo分类算法的onnxruntime推理框架类
 */
class YOLO_ONNXRuntime_Classify : public YOLO_ONNXRuntime, public YOLO_Classify
{
public:
	/**
	 * @description: 					初始化接口
	 * @param {Algo_Type} algo_type		算法类型
	 * @param {Device_Type} device_type	推理设备
	 * @param {Model_Type} model_type	模型精度
	 * @param {string} model_path		模型路径
	 * @return {*}
	 */
	void init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path);

private:
	/**
	 * @description: 模型前处理
	 * @return {*}
	 */
	void pre_process();

	/**
	 * @description: 模型推理
	 * @return {*}
	 */
	void process();

	/**
	 * @description: 模型后处理
	 * @return {*}
	 */
	void post_process();
};

/**
 * @description: yolo检测算法的onnxruntime推理框架类
 */
class YOLO_ONNXRuntime_Detect : public YOLO_ONNXRuntime, public YOLO_Detect
{
public:
	/**
	 * @description: 					初始化接口
	 * @param {Algo_Type} algo_type		算法类型
	 * @param {Device_Type} device_type	推理设备
	 * @param {Model_Type} model_type	模型精度
	 * @param {string} model_path		模型路径
	 * @return {*}
	 */
	void init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path);

private:
	/**
	 * @description: 模型前处理
	 * @return {*}
	 */
	void pre_process();

	/**
	 * @description: 模型推理
	 * @return {*}
	 */
	void process();

	/**
	 * @description: 模型后处理
	 * @return {*}
	 */
	void post_process();
};

/**
 * @description: yolo分割算法的onnxruntime推理框架类
 */
class YOLO_ONNXRuntime_Segment : public YOLO_ONNXRuntime, public YOLO_Segment
{
public:
	/**
	 * @description: 					初始化接口
	 * @param {Algo_Type} algo_type		算法类型
	 * @param {Device_Type} device_type	推理设备
	 * @param {Model_Type} model_type	模型精度
	 * @param {string} model_path		模型路径
	 * @return {*}
	 */
	void init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path);

private:
	/**
	 * @description: 模型前处理
	 * @return {*}
	 */
	void pre_process();

	/**
	 * @description: 模型推理
	 * @return {*}
	 */
	void process();

	/**
	 * @description: 模型后处理
	 * @return {*}
	 */
	void post_process();

	/**
	 * @description: 输出fp16
	 */
	std::vector<uint16_t> m_output0_fp16;

	/**
	 * @description: 输出fp16
	 */
	std::vector<uint16_t> m_output1_fp16;
};