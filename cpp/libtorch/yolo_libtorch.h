/*
 * @Author: taifyang 
 * @Date: 2024-06-12 09:26:41
 * @LastEditors: taifyang 
 * @LastEditTime: 2024-08-06 21:20:25
 * @FilePath: \cpp\libtorch\yolo_libtorch.h
 * @Description: yolo算法的libtorch推理框架头文件
 */

#pragma once

#include "yolo_classify.h"
#include "yolo_detect.h"
#include "yolo_segment.h"
#include "utils.h"
#include <torch/script.h>
#include <torch/torch.h>

/**
 * @description: yolo算法的libtorch推理框架抽象类
 */
class YOLO_Libtorch : virtual public YOLO
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

protected:
	/**
	 * @description: 设备类型
	 */
	torch::DeviceType m_device;

	/**
	 * @description: 推理模块
	 */
	torch::jit::script::Module m_module;

	/**
	 * @description: 模型输入
	 */
	std::vector<torch::jit::IValue> m_input;

	/**
	 * @description: 模型输出
	 */
	torch::jit::IValue m_output;
};

/**
 * @description: yolo分类算法的libtorch推理框架类
 */
class YOLO_Libtorch_Classify : public YOLO_Libtorch, public YOLO_Classify
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
	 * @description: 前处理
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
 * @description: yolo检测算法的libtorch推理框架类
 */
class YOLO_Libtorch_Detect : public YOLO_Libtorch, public YOLO_Detect
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
 * @description: yolo分割算法的libtorch推理框架类
 */
class YOLO_Libtorch_Segment : public YOLO_Libtorch, public YOLO_Segment
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