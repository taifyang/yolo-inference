/*
 * @Author: taifyang 58515915+taifyang@users.noreply.github.com
 * @Date: 2024-06-12 09:26:41
 * @LastEditors: taifyang 58515915+taifyang@users.noreply.github.com
 * @LastEditTime: 2024-06-17 23:04:59
 * @FilePath: \cpp\openvino\yolo_openvino.h
 * @Description: yolo算法的openvino推理框架头文件
 */

#pragma once

#include "yolo_classify.h"
#include "yolo_detect.h"
#include "yolo_segment.h"
#include "utils.h"
#include <openvino/openvino.hpp> 

/**
 * @description: yolo算法的openvino推理框架抽象类
 */
class YOLO_OpenVINO : virtual public YOLO
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
	 * @description: 推理请求
	 */
	ov::InferRequest m_infer_request;

	/**
	 * @description: 输入节点
	 */
	ov::Output<const ov::Node> m_input_port;

	/**
	 * @description: 输入图像
	 */
	cv::Mat m_input;
};

/**
 * @description: yolo分类算法的openvino推理框架类
 */
class YOLO_OpenVINO_Classify : public YOLO_OpenVINO, public YOLO_Classify
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
 * @description: yolo检测算法的openvino推理框架类
 */
class YOLO_OpenVINO_Detect : public YOLO_OpenVINO, public YOLO_Detect
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
 * @description: yolo分割算法的openvino推理框架类
 */
class YOLO_OpenVINO_Segment : public YOLO_OpenVINO, public YOLO_Segment
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