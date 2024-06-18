/*
 * @Author: taifyang 58515915+taifyang@users.noreply.github.com
 * @Date: 2024-06-12 09:26:41
 * @LastEditors: taifyang 58515915+taifyang@users.noreply.github.com
 * @LastEditTime: 2024-06-17 22:54:36
 * @FilePath: \cpp\opencv\yolo_opencv.h
 * @Description: yolo算法的opencv推理框架头文件
 */

#pragma once

#include "yolo_classify.h"
#include "yolo_detect.h"
#include "yolo_segment.h"
#include "utils.h"
#include <opencv2/opencv.hpp>

/**
 * @description: yolo算法的opencv推理框架抽象类
 */
class YOLO_OpenCV : virtual public YOLO
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
	 * @description: 模型推理
	 * @return {*}
	 */
	void process();

	/**
	 * @description: 推理引擎
	 */
	cv::dnn::Net m_net;

	/**
	 * @description: 输入图像
	 */
	cv::Mat m_input;

	/**
	 * @description: 输出结果
	 */
	std::vector<cv::Mat> m_output;
};

/**
 * @description: yolo分类算法的opencv推理框架类
 */
class YOLO_OpenCV_Classify : public YOLO_OpenCV, public YOLO_Classify
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
	 * @description: 模型后处理
	 * @return {*}
	 */
	void post_process();
};

/**
 * @description: yolo检测算法的opencv推理框架类
 */
class YOLO_OpenCV_Detect : public YOLO_OpenCV, public YOLO_Detect
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
	 * @description: 模型后处理
	 * @return {*}
	 */
	void post_process();
};

/**
 * @description: yolo分割算法的opencv推理框架类
 */
class YOLO_OpenCV_Segment : public YOLO_OpenCV, public YOLO_Segment
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
	 * @description: 模型后处理
	 * @return {*}
	 */
	void post_process();
};