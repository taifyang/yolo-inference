/*
 * @Author: taifyang 
 * @Date: 2024-06-12 09:26:41
 * @LastEditors: taifyang 
 * @LastEditTime: 2024-06-17 23:21:07
 * @FilePath: \cpp\tensorrt\yolo_tensorrt.h
 * @Description: yolo算法的tensorrt推理框架头文件
 */

#pragma once

#include <cuda_runtime.h>
#include "yolo_classify.h"
#include "yolo_detect.h"
#include "yolo_segment.h"
#include "utils.h"
#include <NvInfer.h>
#include <NvInferRuntime.h>

/**
 * @description: 
 * @return {*}
 */
class YOLO_TensorRT : virtual public YOLO
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
	 * @description: 推理上下文
	 */
	nvinfer1::IExecutionContext* m_execution_context;

	/**
	 * @description: cuda流
	 */
	cudaStream_t m_stream;

	/**
	 * @description: 输入数据指针device
	 */
	float* m_input_device;

	/**
	 * @description: 最大输入大小
	 */
	const int m_max_input_size = sizeof(float) * m_input_numel;

	/**
	 * @description: 最大包围盒数
	 */
	const int m_max_image_bbox = 1024;
};

/**
 * @description: yolo分类算法的tensorrt推理框架类
 */
class YOLO_TensorRT_Classify : public YOLO_TensorRT, public YOLO_Classify
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
	 * @description: 资源释放
	 * @return {*}
	 */
	void release();

	/**
	 * @description: 输入输出tensor绑定
	 */
	float* m_bindings[2];

	/**
	 * @description: 输入数据指针host
	 */
	float* m_input_host;

	/**
	 * @description: 输出数据指针device
	 */
	float* m_output_device;
};

/**
 * @description: yolo检测算法的tensorrt推理框架类
 */
class YOLO_TensorRT_Detect : public YOLO_TensorRT, public YOLO_Detect
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
	 * @description: 资源释放
	 * @return {*}
	 */
	void release();

	/**
	 * @description: 输入输出tensor绑定
	 */
	float* m_bindings[2];

#ifndef _CUDA_PREPROCESS
	/**
	 * @description: 输入数据指针host
	 */
	float* m_input_host;
#else
	/**
	 * @description: 输入int8数据指针host
	 */
	uint8_t* m_input_host;
#endif // !_CUDA_PREPROCESS

	/**
	 * @description: 输出数据指针device
	 */
	float* m_output_device;

#ifdef _CUDA_PREPROCESS
	/**
	 * @description: 仿射变换矩阵host
	 */
	float* m_affine_matrix_host;

	/**
	 * @description: 仿射变换矩阵device
	 */
	float* m_affine_matrix_device;
#endif // _CUDA_PREPROCESS

#ifdef _CUDA_POSTPROCESS
	/**
	 * @description: 输出包围盒host
	 */
	float* m_output_box_host;

	/**
	 * @description: 输出包围盒device
	 */
	float* m_output_box_device;
#endif // _CUDA_POSTPROCESS
};

/**
 * @description: yolo分割算法的tensorrt推理框架类
 */
class YOLO_TensorRT_Segment : public YOLO_TensorRT, public YOLO_Segment
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
	 * @description: 资源释放
	 * @return {*}
	 */
	void release();

	/**
	 * @description: 输入输出tensor绑定
	 */
	float* m_bindings[3];

#ifndef _CUDA_PREPROCESS
	/**
	 * @description: 输入数据host
	 */
	float* m_input_host;
#else
	/**
	 * @description: 输入int8数据host
	 */
	uint8_t* m_input_host;
#endif // !_CUDA_PREPROCESS

	/**
	 * @description: 输出数据device
	 */
	float* m_output0_device;

	/**
	 * @description: 输出数据device
	 */
	float* m_output1_device;

#ifdef _CUDA_PREPROCESS
	/**
	 * @description: 仿射变换矩阵host
	 */
	float* m_affine_matrix_host;

	/**
	 * @description: 仿射变换矩阵device
	 */
	float* m_affine_matrix_device;
#endif // _CUDA_PREPROCESS

#ifdef _CUDA_POSTPROCESS
	/**
	 * @description: 输出包围盒host
	 */
	float* m_output_box_host;

	/**
	 * @description: 输出包围盒device
	 */
	float* m_output_box_device;
#endif // _CUDA_POSTPROCESS
};