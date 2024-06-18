/*
 * @Author: taifyang 58515915+taifyang@users.noreply.github.com
 * @Date: 2024-06-12 09:26:41
 * @LastEditors: taifyang 58515915+taifyang@users.noreply.github.com
 * @LastEditTime: 2024-06-18 22:39:32
 * @FilePath: \cpp\yolo.h
 * @Description: yolo接口类头文件
 */

#pragma once

//#define _YOLO_LIBTORCH
//#define _YOLO_ONNXRUNTIME
//#define _YOLO_OPENCV
//#define _YOLO_OPENVINO
//#define _YOLO_TENSORRT

#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>

/**
 * @description: 推理后端
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
 * @description: 推理任务
 */
enum Task_Type
{
	Classify,
	Detect,
	Segment,
};

/**
 * @description: 算法类型
 */
enum Algo_Type
{
	YOLOv5,
	YOLOv8,
};

/**
 * @description: 推理设备
 */
enum Device_Type
{
	CPU,
	GPU,
};

/**
 * @description: 模型精度
 */
enum Model_Type
{
	FP32,
	FP16,
	INT8,
};

/**
 * @description: 接口类
 */
class YOLO
{
public:
	/**
	 * @description: 析构函数
	 * @return {*}
	 */	
	virtual ~YOLO() {};	//这句代码保证虚拟继承析构的时候不会内存泄漏

	/**
	 * @description: 					初始化接口
	 * @param {Algo_Type} algo_type		算法类型
	 * @param {Device_Type} device_type	推理设备
	 * @param {Model_Type} model_type	模型精度
	 * @param {string} model_path		模型路径
	 * @return {*}
	 */
	virtual void init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path) = 0;

	/**
	 * @description: 				推理接口
	 * @param {string} file_path	文件路径		
	 * @param {bool} save_result	保存结果
	 * @param {bool} show_result	显示结果
	 * @param {char*} argv			程序入参
	 * @return {*}
	 */	
	void infer(const std::string file_path, bool save_result = true, bool show_result = true, char* argv[] = {}); 

	/**
	 * @description: 资源释放接口
	 * @return {*}
	 */
	virtual void release() {};

protected:
	/**
	 * @description: 模型前处理接口
	 * @return {*}
	 */
	virtual void pre_process() = 0;

	/**
	 * @description: 模型推理接口
	 * @return {*}
	 */
	virtual void process() = 0;

	/**
	 * @description: 模型后处理接口
	 * @return {*}
	 */
	virtual void post_process() = 0;

	/**
	 * @description: 输入图像
	 */
	cv::Mat m_image;

	/**
	 * @description: 图像结果
	 */
	cv::Mat m_result;

	/**
	 * @description: 推理图像宽度
	 */
	int m_input_width = 640;

	/**
	 * @description: 推理图像高度
	 */
	int m_input_height = 640;

	/**
	 * @description: 推理图像大小
	 */
	int m_input_numel = 1 * 3 * m_input_width * m_input_height;

	/**
	 * @description: 算法类型
	 */
	Algo_Type m_algo;

	/**
	 * @description: 模型精度
	 */
	Model_Type m_model;
};

/**
 * @description: 抽象工厂类
 */
class  CreateFactory
{
public:
	/**
	 * @description: 算法类创建函数指针别名
	 */
	typedef std::unique_ptr<YOLO>(*CreateFunction)();

	/**
	 * @description: 算法类实例
	 */
	static CreateFactory& instance();

	/**
	 * @description: 							算法类注册函数
	 * @param {Backend_Type&} backend_type		推理后端
	 * @param {Task_Type&} task_type			任务类型
	 * @param {CreateFunction} create_function	算法类创建函数指针
	 * @return {*}
	 */
	void register_class(const Backend_Type& backend_type, const Task_Type& task_type, CreateFunction create_function);

	/**
	 * @description: 						算法类创建函数
	 * @param {Backend_Type&} backend_type	推理后端
	 * @param {Task_Type&} task_type		任务类型
	 * @return {*}							具体算法类
	 */
	std::unique_ptr<YOLO> create(const Backend_Type& backend_type, const Task_Type& task_type);

private:
	/**
	 * @description: 构造函数
	 * @return {*}
	 */
	CreateFactory();

	/**
	 * @description: 算法类注册表
	 */
	std::vector<std::vector<CreateFunction>> m_create_registry;
};
