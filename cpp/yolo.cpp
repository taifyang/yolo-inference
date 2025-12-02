/*
 * @Author: taifyang 
 * @Date: 2024-06-12 09:26:41
 * @LastEditors: taifyang 58515915+taifyang@users.noreply.github.com
 * @LastEditTime: 2025-10-15 21:59:51
 * @FilePath: \cpp\yolo.cpp
 * @Description: source file for YOLO algorithm
 */

#include <chrono>
#include "yolo.h"
#include "utils.h"

#ifdef _YOLO_LIBTORCH
	#include "yolo_libtorch.h"
#endif // _YOLO_Libtorch

#ifdef _YOLO_ONNXRUNTIME
	#include "yolo_onnxruntime.h"
#endif // _YOLO_ONNXRuntime

#ifdef _YOLO_OPENCV
	#include "yolo_opencv.h"
#endif // _YOLO_OpenCV

#ifdef _YOLO_OPENVINO
	#include "yolo_openvino.h"
#endif // _YOLO_OpenVINO

#ifdef _YOLO_TENSORRT
	#include "yolo_tensorrt.h"
#endif // _YOLO_TensorRT

void YOLO::infer(const std::string file_path, bool save_result, bool show_result, char* argv[])
{
	if(!std::filesystem::exists(file_path))
	{
		std::cerr << "file not exists!" << std::endl;
		std::exit(-1);
	}

	m_draw_result = save_result || show_result;
	std::string extension = std::filesystem::path(file_path).extension().string();
	if (extension == ".bmp" || extension == ".jpg" || extension == ".png")
	{
		m_image = cv::imread(file_path);
		if (m_image.empty())
		{
			std::cerr << "read image empty!" << std::endl;
			std::exit(-1);
		}

		//warm up
		for(int i=0; i<10; ++i)
		{
			pre_process();
			process();
			post_process();
		}	

		auto start = std::chrono::steady_clock::now();
		for(int i=0; i<1000; ++i)
		{
		 	pre_process();
		 	process();
		 	post_process();
		}		
		auto end = std::chrono::steady_clock::now();	
		std::chrono::duration<double> duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);		
		std::cout << "avg cost run on 1000 times:" << duration.count() << "ms" << std::endl;

		if (save_result)
		{
			std::string result_name = "./result/" + std::string(argv[1]) + "_" + std::string(argv[2]) + "_" +std::string(argv[3]) + "_"  + std::string(argv[4]) + "_"  + std::string(argv[5]) + ".jpg";
			cv::imwrite(result_name, m_result);
		}
		if (show_result)
		{
			cv::imshow("result", m_result);
			cv::waitKey(0);
			cv::destroyAllWindows();
		}
	}
	else if (extension == ".mp4")
	{
		cv::VideoCapture cap;
		if (!cap.open(file_path))
		{
			std::cerr << "cannot read video!" << std::endl;
			std::exit(-1);
		}

		cv::VideoWriter wri;
		if (save_result)
		{
			std::string result_name = "./result/" + std::string(argv[1]) + "_" + std::string(argv[2]) + "_" +std::string(argv[3]) + "_"  + std::string(argv[4]) + "_"  + std::string(argv[5]) + ".avi";
			int width = int(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    		int height = int(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
			wri.open(result_name, cv::VideoWriter::fourcc('M', 'P', '4', '2'), 30, cv::Size(width, height));
		}

		auto start = std::chrono::steady_clock::now();

		while (true)
		{
			cap.read(m_image);
			if (m_image.empty())
			{
				break;
			}
			m_result = m_image.clone();

			pre_process();
			process();
			post_process();

			if (save_result)
			{
				wri << m_result;
			}
			if (show_result)
			{
				cv::imshow("result", m_result);
				cv::waitKey(1);
			}
		}	

		auto end = std::chrono::steady_clock::now();	
		std::chrono::duration<double> duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);		
		std::cout << duration.count() * 1000 << "ms" << std::endl;
		cap.release();

		if (save_result)
		{
			wri.release();
		}
		if (show_result)
		{
			cv::destroyAllWindows();
		}
	}
	else
	{
		std::cerr << "supported input file types: .bmp .jpg .png .mp4" << std::endl;
		std::exit(-1);
	}
}

CreateFactory& CreateFactory::instance()
{
	static CreateFactory create_factory;
	return create_factory;
}

void CreateFactory::register_class(const Backend_Type& backend_type, const Task_Type& task_type, CreateFunction create_function)
{
	m_create_registry[backend_type][task_type] = create_function;
}

std::unique_ptr<YOLO> CreateFactory::create(const Backend_Type& backend_type, const Task_Type& task_type)
{
	std::unique_ptr<YOLO> yolo = m_create_registry[backend_type][task_type]();
	if(yolo == nullptr)
	{
		std::cerr << "algo create failed!" <<std::endl;
		std::exit(-1);
	}
	else 
	{
		return yolo;
	}
}

CreateFactory::CreateFactory()
{
	size_t backend_size = magic_enum::enum_count<Backend_Type>();
	size_t task_size = magic_enum::enum_count<Task_Type>();
	m_create_registry.resize(backend_size, std::vector<CreateFunction>(task_size));

#ifdef _YOLO_LIBTORCH
	register_class(Backend_Type::Libtorch, Task_Type::Classify, []() -> std::unique_ptr<YOLO> { return std::make_unique<YOLO_Libtorch_Classify>(); });
	register_class(Backend_Type::Libtorch, Task_Type::Detect, []() -> std::unique_ptr<YOLO> { return std::make_unique<YOLO_Libtorch_Detect>(); });
	register_class(Backend_Type::Libtorch, Task_Type::Segment, []() -> std::unique_ptr<YOLO> { return std::make_unique<YOLO_Libtorch_Segment>(); });
#else
	register_class(Backend_Type::Libtorch, Task_Type::Classify, []() -> std::unique_ptr<YOLO> { return nullptr; });
	register_class(Backend_Type::Libtorch, Task_Type::Detect, []() -> std::unique_ptr<YOLO> { return nullptr; });
	register_class(Backend_Type::Libtorch, Task_Type::Segment, []() -> std::unique_ptr<YOLO> { return nullptr; });
#endif // _YOLO_Libtorch

#ifdef _YOLO_ONNXRUNTIME
	register_class(Backend_Type::ONNXRuntime, Task_Type::Classify, []() -> std::unique_ptr<YOLO> { return std::make_unique<YOLO_ONNXRuntime_Classify>(); });
	register_class(Backend_Type::ONNXRuntime, Task_Type::Detect, []() -> std::unique_ptr<YOLO> { return std::make_unique<YOLO_ONNXRuntime_Detect>(); });
	register_class(Backend_Type::ONNXRuntime, Task_Type::Segment, []() -> std::unique_ptr<YOLO> { return std::make_unique<YOLO_ONNXRuntime_Segment>(); });
#else
	register_class(Backend_Type::ONNXRuntime, Task_Type::Classify, []() -> std::unique_ptr<YOLO> { return nullptr; });
	register_class(Backend_Type::ONNXRuntime, Task_Type::Detect, []() -> std::unique_ptr<YOLO> { return nullptr; });
	register_class(Backend_Type::ONNXRuntime, Task_Type::Segment, []() -> std::unique_ptr<YOLO> { return nullptr; });
#endif // _YOLO_ONNXRuntime

#ifdef _YOLO_OPENCV
	register_class(Backend_Type::OpenCV, Task_Type::Classify, []() -> std::unique_ptr<YOLO> { return std::make_unique<YOLO_OpenCV_Classify>(); });
	register_class(Backend_Type::OpenCV, Task_Type::Detect, []() -> std::unique_ptr<YOLO> { return std::make_unique<YOLO_OpenCV_Detect>(); });
	register_class(Backend_Type::OpenCV, Task_Type::Segment, []() -> std::unique_ptr<YOLO> { return std::make_unique<YOLO_OpenCV_Segment>(); });
#else
	register_class(Backend_Type::OpenCV, Task_Type::Classify, []() -> std::unique_ptr<YOLO> { return nullptr; });
	register_class(Backend_Type::OpenCV, Task_Type::Detect, []() -> std::unique_ptr<YOLO> { return nullptr; });
	register_class(Backend_Type::OpenCV, Task_Type::Segment, []() -> std::unique_ptr<YOLO> { return nullptr; });
#endif // _YOLO_OpenCV

#ifdef _YOLO_OPENVINO
	register_class(Backend_Type::OpenVINO, Task_Type::Classify, []() -> std::unique_ptr<YOLO> { return std::make_unique<YOLO_OpenVINO_Classify>(); });
	register_class(Backend_Type::OpenVINO, Task_Type::Detect,[]() -> std::unique_ptr<YOLO> { return std::make_unique<YOLO_OpenVINO_Detect>(); });
	register_class(Backend_Type::OpenVINO, Task_Type::Segment, []() -> std::unique_ptr<YOLO> { return std::make_unique<YOLO_OpenVINO_Segment>(); });
#else
	register_class(Backend_Type::OpenVINO, Task_Type::Classify, []() -> std::unique_ptr<YOLO> { return nullptr; });
	register_class(Backend_Type::OpenVINO, Task_Type::Detect,[]() -> std::unique_ptr<YOLO> { return nullptr; });
	register_class(Backend_Type::OpenVINO, Task_Type::Segment, []() -> std::unique_ptr<YOLO> { return nullptr; });
#endif // _YOLO_OpenVINO

#ifdef _YOLO_TENSORRT
	register_class(Backend_Type::TensorRT, Task_Type::Classify, []() -> std::unique_ptr<YOLO> { return std::make_unique<YOLO_TensorRT_Classify>(); });
	register_class(Backend_Type::TensorRT, Task_Type::Detect, []() -> std::unique_ptr<YOLO> { return std::make_unique<YOLO_TensorRT_Detect>(); });
	register_class(Backend_Type::TensorRT, Task_Type::Segment, []() -> std::unique_ptr<YOLO> { return std::make_unique<YOLO_TensorRT_Segment>(); });
#else
	register_class(Backend_Type::TensorRT, Task_Type::Classify, []() -> std::unique_ptr<YOLO> { return nullptr; });
	register_class(Backend_Type::TensorRT, Task_Type::Detect, []() -> std::unique_ptr<YOLO> { return nullptr; });
	register_class(Backend_Type::TensorRT, Task_Type::Segment, []() -> std::unique_ptr<YOLO> { return nullptr; });
#endif // _YOLO_TensorRT
}
