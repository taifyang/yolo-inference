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

void YOLO::infer(const std::string file_path, char* argv[], bool save_result, bool show_result)
{
	std::string suffix = file_path.substr(file_path.size() - 4);
	if (suffix == ".bmp" || suffix == ".jpg" || suffix == ".png")
	{
		m_image = cv::imread(file_path);
		if (m_image.empty())
		{
			std::cout << "read image empty!" << std::endl;
			std::exit(-1);
		}
		m_result = m_image.clone();

		pre_process();
		process();
		post_process();

		if (save_result)
		{
			std::string result_name = "./result/" + std::string(argv[1]) + std::string(argv[2]) + std::string(argv[3]) + std::string(argv[4]) + std::string(argv[5]) + ".jpg";
			cv::imwrite(result_name, m_result);
		}
		if (show_result)
		{
			cv::imshow("result", m_result);
			cv::waitKey(0);
			cv::destroyAllWindows();
		}
	}
	else if (suffix == ".mp4")
	{
		cv::VideoCapture cap;
		if (!cap.open(file_path))
		{
			std::cout << "read video empty!" << std::endl;
			std::exit(-1);
		}

		cv::VideoWriter wri;
		if (save_result)
		{
			std::string result_name = "./result/" + std::string(argv[1]) + std::string(argv[2]) + std::string(argv[3]) + std::string(argv[4]) + std::string(argv[5]) + ".avi";
			wri.open(result_name, cv::VideoWriter::fourcc('M', 'P', '4', '2'), 30, cv::Size(1280, 720));
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
		std::cout << "supported input file types: .bmp .jpg .png .mp4" << std::endl;
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
	//if (m_backend_registry.find(backend_type) == m_backend_registry.end())
	//{
	//	std::cout << "unsupported backend type!" << std::endl;
	//	std::exit(-1);
	//}

	if (backend_type >= m_create_registry.size())
	{
		std::cout << "unsupported backend type!" << std::endl;
		std::exit(-1);
	}
	if (task_type >= m_create_registry[task_type].size())
	{
		std::cout << "unsupported task type!" << std::endl;
		std::exit(-1);
	}

	return m_create_registry[backend_type][task_type]();
}

CreateFactory::CreateFactory()
{
	m_create_registry.resize(5, std::vector<CreateFunction>(3));

#ifdef _YOLO_LIBTORCH
	register_class(Backend_Type::Libtorch, Task_Type::Classification, []() -> std::unique_ptr<YOLO> { return std::make_unique<YOLO_Libtorch_Classification>(); });
	register_class(Backend_Type::Libtorch, Task_Type::Detection, []() -> std::unique_ptr<YOLO> { return std::make_unique<YOLO_Libtorch_Detection>(); });
	register_class(Backend_Type::Libtorch, Task_Type::Segmentation, []() -> std::unique_ptr<YOLO> { return std::make_unique<YOLO_Libtorch_Segmentation>(); });
#endif // _YOLO_Libtorch

#ifdef _YOLO_ONNXRUNTIME
	register_class(Backend_Type::ONNXRuntime, Task_Type::Classification, []() -> std::unique_ptr<YOLO> { return std::make_unique<YOLO_ONNXRuntime_Classification>(); });
	register_class(Backend_Type::ONNXRuntime, Task_Type::Detection, []() -> std::unique_ptr<YOLO> { return std::make_unique<YOLO_ONNXRuntime_Detection>(); });
	register_class(Backend_Type::ONNXRuntime, Task_Type::Segmentation, []() -> std::unique_ptr<YOLO> { return std::make_unique<YOLO_ONNXRuntime_Segmentation>(); });
#endif // _YOLO_ONNXRuntime

#ifdef _YOLO_OPENCV
	register_class(Backend_Type::OpenCV, Task_Type::Classification, []() -> std::unique_ptr<YOLO> { return std::make_unique<YOLO_OpenCV_Classification>(); });
	register_class(Backend_Type::OpenCV, Task_Type::Detection, []() -> std::unique_ptr<YOLO> { return std::make_unique<YOLO_OpenCV_Detection>(); });
	register_class(Backend_Type::OpenCV, Task_Type::Segmentation, []() -> std::unique_ptr<YOLO> { return std::make_unique<YOLO_OpenCV_Segmentation>(); });
#endif // _YOLO_OpenCV

#ifdef _YOLO_OPENVINO
	register_class(Backend_Type::OpenVINO, Task_Type::Classification, []() -> std::unique_ptr<YOLO> { return std::make_unique<YOLO_OpenVINO_Classification>(); });
	register_class(Backend_Type::OpenVINO, Task_Type::Detection,[]() -> std::unique_ptr<YOLO> { return std::make_unique<YOLO_OpenVINO_Detection>(); });
	register_class(Backend_Type::OpenVINO, Task_Type::Segmentation, []() -> std::unique_ptr<YOLO> { return std::make_unique<YOLO_OpenVINO_Segmentation>(); });
#endif // _YOLO_OpenVINO

#ifdef _YOLO_TENSORRT
	register_class(Backend_Type::TensorRT, Task_Type::Classification, []() -> std::unique_ptr<YOLO> { return std::make_unique<YOLO_TensorRT_Classification>(); });
	register_class(Backend_Type::TensorRT, Task_Type::Detection, []() -> std::unique_ptr<YOLO> { return std::make_unique<YOLO_TensorRT_Detection>(); });
	register_class(Backend_Type::TensorRT, Task_Type::Segmentation, []() -> std::unique_ptr<YOLO> { return std::make_unique<YOLO_TensorRT_Segmentation>(); });
#endif // _YOLO_TensorRT
}

