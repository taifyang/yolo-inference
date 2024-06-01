#include "yolo.h"
#include "utils.h"
#include "yolo_libtorch.h"
#include "yolo_onnxruntime.h"
#include "yolo_opencv.h"
#include "yolo_openvino.h"
#include "yolo_tensorrt.h"
#include <chrono>

//#define SAVE_RESULT
//#define SHOW_RESULT


void YOLO::infer(const std::string input_path, char* argv[])
{
	std::string suffix = input_path.substr(input_path.size() - 4);
	if (suffix == ".bmp" || suffix == ".jpg" || suffix == ".png")
	{
		m_image = cv::imread(input_path);
		if (m_image.empty())
		{
			std::cout << "read image empty!" << std::endl;
			std::exit(-1);
		}

		m_result = m_image.clone();
		pre_process();
		process();
		post_process();

		cv::imwrite("result.jpg", m_result);
		cv::imshow("result", m_result);
		cv::waitKey(0);
		cv::destroyAllWindows();
	}
	else if (suffix == ".mp4")
	{
		cv::VideoCapture cap(input_path);

#ifdef SAVE_RESULT
		cv::VideoWriter wri;
		std::string video_name = std::string(argv[1])+std::string(argv[2])+std::string(argv[3])+std::string(argv[4])+".avi";
		std::cout << video_name << " ";
		wri.open(video_name, cv::VideoWriter::fourcc('M', 'P', '4', '2'), 30, cv::Size(1280, 720));
#endif // SAVE_RESULT

		auto start = std::chrono::steady_clock::now();

		while (
#ifdef SHOW_RESULT
			cv::waitKey(1) < 0
#else
			1
#endif // SHOW_RESULT
			)
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

#ifdef SAVE_RESULT
			wri << m_result;
#endif // SAVE_RESULT

#ifdef SHOW_RESULT
			cv::imshow("result", m_result);
#endif // SHOW_RESULT
		}	

		auto end = std::chrono::steady_clock::now();	
		std::chrono::duration<double> duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);		
		std::cout << duration.count() * 1000 << "ms" << std::endl;
		cap.release();
		cv::destroyAllWindows();

#ifdef SAVE_RESULT
		wri.release();
#endif // SAVE_RESULT
	}
	else
	{
		std::cout << "supported input types: .bmp .jpg .png .mp4" << std::endl;
		std::exit(-1);
	}
}


BackendFactory& BackendFactory::instance()
{
	static BackendFactory backend_factory;
	return backend_factory;
}


void BackendFactory::register_backend(const Backend_Type& backend_type, CreateFunction create_function)
{
	m_backend_registry[backend_type] = create_function;
}


std::unique_ptr<YOLO> BackendFactory::create(const Backend_Type& backend_type)
{
	if (m_backend_registry.find(backend_type) == m_backend_registry.end())
	{
		std::cout << "unsupported backend type!" << std::endl;
		std::exit(-1);
	}
	return m_backend_registry[backend_type]();
}


BackendFactory::BackendFactory()
{
	register_backend(Backend_Type::Libtorch, []() -> std::unique_ptr<YOLO> { return std::make_unique<YOLO_Libtorch>(); });
	register_backend(Backend_Type::ONNXRuntime, []() -> std::unique_ptr<YOLO> { return std::make_unique<YOLO_ONNXRuntime>(); });
	register_backend(Backend_Type::OpenCV, []() -> std::unique_ptr<YOLO> { return std::make_unique<YOLO_OpenCV>(); });
	register_backend(Backend_Type::OpenVINO, []() -> std::unique_ptr<YOLO> { return std::make_unique<YOLO_OpenVINO>(); });
	register_backend(Backend_Type::TensorRT, []() -> std::unique_ptr<YOLO> { return std::make_unique<YOLO_TensorRT>(); });
}

