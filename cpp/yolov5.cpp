#include "yolov5.h"
#include "utils.h"
#include "yolov5_libtorch.h"
#include "yolov5_onnxruntime.h"
#include "yolov5_opencv.h"
#include "yolov5_openvino.h"
#include "yolov5_tensorrt.h"


void YOLOv5::infer(const std::string image_path)
{
	std::string suffix = image_path.substr(image_path.size() - 4);
	if (suffix == ".bmp" || suffix == ".jpg" || suffix == ".png")
	{
		m_image = cv::imread(image_path);
		m_result = m_image.clone();
		pre_process();
		process();
		post_process();
		cv::imwrite("result.jpg", m_result);
		cv::imshow("result", m_result);
		cv::waitKey(0);
	}
	else if (suffix == ".mp4")
	{
		cv::VideoCapture cap(image_path);
		while (cv::waitKey(1) < 0)
		{
			clock_t start = clock();

			cap.read(m_image);
			if (m_image.empty())
				break;

			m_result = m_image.clone();
			pre_process();
			process();
			post_process();
			cv::imshow("result", m_result);

			clock_t end = clock();
			std::cout << end - start << "ms" << std::endl;
		}
	}
}

	
void YOLOv5::post_process()
{
	std::vector<cv::Rect> boxes;
	std::vector<float> scores;
	std::vector<int> class_ids;

	for (int i = 0; i < output_numbox; ++i)
	{
		float* ptr = m_outputs_host + i * output_numprob;
		float objness = ptr[4];
		if (objness < confidence_threshold)
			continue;

		float* classes_scores = ptr + 5;
		int class_id = std::max_element(classes_scores, classes_scores + num_classes) - classes_scores;
		float max_class_score = classes_scores[class_id];
		float score = max_class_score * objness;
		if (score < score_threshold)
			continue;

		float x = ptr[0];
		float y = ptr[1];
		float w = ptr[2];
		float h = ptr[3];
		int left = int(x - 0.5 * w);
		int top = int(y - 0.5 * h);
		int width = int(w);
		int height = int(h);

		cv::Rect box = cv::Rect(left, top, width, height);
		scale_box(box, m_image.size());
		boxes.push_back(box);
		scores.push_back(score);
		class_ids.push_back(class_id);
	}

	std::vector<int> indices;
	nms(boxes, scores, score_threshold, nms_threshold, indices);
	for (int i = 0; i < indices.size(); i++)
	{
		int idx = indices[i];
		cv::Rect box = boxes[idx];
		std::string label = class_names[class_ids[idx]] + ":" + cv::format("%.2f", scores[idx]);
		draw_result(m_result, label, box);
	}
}


AlgoFactory& AlgoFactory::instance()
{
	static AlgoFactory algo_factory;
	return algo_factory;
}


void AlgoFactory::register_algo(const Algo_Type& algo_type, CreateFunction create_function)
{
	m_algo_registry[algo_type] = create_function;
}


std::unique_ptr<YOLOv5> AlgoFactory::create(const Algo_Type& algo_type)
{
	assert(("algo type not exists!", m_algo_registry.find(algo_type) != m_algo_registry.end()));
	return m_algo_registry[algo_type]();
}


AlgoFactory::AlgoFactory()
{
	register_algo(Algo_Type::Libtorch, []() -> std::unique_ptr<YOLOv5> { return std::make_unique<YOLOv5_Libtorch>(); });
	register_algo(Algo_Type::ONNXRuntime, []() -> std::unique_ptr<YOLOv5> { return std::make_unique<YOLOv5_ONNXRuntime>(); });
	register_algo(Algo_Type::OpenCV, []() -> std::unique_ptr<YOLOv5> { return std::make_unique<YOLOv5_OpenCV>(); });
	register_algo(Algo_Type::OpenVINO, []() -> std::unique_ptr<YOLOv5> { return std::make_unique<YOLOv5_OpenVINO>(); });
	register_algo(Algo_Type::TensorRT, []() -> std::unique_ptr<YOLOv5> { return std::make_unique<YOLOv5_TensorRT>(); });
}

