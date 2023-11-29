#include "yolov5_tensorrt.h"
#include "utils.h"


class TRTLogger : public nvinfer1::ILogger
{
public:
	virtual void log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept override
	{
	}
} logger;


std::vector<unsigned char> load_file(const std::string & file)
{
	std::ifstream in(file, std::ios::in | std::ios::binary);
	if (!in.is_open())
		return {};

	in.seekg(0, std::ios::end);
	size_t length = in.tellg();

	std::vector<uint8_t> data;
	if (length > 0)
	{
		in.seekg(0, std::ios::beg);
		data.resize(length);
		in.read((char*)& data[0], length);
	}
	in.close();
	return data;
}


YOLOv5_TensorRT::YOLOv5_TensorRT(std::string model_path, Device_Type device_type)
{
	TRTLogger logger;
	auto engine_data = load_file(model_path);
	auto runtime = nvinfer1::createInferRuntime(logger);
	auto engine = runtime->deserializeCudaEngine(engine_data.data(), engine_data.size());

	cudaStreamCreate(&m_stream);
	m_execution_context = engine->createExecutionContext();

	cudaMalloc(&m_inputs_device, sizeof(float) * input_numel);
	cudaMalloc(&m_outputs_device, sizeof(float) * output_numel);

	cudaMallocHost(&m_inputs_host, sizeof(float) * input_numel);
	cudaMallocHost(&m_outputs_host, sizeof(float) * output_numel);

	m_bindings[0] = m_inputs_device;
	m_bindings[1] = m_outputs_device;
}


void YOLOv5_TensorRT::pre_process()
{
	cv::Mat letterbox;
	LetterBox(m_image, letterbox, cv::Size(input_width, input_height));
	letterbox.convertTo(letterbox, CV_32FC3, 1.0f / 255.0f);

	int image_area = letterbox.cols * letterbox.rows;
	float* pimage = (float*)letterbox.data;
	float* phost_b = m_inputs_host + image_area * 0;
	float* phost_g = m_inputs_host + image_area * 1;
	float* phost_r = m_inputs_host + image_area * 2;
	for (int i = 0; i < image_area; ++i, pimage += 3)
	{
		*phost_r++ = pimage[0];
		*phost_g++ = pimage[1];
		*phost_b++ = pimage[2];
	}
}


void YOLOv5_TensorRT::process()
{
	cudaMemcpyAsync(m_inputs_device, m_inputs_host, sizeof(float) * input_numel, cudaMemcpyHostToDevice, m_stream);
	m_execution_context->enqueueV2((void**)m_bindings, m_stream, nullptr);
	cudaMemcpyAsync(m_outputs_host, m_outputs_device, sizeof(float) * output_numel, cudaMemcpyDeviceToHost, m_stream);
	cudaStreamSynchronize(m_stream);
}


void YOLOv5_TensorRT::post_process()
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

		float* classes_scores = 5 + ptr;
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
		scale_boxes(box, m_image.size());
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
		std::string label = class_names[class_ids[idx]] + ":" + cv::format("%.2f", scores[idx]); //class_ids[idx]��class_id
		draw_result(m_result, label, box);
	}
}


YOLOv5_TensorRT::~YOLOv5_TensorRT()
{
	cudaStreamDestroy(m_stream);
	cudaFree(m_inputs_device);
	cudaFree(m_outputs_device);
	cudaFreeHost(m_inputs_host);
	cudaFreeHost(m_outputs_host);
}