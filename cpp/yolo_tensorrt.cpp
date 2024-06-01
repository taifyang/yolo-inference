#include "yolo_tensorrt.h"
#include "utils.h"
#include "preprocess.h"
#include "decode.h"


class TRTLogger : public nvinfer1::ILogger
{
public:
	virtual void log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept override
	{
	}
} logger;


void YOLO_TensorRT::init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path)
{
	if (algo_type != YOLOv5 && algo_type != YOLOv8)
	{
		std::cout << "unsupported algo type!" << std::endl;
		std::exit(-1);
	}
	m_algo = algo_type;
	if (m_algo == YOLOv5)
	{
		m_output_numprob = 5 + num_classes;
		m_output_numbox = 3 * (input_width / 8 * input_height / 8 + input_width / 16 * input_height / 16 + input_width / 32 * input_height / 32);
		m_output_numel = 1 * m_output_numprob * m_output_numbox;
	}
	if (m_algo == YOLOv8)
	{
		m_output_numprob = 4 + num_classes;
		m_output_numbox = input_width / 8 * input_height / 8 + input_width / 16 * input_height / 16 + input_width / 32 * input_height / 32;
		m_output_numel = 1 * m_output_numprob * m_output_numbox;
	}

	if (device_type != GPU)
	{
		std::cout << "only support GPU!" << std::endl;
		std::exit(-1);
	}

	m_model = model_type;

	std::ifstream file(model_path, std::ios::binary);
	assert(file.good());
	file.seekg(0, file.end);
	size_t engine_data_size = file.tellg();
	file.seekg(0, file.beg);
	char* engine_data = new char[engine_data_size];
	assert(engine_data);
	file.read(engine_data, engine_data_size);
	file.close();

	TRTLogger logger;
	auto runtime = nvinfer1::createInferRuntime(logger);
	auto engine = runtime->deserializeCudaEngine(engine_data, engine_data_size);

	cudaStreamCreate(&m_stream);
	m_execution_context = engine->createExecutionContext();

	cudaMallocHost(&m_inputs_host, MAX_IMAGE_INPUT_SIZE_THRESH);
	cudaMallocHost(&m_outputs_host, sizeof(float) * m_output_numel);

	cudaMalloc(&m_inputs_device, MAX_IMAGE_INPUT_SIZE_THRESH);
	cudaMalloc(&m_outputs_device, sizeof(float) * m_output_numel);

	m_bindings[0] = m_inputs_device;
	m_bindings[1] = m_outputs_device;

#ifdef CUDA_PREPROCESS
	cudaMallocHost(&m_affine_matrix_host, sizeof(float) * 6);
	cudaMalloc(&m_affine_matrix_device, sizeof(float) * 6);
#endif // CUDA_PREPROCESS

#ifdef CUDA_POSTPROCESS
	cudaMallocHost(&m_outputs_box_host, sizeof(float) * (NUM_BOX_ELEMENT * MAX_IMAGE_BBOX + 1));
	cudaMalloc(&m_outputs_box_device, sizeof(float) * (NUM_BOX_ELEMENT * MAX_IMAGE_BBOX + 1));
#endif // CUDA_POSTPROCESS
}


void YOLO_TensorRT::pre_process()
{
#ifdef CUDA_PREPROCESS

	cudaMemcpyAsync(m_inputs_host, m_image.data, sizeof(uint8_t) * 3 * m_image.cols * m_image.rows, cudaMemcpyHostToDevice, m_stream);
	preprocess_kernel_img(m_inputs_host, m_image.cols, m_image.rows, m_inputs_device, input_width, input_height, m_affine_matrix_host, m_stream);
	cudaMemcpyAsync(m_affine_matrix_device, m_affine_matrix_host, sizeof(float) * 6, cudaMemcpyHostToDevice, m_stream);

#else

	cv::Mat letterbox;
	LetterBox(m_image, letterbox, cv::Size(input_width, input_height));

	int image_area = letterbox.cols * letterbox.rows;

	uchar* pimage = letterbox.data;
	float* phost_b = m_inputs_host + image_area * 0;
	float* phost_g = m_inputs_host + image_area * 1;
	float* phost_r = m_inputs_host + image_area * 2;
	for (int i = 0; i < image_area; ++i, pimage += 3)
	{
		*phost_r++ = pimage[0] / 255.0f;
		*phost_g++ = pimage[1] / 255.0f;
		*phost_b++ = pimage[2] / 255.0f;
	}

	cudaMemcpyAsync(m_inputs_device, m_inputs_host, sizeof(float) * input_numel, cudaMemcpyHostToDevice, m_stream);

#endif // CUDA_PREPROCESS
}


void YOLO_TensorRT::process()
{
	m_execution_context->enqueueV2((void**)m_bindings, m_stream, nullptr);

#ifndef CUDA_POSTPROCESS
	cudaMemcpyAsync(m_outputs_host, m_outputs_device, sizeof(float) * output_numel, cudaMemcpyDeviceToHost, m_stream);
	cudaStreamSynchronize(m_stream);
#endif // !CUDA_POSTPROCESS
}


void YOLO_TensorRT::post_process()
{
	std::vector<cv::Rect> boxes;
	std::vector<float> scores;
	std::vector<int> class_ids;

#ifdef CUDA_POSTPROCESS
	cudaMemset(m_outputs_box_device, 0, sizeof(float) * (NUM_BOX_ELEMENT * MAX_IMAGE_BBOX + 1));	
	decode_kernel_invoker(m_outputs_device, m_output_numbox, num_classes, confidence_threshold, m_affine_matrix_device, m_outputs_box_device, MAX_IMAGE_BBOX, m_stream, m_algo);
	nms_kernel_invoker(m_outputs_box_device, nms_threshold, MAX_IMAGE_BBOX, m_stream);
	cudaMemcpyAsync(m_outputs_box_host, m_outputs_box_device, sizeof(float) * (NUM_BOX_ELEMENT * MAX_IMAGE_BBOX + 1), cudaMemcpyDeviceToHost, m_stream);
	cudaStreamSynchronize(m_stream);

	for (size_t i = 0; i < MAX_IMAGE_BBOX; i++)
	{
		if (m_outputs_box_host[7 * i + 7])
		{
			float x1 = m_outputs_box_host[7 * i + 1];
			float y1 = m_outputs_box_host[7 * i + 2];
			float x2 = m_outputs_box_host[7 * i + 3];
			float y2 = m_outputs_box_host[7 * i + 4];
			int left = int(x1);
			int top = int(y1);
			int width = int(x2 - x1);
			int height = int(y2 - y1);

			boxes.push_back(cv::Rect(left, top, width, height));
			scores.push_back(m_outputs_box_host[7 * i + 5]);
			class_ids.push_back(m_outputs_box_host[7 * i + 6]);
		}
	}

	for (int i = 0; i < boxes.size(); i++)
	{
		cv::Rect box = boxes[i];
		std::string label = class_names[class_ids[i]] + ":" + cv::format("%.2f", scores[i]);
		draw_result(m_result, label, box);
	}

#else

	for (int i = 0; i < m_output_numbox; ++i)
	{
		float* ptr = m_outputs_host + i * m_output_numprob;
		int class_id;
		float score;
		if (m_algo == YOLOv5)
		{
			float objness = ptr[4];
			if (objness < confidence_threshold)
				continue;
			float* classes_scores = ptr + 5;
			class_id = std::max_element(classes_scores, classes_scores + num_classes) - classes_scores;
			score = classes_scores[class_id] * objness;
		}
		if (m_algo == YOLOv8)
		{
			float* classes_scores = ptr + 4;
			class_id = std::max_element(classes_scores, classes_scores + num_classes) - classes_scores;
			score = classes_scores[class_id];
		}
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

#endif // CUDA_POSTPROCESS
}


void YOLO_TensorRT::release()
{
	cudaStreamDestroy(m_stream);
	cudaFree(m_inputs_device);
	cudaFree(m_outputs_device);

#ifdef CUDA_PREPROCESS
	cudaFree(m_affine_matrix_device);
	cudaFreeHost(m_affine_matrix_host);
#endif // CUDA_PREPROCESS

#ifdef CUDA_POSTPROCESS
	cudaFree(m_outputs_box_device);
	cudaFreeHost(m_outputs_box_host);
#endif // CUDA_POSTPROCESS
}