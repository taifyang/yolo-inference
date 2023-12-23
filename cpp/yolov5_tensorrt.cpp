#include "yolov5_tensorrt.h"
#include "utils.h"


class TRTLogger : public nvinfer1::ILogger
{
public:
	virtual void log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept override
	{
	}
} logger;


YOLOv5_TensorRT::YOLOv5_TensorRT(std::string model_path, Device_Type device_type, Model_Type model_type)
{
	assert(("only support GPU!", device_type == GPU));

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

	cudaMalloc(&m_inputs_device, sizeof(float) * input_numel);
	cudaMalloc(&m_outputs_device, sizeof(float) * output_numel);

	cudaMallocHost(&m_inputs_host, sizeof(float) * input_numel);
	cudaMallocHost(&m_outputs_host, sizeof(float) * output_numel);
	if (m_model == FP16)
	{
		cudaMallocHost(&m_inputs_host_fp16, sizeof(uint16_t) * input_numel);
		cudaMallocHost(&m_outputs_host_fp16, sizeof(uint16_t) * output_numel);
	}

	m_bindings[0] = m_inputs_device;
	m_bindings[1] = m_outputs_device;
}


void YOLOv5_TensorRT::pre_process()
{
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
	if (m_model == FP16)
	{		
		for (size_t i = 0; i < input_numel; i++)
		{
			m_inputs_host_fp16[i] = float32_to_float16(m_inputs_host[i]);
		}
	}
}


void YOLOv5_TensorRT::process()
{
	if (m_model == FP32 || m_model == INT8)
	{
		cudaMemcpyAsync(m_inputs_device, m_inputs_host, sizeof(float) * input_numel, cudaMemcpyHostToDevice, m_stream);
		m_execution_context->enqueueV2((void**)m_bindings, m_stream, nullptr);
		cudaMemcpyAsync(m_outputs_host, m_outputs_device, sizeof(float) * output_numel, cudaMemcpyDeviceToHost, m_stream);
		cudaStreamSynchronize(m_stream);
	}
	else if (m_model == FP16)
	{
		cudaMemcpyAsync(m_inputs_device, m_inputs_host_fp16, sizeof(uint16_t) * input_numel, cudaMemcpyHostToDevice, m_stream);
		m_execution_context->enqueueV2((void**)m_bindings, m_stream, nullptr);
		cudaMemcpyAsync(m_outputs_host_fp16, m_outputs_device, sizeof(uint16_t) * output_numel, cudaMemcpyDeviceToHost, m_stream);
		cudaStreamSynchronize(m_stream);
		for (size_t i = 0; i < output_numel; i++)
		{
			m_outputs_host[i] = float16_to_float32(m_outputs_host_fp16[i]);
		}
	}
}


YOLOv5_TensorRT::~YOLOv5_TensorRT()
{
	cudaStreamDestroy(m_stream);
	cudaFree(m_inputs_device);
	cudaFree(m_outputs_device);
	cudaFreeHost(m_inputs_host);
	cudaFreeHost(m_outputs_host);
	if (m_model == FP16)
	{
		cudaFreeHost(m_inputs_host_fp16);
		cudaFreeHost(m_outputs_host_fp16);
	}
}