/* 
 * @Author: taifyang
 * @Date: 2024-06-12 09:26:41
 * @LastEditTime: 2026-01-03 20:41:05
 * @Description: source file for YOLO tensorrt inference
 */

#include "yolo_tensorrt.h"

class TRTLogger : public nvinfer1::ILogger
{
public:
	void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept
	{
	}
} logger;

void YOLO_TensorRT::init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path)
{
	m_algo_type = algo_type;

	if (device_type != GPU)
	{
		std::cerr << "TensorRT only support GPU!" << std::endl;
		std::exit(-1);
	}

	m_model_type = model_type;
	
	if(!std::filesystem::exists(model_path))
	{
		std::cerr << "model not exists!" << std::endl;
		std::exit(-1);
	}
	std::ifstream file(model_path, std::ios::binary);
	if (!file.good())
	{
		std::cerr << "read model error!" << std::endl;
		std::exit(-1);
	}

    std::stringstream buffer;
    buffer << file.rdbuf();

    std::string stream_model(buffer.str());

	TRTLogger logger;
	m_runtime = nvinfer1::createInferRuntime(logger);
	m_engine = m_runtime->deserializeCudaEngine(stream_model.data(), stream_model.size());
	if (m_engine == nullptr)
	{
		std::cerr << "tensorrt create engine error!" << std::endl;
		std::exit(-1);
	}

	cudaStreamCreate(&m_stream);
	m_execution_context = m_engine->createExecutionContext();
}

void YOLO_TensorRT::release()
{
	cudaStreamDestroy(m_stream);
	cudaFree(m_input_device);

#if NV_TENSORRT_MAJOR < 10
	m_execution_context->destroy();
	m_engine->destroy();
	m_runtime->destroy();
#endif 
}
