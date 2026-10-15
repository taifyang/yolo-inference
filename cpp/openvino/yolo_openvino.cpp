/* 
 * @Author: taifyang
 * @Date: 2024-06-12 09:26:41
 * @LastEditTime: 2026-10-12 21:35:22
 * @Description: source file for YOLO openvino inference
 */

#include "yolo_openvino.h"

void YOLO_OpenVINO::init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path)
{
	m_algo_type = algo_type;

	if(!std::filesystem::exists(model_path))
	{
		std::cerr << "model not exists!" << std::endl;
		std::exit(-1);
	}

	ov::Core core; //Initialize OpenVINO Runtime Core 
	ov::AnyMap config;
	if (device_type == CPU)
	{
		config["PERFORMANCE_HINT"] = ov::Any(ov::hint::PerformanceMode::LATENCY);
		config["NUM_STREAMS"] = ov::Any(1);
		config["INFERENCE_NUM_THREADS"] = ov::Any(m_threads);
		config["ENABLE_HYPER_THREADING"] = ov::Any(false);
	}

	ov::CompiledModel compiled_model;
	try
	{
		compiled_model = core.compile_model(model_path, device_type == GPU ? "GPU" : "CPU", config); //Compile the Model 
	}
	catch (const std::exception& e)
	{
		std::cerr << "openvino load model failed!" << std::endl;
		std::exit(-1);
	}

	m_infer_request = compiled_model.create_infer_request(); //Create an Inference Request 
	m_input_port = compiled_model.input(); //Get input port for model with one input
}
