/* 
 * @Author: taifyang
 * @Date: 2026-01-03 21:57:36
 * @LastEditTime: 2026-01-12 15:06:30
 * @Description: source file for YOLO tensorrt pose
 */

#include "yolo_tensorrt.h"
#include "cuda/preprocess.cuh"
#include "cuda/decode.cuh"

void YOLO_TensorRT_OBB::init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path)
{	
	if (algo_type != YOLOv8 && algo_type != YOLOv11 && algo_type != YOLOv12)
	{
		std::cerr << "unsupported algo type!" << std::endl;
		std::exit(-1);
	}
	YOLO_TensorRT::init(algo_type, device_type, model_type, model_path);
	YOLO_OBB::init(algo_type, device_type, model_type, model_path);

	m_task_type = OBB;

	cudaMalloc(&m_input_host, m_max_input_size);
	cudaMallocHost(&m_output0_host, sizeof(float) * m_output_numdet);

	cudaMalloc(&m_input_device, m_max_input_size);
	cudaMalloc(&m_output_device, sizeof(float) * m_output_numdet);

	m_bindings[0] = m_input_device;
	m_bindings[1] = m_output_device;

#ifdef _CUDA_PREPROCESS
	cudaMallocHost(&m_d2s_host, sizeof(float) * 6);
	cudaMalloc(&m_d2s_device, sizeof(float) * 6);
	cudaMallocHost(&m_s2d_host, sizeof(float) * 6);
	cudaMalloc(&m_s2d_device, sizeof(float) * 6);
#endif // _CUDA_PREPROCESS

#ifdef _CUDA_POSTPROCESS
	cudaMallocHost(&m_output_box_host, sizeof(float) * (m_num_box_element * m_max_box + 1));
	cudaMalloc(&m_output_box_device, sizeof(float) * (m_num_box_element * m_max_box + 1));
#endif // _CUDA_POSTPROCESS
}

void YOLO_TensorRT_OBB::pre_process()
{	
#ifdef _CUDA_PREPROCESS
	cudaMemcpyAsync(m_input_host, m_image.data, sizeof(uint8_t) * 3 * m_image.cols * m_image.rows, cudaMemcpyHostToDevice, m_stream);
	cuda_preprocess_img(m_input_host, m_image.cols, m_image.rows, m_input_device, m_input_size.width, m_input_size.height, m_d2s_host, m_s2d_host, m_stream);
	cudaMemcpyAsync(m_d2s_device, m_d2s_host, sizeof(float) * 6, cudaMemcpyHostToDevice, m_stream);
	cudaMemcpyAsync(m_s2d_device, m_s2d_host, sizeof(float) * 6, cudaMemcpyHostToDevice, m_stream);
#else
	cv::Mat letterbox;
	LetterBox(m_image, letterbox, m_params, cv::Size(m_input_size.width, m_input_size.height));
	int image_area = letterbox.cols * letterbox.rows;
	uchar* pimage = letterbox.data;
	float* phost_r = m_input_host + image_area * 0;
	float* phost_g = m_input_host + image_area * 1;
	float* phost_b = m_input_host + image_area * 2;
	for (int i = 0; i < letterbox.cols * letterbox.rows; ++i, pimage += 3)
	{
		*phost_r++ = pimage[2] / 255.0f;
		*phost_g++ = pimage[1] / 255.0f;
		*phost_b++ = pimage[0] / 255.0f;
	}
	cudaMemcpyAsync(m_input_device, m_input_host, sizeof(float) * m_input_numel, cudaMemcpyHostToDevice, m_stream);
#endif // _CUDA_PREPROCESS
}

void YOLO_TensorRT_OBB::process()
{
	m_execution_context->executeV2((void**)m_bindings);

#ifndef _CUDA_POSTPROCESS
	cudaMemcpyAsync(m_output0_host, m_output_device, sizeof(float) * m_output_numdet, cudaMemcpyDeviceToHost, m_stream);
#endif // !_CUDA_POSTPROCESS
}

void YOLO_TensorRT_OBB::post_process()
{	
	std::vector<std::vector<float>> boxes_rotated;
	std::vector<float> scores;
	std::vector<float> angles;
	std::vector<int> class_ids;

#ifdef _CUDA_POSTPROCESS
	cudaMemset(m_output_box_device, 0, sizeof(float) * (m_num_box_element * m_max_box + 1));
	cuda_decode(m_output_device, m_output_numbox, m_class_num, m_confidence_threshold, m_score_threshold, m_d2s_device,
		 m_output_box_device, m_max_box, m_num_box_element, m_input_size, m_stream, m_algo_type, m_task_type);
	cudaMemcpyAsync(m_output_box_host, m_output_box_device, sizeof(float) * (m_num_box_element * m_max_box + 1), cudaMemcpyDeviceToHost, m_stream);

	float* scores_device;
	cudaMalloc(&scores_device, sizeof(float) * m_max_box);
	cuda_extract_col(m_output_box_device, scores_device, 6, m_max_box, m_num_box_element, m_stream) ;

	int* sorted_idx_device;
	cudaMalloc(&sorted_idx_device, sizeof(int) * m_max_box);
	thrust_argsort(scores_device, sorted_idx_device, m_max_box);

	float* output_box_sorted_device;
	cudaMalloc(&output_box_sorted_device, sizeof(float) * m_max_box * m_num_box_element);
	cuda_extract_rows(m_output_box_device, sorted_idx_device, output_box_sorted_device, m_max_box, m_stream);

    float *a_d, *b_d, *c_d, *hd_d;
    cudaMalloc(&a_d, sizeof(float) * m_max_box);
    cudaMalloc(&b_d, sizeof(float) * m_max_box);
    cudaMalloc(&c_d, sizeof(float) * m_max_box );
    cudaMalloc(&hd_d,  sizeof(float) * m_max_box * m_max_box);

    cuda_compute_covariance_matrix(output_box_sorted_device, a_d, b_d, c_d, m_max_box, m_stream);

    cuda_compute_hd(output_box_sorted_device, output_box_sorted_device, a_d, b_d, c_d, a_d, b_d, c_d, hd_d, m_max_box, m_stream);

    cuda_triu_k1(hd_d, m_max_box, m_max_box, m_stream);

	int* picked_device;
	int picked_size;
	cudaMalloc((void**)&picked_device, sizeof(int) * m_max_box); 
	cuda_pick_bbox(hd_d, picked_device, m_nms_threshold, m_max_box, picked_size, m_stream);

	float* picked_boxes;
	cudaMalloc(&picked_boxes, sizeof(float) * picked_size * m_num_box_element);
	cuda_extract_rows(output_box_sorted_device, picked_device, picked_boxes, picked_size, m_stream);

	cuda_regularize_bbox(picked_boxes, picked_size, m_stream);

	float gain = std::min(m_input_size.width / (float)m_image.cols, m_input_size.height / (float)m_image.rows);
    float pad_w = (m_input_size.width - m_image.cols * gain) / 2;
	float pad_h = (m_input_size.height - m_image.rows * gain) / 2;
	cuda_scale_boxes(picked_boxes, picked_size, m_image.cols, m_image.rows, gain, pad_w, pad_h, m_stream);

	float* picked_boxes_host = new float[m_num_box_element*picked_size];
	cudaMemcpy(picked_boxes_host, picked_boxes, sizeof(float) * m_num_box_element * picked_size, cudaMemcpyDeviceToHost);

	m_output_obb.clear();
	m_output_obb.resize(picked_size);
	for (int i = 0; i < picked_size; i++)
	{
		OutputOBB output;
		output.score = picked_boxes_host[i*8+5];
		output.id = picked_boxes_host[i*8+6];
		float angle = picked_boxes_host[i*8+7] * 180 / CV_PI;
		output.box_rotate = cv::RotatedRect(cv::Point2f(picked_boxes_host[i*8+1], picked_boxes_host[i*8+2]), cv::Size2f(picked_boxes_host[i*8+3], picked_boxes_host[i*8+4]), angle);
		m_output_obb[i] = output;
	}

#else
	cudaMemcpyAsync(m_output_host, m_output_device, sizeof(float) * m_output_numdet, cudaMemcpyDeviceToHost, m_stream);
	cudaStreamSynchronize(m_stream);

	for (int i = 0; i < m_output_numbox; ++i)
	{
		float* ptr = m_output_host + i * m_output_numprob;
		int class_id;
		float score;
		float angle;
		if (m_algo_type == YOLOv8 || m_algo_type == YOLOv11 || m_algo_type == YOLOv12)
		{
			float* classes_scores = ptr + 4;
			class_id = std::max_element(classes_scores, classes_scores + m_class_num) - classes_scores;
			score = classes_scores[class_id];
			angle = ptr[m_class_num + 4];
		}

		if (score < m_score_threshold)
			continue;

		boxes_rotated.push_back(std::vector<float>{ptr[0], ptr[1], ptr[2], ptr[3], score, class_id, angle});
		scores.push_back(score);
		angles.push_back(angle);
		class_ids.push_back(class_id);
	}

	std::vector<int> indices;
	nms_rotated(boxes_rotated, scores, m_score_threshold, m_nms_threshold, indices);

	std::vector<std::vector<float>> boxes_nms(indices.size());
	for(int i=0; i<indices.size(); i++)	
		boxes_nms[i] = boxes_rotated[indices[i]];

	regularize_rboxes(boxes_nms);
	
	scale_rboxes(boxes_nms, m_image.size());

	m_output_obb.clear();
	m_output_obb.resize(indices.size());
	for (int i = 0; i < indices.size(); i++)
	{
		OutputOBB output;
		output.score = boxes_nms[i][4];
		output.id = boxes_nms[i][5];
		float angle = boxes_nms[i][6] * 180 / CV_PI;
		output.box_rotate = cv::RotatedRect(cv::Point2f(boxes_nms[i][0], boxes_nms[i][1]), cv::Size2f(boxes_nms[i][2], boxes_nms[i][3]), angle);
		m_output_obb[i] = output;
	}
#endif

	if(m_draw_result)
		draw_result(m_output_obb);
}

void YOLO_TensorRT_OBB::release()
{
	YOLO_TensorRT::release();

	cudaFree(m_output_device);

#ifdef _CUDA_PREPROCESS
	cudaFree(m_d2s_device);
	cudaFreeHost(m_d2s_host);
#endif // _CUDA_PREPROCESS

#ifdef _CUDA_POSTPROCESS
	cudaFree(m_output_box_device);
	cudaFreeHost(m_output_box_host);
#endif // _CUDA_POSTPROCESS
}