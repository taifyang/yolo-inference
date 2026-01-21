/*
 * @Author: taifyang 
 * @Date: 2024-06-12 09:26:41
 * @LastEditTime: 2026-01-19 18:09:32
 * @Description: source file for cuda post-processing decoding
 */

#include "decode.cuh"

dim3 grid_dims(int numJobs) 
{
	int numBlockThreads = numJobs < BLOCK_SIZE ? numJobs : BLOCK_SIZE;
	return dim3(((numJobs + numBlockThreads - 1) / (float)numBlockThreads));
}

dim3 block_dims(int numJobs) 
{
	return numJobs < BLOCK_SIZE ? numJobs : BLOCK_SIZE;
}

static __device__ void affine_project_gpu(float* matrix, float x, float y, float* ox, float* oy) 
{
	*ox = matrix[0] * x + matrix[1] * y + matrix[2];
	*oy = matrix[3] * x + matrix[4] * y + matrix[5];
}

static __global__ void decode_kernel(float* predict, int num_bboxes, int num_classes, float confidence_threshold, float score_threshold, float* inverse_affine_matrix, float* parray, int max_objects, int num_box_element, cv::Size input_size, Algo_Type algo_type, Task_Type task_type)
{
	int position = blockDim.x * blockIdx.x + threadIdx.x;
	if (position >= num_bboxes)
		return;

	float* pitem;
	float objectness;
	float* class_score;
	float angle;
	if(algo_type == YOLOv3 || algo_type == YOLOv4 || algo_type == YOLOv6 || algo_type == YOLOv8 || algo_type == YOLOv9 || algo_type == YOLOv10 || algo_type == YOLOv11 || algo_type == YOLOv12 || algo_type == YOLOv13 || algo_type == YOLO26)
	{
		if(task_type == Detect)
		{
            if(algo_type == YOLO26)
                pitem = predict + 6 * position;
            else
			    pitem = predict + (4 + num_classes) * position;
		}
		else if(task_type == Segment)
		{
            if(algo_type == YOLO26)
                pitem = predict + 38 * position;
            else
			    pitem = predict + (36 + num_classes) * position;
		}
        else if(task_type == Pose)
        {
            if (algo_type == YOLOv8 || algo_type == YOLOv11 || algo_type == YOLOv12)
                pitem = predict + 56 * position;
            else if(algo_type == YOLO26)
                pitem = predict + 57 * position;
        }
		else if(task_type == OBB)
		{
            if (algo_type == YOLOv8 || algo_type == YOLOv11 || algo_type == YOLOv12)
            {
                pitem = predict + (5 + num_classes) * position;
                angle = pitem[4 + num_classes];
            }
            else if(algo_type == YOLO26)
            {
                pitem = predict + 7 * position;
                angle = pitem[6];
            }
		}
		class_score = pitem + 4;
	}
	else if(algo_type == YOLOv5 || algo_type == YOLOv7)
	{
		if(task_type == Detect)
		{
			pitem = predict + (5 + num_classes) * position;
		}
		if(task_type == Segment)
		{
			pitem = predict + (37 + num_classes) * position;
		}
		objectness = pitem[4];
		if (objectness < confidence_threshold)
			return;
		class_score = pitem + 5;
	}
	
	float score = *class_score++;
	int label = 0;
    if(algo_type == YOLO26)
    {
        label = pitem[5];
    }
    else
    {
        for (int i = 1; i < num_classes; ++i, ++class_score)
        {
            if (*class_score > score && algo_type != Pose)
            {
                score = *class_score;
                label = i;
            }
        }
    }
	
	if(algo_type == YOLOv5 || algo_type == YOLOv7)
	{
		score *= objectness;
	}
	if (score < score_threshold)
		return;

	int index = atomicAdd(parray, 1);
	if (index >= max_objects)
		return;

	if(task_type == OBB)
	{
		float* pout_item = parray + 1 + index * num_box_element;
		*pout_item++ = *pitem++;
		*pout_item++ = *pitem++;
		*pout_item++ = *pitem++;
		*pout_item++ = *pitem++;
		*pout_item++ = score;
		*pout_item++ = label;
		*pout_item++ = angle;
		*pout_item++ = 1; // 1 = keep, 0 = ignore
	}
	else
	{
		float left, top, right, bottom;
		if(algo_type == YOLOv3 || algo_type == YOLOv5 || algo_type == YOLOv6 || algo_type == YOLOv7 || algo_type == YOLOv8 || algo_type == YOLOv9 || algo_type == YOLOv11 || algo_type == YOLOv12 || algo_type == YOLOv13)	
		{
			float cx = *pitem++;
			float cy = *pitem++;
			float width = *pitem++;
			float height = *pitem++;
			left = cx - width * 0.5f;
			top = cy - height * 0.5f;
			right = cx + width * 0.5f;
			bottom = cy + height * 0.5f;
		}
		else if (algo_type == YOLOv4)
		{
			left = *pitem++ * input_size.width;
			top = *pitem++ * input_size.height;
			right = *pitem++ * input_size.width;
			bottom = *pitem++ * input_size.height;
		}
		else if (algo_type == YOLOv10 || algo_type == YOLO26)
		{
			left = *pitem++;
			top = *pitem++;
			right = *pitem++;
			bottom = *pitem++;
		}

		affine_project_gpu(inverse_affine_matrix, left, top, &left, &top);
		affine_project_gpu(inverse_affine_matrix, right, bottom, &right, &bottom);

		float* pout_item = parray + 1 + index * num_box_element;
		*pout_item++ = left;
		*pout_item++ = top;
		*pout_item++ = right;
		*pout_item++ = bottom;
		*pout_item++ = score;
		if(task_type == Pose)
		{
			*pitem++;
            if(algo_type == YOLO26)
                *pitem++;
			*pout_item++ = 0;
			*pout_item++ = 1; // 1 = keep, 0 = ignore
			for(int i=0; i<51; i++)  
				*pout_item++ = *pitem++;  
			pout_item = parray + 1 + index * num_box_element;
			for(int i=0; i<17; i++)        
				affine_project_gpu(inverse_affine_matrix, pout_item[3 * i + 7],  pout_item[3 * i + 8], &pout_item[3 * i + 7],  &pout_item[3 * i + 8]);          
		}
		else
		{
			*pout_item++ = label;
			*pout_item++ = 1; // 1 = keep, 0 = ignore
			if(task_type == Segment)
			{
				*pout_item++ = position;
			}
		}
	}
}

static __device__ float box_iou(float aleft, float atop, float aright, float abottom, float bleft, float btop, float bright, float bbottom) 
{
	float cleft = max(aleft, bleft);
	float ctop = max(atop, btop);
	float cright = min(aright, bright);
	float cbottom = min(abottom, bbottom);

	float c_area = max(cright - cleft, 0.0f) * max(cbottom - ctop, 0.0f);
	if (c_area == 0.0f)
		return 0.0f;

	float a_area = max(0.0f, aright - aleft) * max(0.0f, abottom - atop);
	float b_area = max(0.0f, bright - bleft) * max(0.0f, bbottom - btop);
	return c_area / (a_area + b_area - c_area);
}

static __global__ void nms_kernel(float* bboxes, int max_objects, float threshold, int num_box_element) 
{
	int position = (blockDim.x * blockIdx.x + threadIdx.x);
	int count = min((int)* bboxes, max_objects);
	if (position >= count)
		return;

	// left, top, right, bottom, confidence, class, keepflag
	float* pcurrent = bboxes + 1 + position * num_box_element;
	for (int i = 0; i < count; ++i) 
	{
		float* pitem = bboxes + 1 + i * num_box_element;
		if (i == position || pcurrent[5] != pitem[5]) 
			continue;

		if (pitem[4] >= pcurrent[4]) 
		{
			if (pitem[4] == pcurrent[4] && i < position)
				continue;

			float iou = box_iou(pcurrent[0], pcurrent[1], pcurrent[2], pcurrent[3], pitem[0], pitem[1], pitem[2], pitem[3]);

			if (iou > threshold)
			{
				pcurrent[6] = 0;  // 1=keep, 0=ignore
				return;
			}
		}
	}
}

void cuda_decode(float* predict, int num_bboxes, int num_classes, float confidence_threshold, float score_threshold, float* inverse_affine_matrix, 
    float* parray, int max_objects, int num_box_element, cv::Size input_size, Algo_Type algo_type, Task_Type task_type)
{
	auto grid = grid_dims(num_bboxes);
	auto block = block_dims(num_bboxes);
	decode_kernel<<<grid, block>>> (predict, num_bboxes, num_classes, confidence_threshold, score_threshold, inverse_affine_matrix, parray, max_objects, num_box_element, input_size, algo_type, task_type);
}

void cuda_nms(float* parray, float nms_threshold, int max_objects, int num_box_element)
{
	auto grid = grid_dims(max_objects);
	auto block = block_dims(max_objects);
	nms_kernel<<<grid, block>>> (parray, max_objects, nms_threshold, num_box_element);
}

 static __global__ void cuda_decode_mask_kernel(int left, int top, float* mask_weights, float* mask_predict, int mask_width, int mask_height, 
    unsigned char *mask_out, int mask_dim, int out_width, int out_height)
{
	// mask_predict to mask_out
	// mask_weights @ mask_predict
	int dx = blockDim.x * blockIdx.x + threadIdx.x;
	int dy = blockDim.y * blockIdx.y + threadIdx.y;
	if (dx >= out_width || dy >= out_height)
		return;

	int sx = left + dx;
	int sy = top + dy;
	if (sx < 0 || sx >= mask_width || sy < 0 || sy >= mask_height)
	{
		mask_out[dy * out_width + dx] = 0;
		return;
	}

	float cumprod = 0;
	for (int ic = 0; ic < mask_dim; ++ic)
	{
		float cval = mask_predict[(ic * mask_height + sy) * mask_width + sx];
		float wval = mask_weights[ic];
		cumprod += cval * wval;
	}

	float alpha = 1.0f / (1.0f + exp(-cumprod));
	mask_out[dy * out_width + dx] = alpha*255;
}

void cuda_decode_mask(float left, float top, float* mask_weights, float* mask_predict, int mask_width, int mask_height, 
    uint8_t* mask_out, int mask_dim, int out_width, int out_height)
{
	// mask_weights is mask_dim(32 element) gpu pointer
	dim3 grid((out_width + 31) / 32, (out_height + 31) / 32);
	dim3 block(32, 32);
	cuda_decode_mask_kernel<<<grid, block>>>(left, top, mask_weights, mask_predict, mask_width, mask_height, mask_out, mask_dim, out_width, out_height);
}

__global__ void extract_colKernel(const float* input_d, float* output_d, int target_col, int rows, int cols) 
{
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    if (row < rows) 
        output_d[row] = input_d[row * cols + target_col];
}

void cuda_extract_col(const float* input_d, float* output_d, int target_col, int rows, int cols) 
{       
    auto grid = grid_dims(rows);
	auto block = block_dims(rows);  
    extract_colKernel<<<grid, block>>>(input_d, output_d, target_col, rows, cols);
}

void thrust_argsort(float* scores_d, int* sorted_idx_d, int num_rboxes)
{
    thrust::device_ptr<float> dev_ptr_scores(scores_d);
    thrust::device_ptr<int> dev_ptr_indices(sorted_idx_d);
    thrust::sequence(thrust::device,  dev_ptr_indices,  dev_ptr_indices + num_rboxes);
    thrust::sort_by_key(thrust::device, dev_ptr_scores, dev_ptr_scores + num_rboxes, dev_ptr_indices, thrust::greater<float>());
}

__global__ void columnMaxKernel(const float* ious_d, float* col_max_d, int dim)
 {
    int col = blockIdx.x;
    int start_row = threadIdx.x;
    int step = blockDim.x;

    if (col < dim) 
    {
        float local_max = -1e9f;
        for (int row = start_row; row < dim; row += step) 
        {
            float current_val = ious_d[row * dim + col];
            if (current_val > local_max) 
                local_max = current_val;
        }
        atomicMax((unsigned int*)&col_max_d[col], __float_as_uint(local_max));
    }
}

__global__ void thresholdMaskKernel(const float* col_max_d, int* mask_d, float threshold, int dim) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < dim) 
        mask_d[idx] = (col_max_d[idx] < threshold) ? 1 : 0;
}

__global__ void prefixSumKernel(const int* mask_d, int* prefix_sum_d, int dim) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < dim) 
    {
        prefix_sum_d[idx] = 0;
        for (int i = 0; i <= idx; ++i) 
            prefix_sum_d[idx] += mask_d[i];
    }
}

__global__ void collectIndicesKernel(const int* mask_d, const int* prefix_sum_d, int* pick_d, int dim)
 {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < dim) 
    {
        if (mask_d[idx] == 1) 
            pick_d[prefix_sum_d[idx] - 1] = idx;
    }
}

void cuda_pick_bbox(const float* ious_d, int* pick_d, float threshold, int num_rboxes, int& pick_size) 
{
    float* col_max_d = nullptr;
    int* mask_d = nullptr;
    int* prefix_sum_d = nullptr;

    cudaMalloc((void**)&col_max_d, num_rboxes * sizeof(float));
    cudaMalloc((void**)&mask_d, num_rboxes * sizeof(int));
    cudaMalloc((void**)&prefix_sum_d, num_rboxes * sizeof(int)); 
    
    auto grid = grid_dims(num_rboxes);
	auto block = block_dims(num_rboxes);  

    columnMaxKernel<<<num_rboxes, block>>>(ious_d, col_max_d, num_rboxes);

    thresholdMaskKernel<<<grid, block>>>(col_max_d, mask_d, threshold, num_rboxes);

    prefixSumKernel<<<grid, block>>>(mask_d, prefix_sum_d, num_rboxes);

    cudaMemcpy(&pick_size, &prefix_sum_d[num_rboxes - 1], sizeof(int), cudaMemcpyDeviceToHost);

    collectIndicesKernel<<<grid, block>>>(mask_d, prefix_sum_d, pick_d, num_rboxes);
}


__global__ void extract_rowsKernel(const float* input_d, const int* index_d, float* output_d, int num_rboxes) 
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_rboxes) 
    {
        int input_row = index_d[tid];
        for (int j = 0; j < 8; ++j) 
            output_d[tid * 8 + j] = input_d[input_row * 8 + j];
    }
}

void cuda_extract_rows(const float* input_d, const int* index_d, float* output_d, int num_rboxes) 
{   
    auto grid = grid_dims(num_rboxes);
	auto block = block_dims(num_rboxes);                  
    extract_rowsKernel<<<grid, block>>>(input_d, index_d, output_d, num_rboxes);
}


__global__ void compute_covariance_matrix_kernel(const float* rboxes,  float* out1,  float* out2,  float* out3, int num_rboxes) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_rboxes)
        return;
    
    float w = rboxes[i * 8 + 3];
    float h = rboxes[i * 8 + 4];
    float theta = rboxes[i * 8 + 7];

    float a = (w * w) / 12.0f;
    float b = (h * h) / 12.0f;

    float cos_theta = __cosf(theta);  
    float sin_theta = __sinf(theta);  
    float cos2 = cos_theta * cos_theta;
    float sin2 = sin_theta * sin_theta;

    out1[i] = a * cos2 + b * sin2;
    out2[i] = a * sin2 + b * cos2;
    out3[i] = (a - b) * cos_theta * sin_theta;
}

void cuda_compute_covariance_matrix(const float* rboxes, float* a_d, float* b_d, float* c_d, int num_rboxes)
 {
    auto grid = grid_dims(num_rboxes);
	auto block = block_dims(num_rboxes);  
    compute_covariance_matrix_kernel<<<grid, block>>>(rboxes, a_d, b_d, c_d, num_rboxes);
}

__global__ void compute_obb_pairwise_hd_kernel(
    const float* __restrict__ obb1_d,
    const float* __restrict__ obb2_d,
    const float* __restrict__ a1_d,
    const float* __restrict__ b1_d,
    const float* __restrict__ c1_d,
    const float* __restrict__ a2_d,
    const float* __restrict__ b2_d,
    const float* __restrict__ c2_d,
    float* __restrict__ hd_d,
    int num_bbox) 
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= num_bbox || j >= num_bbox) 
        return;

    const float x1 = obb1_d[i * 8 + 1];
    const float y1 = obb1_d[i * 8 + 2];

    const float x2 = obb2_d[j * 8 + 1];
    const float y2 = obb2_d[j * 8 + 2];

    const float a1i = a1_d[i];
    const float b1i = b1_d[i];
    const float c1i = c1_d[i];

    const float a2j = a2_d[j];
    const float b2j = b2_d[j];
    const float c2j = c2_d[j];

    const float dx = x1 - x2;
    const float dy = y1 - y2;
    const float dx_sq = dx * dx;
    const float dy_sq = dy * dy;

    const float a_sum = a1i + a2j;
    const float b_sum = b1i + b2j;
    const float c_sum = c1i + c2j;

    const float den = a_sum * b_sum - c_sum * c_sum + EPS;

    const float t1_numer = a_sum * dy_sq + b_sum * dx_sq;
    const float t1 = (t1_numer / den) * 0.25f;

    const float t2_numer = c_sum * (-dx) * dy;
    const float t2 = (t2_numer / den) * 0.5f;

    const float term1 = fmaxf(a1i * b1i - c1i * c1i, 0.0f);
    const float term2 = fmaxf(a2j * b2j - c2j * c2j, 0.0f);

    const float t3_numer = a_sum * b_sum - c_sum * c_sum;
    const float t3_denom = 4.0f * sqrtf(term1 * term2) + EPS;
    const float t3_log_arg = (t3_numer / t3_denom) + EPS;
    const float t3 = logf(t3_log_arg) * 0.5f;

    float bd = t1 + t2 + t3;
    bd = fmaxf(bd, EPS);
    bd = fminf(bd, 100.0f);
    hd_d[i * num_bbox + j] = 1 - sqrtf(1.0f - expf(-bd) + EPS);
}

void cuda_compute_hd(
    const float* obb1_d,
    const float* obb2_d,
    const float* a1_d,
    const float* b1_d,
    const float* c1_d,
    const float* a2_d,
    const float* b2_d,
    const float* c2_d,
    float* hd_d, int num_bbox) 
{
    dim3 block_dim(32, 32);
    dim3 grid_dim((num_bbox + block_dim.x - 1) / block_dim.x, (num_bbox + block_dim.y - 1) / block_dim.y);
    compute_obb_pairwise_hd_kernel<<<grid_dim, block_dim>>>(obb1_d, obb2_d, a1_d, b1_d, c1_d, a2_d, b2_d, c2_d, hd_d, num_bbox);
}


__global__ void triu_k1_kernel(float* d_mat, int rows, int cols)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;  
    int j = blockIdx.y * blockDim.y + threadIdx.y;  
    if (i >= rows || j >= cols) 
        return;
    if (j - i < 1) 
        d_mat[i * cols + j] = 0.0f;
}

void cuda_triu_k1(float* d_mat, int rows, int cols) 
{
    dim3 block(32, 32); 
    dim3 grid((rows + block.x - 1) / block.x, (cols + block.y - 1) / block.y);
    triu_k1_kernel<<<grid, block>>>(d_mat, rows, cols);
}

__global__ void regularize_bboxKernel(float* rboxes_d, int num_bbox) 
{
    int box_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (box_idx >= num_bbox) 
        return;

    float w = rboxes_d[box_idx * 8 + 3];
    float h = rboxes_d[box_idx * 8 + 4];
    float t = rboxes_d[box_idx * 8 + 7];

    if (w <= h) 
    { 
        rboxes_d[box_idx * 8 + 3] = h;
        rboxes_d[box_idx * 8 + 4] = w;
        float new_t = fmodf(t + M_PI / 2.0, M_PI);     
        if (new_t < 0.0f)   new_t += M_PI;
        rboxes_d[box_idx * 8 + 7] = new_t;
    }
}

void cuda_regularize_bbox(float* rboxes_d, int num_bboxes)
{
    auto grid = grid_dims(num_bboxes);
	auto block = block_dims(num_bboxes);
    regularize_bboxKernel<<<grid, block>>>(rboxes_d, num_bboxes);
}

__device__ float clamp_dice(float val, float min_val, float max_val) 
{
    if (val < min_val) return min_val;
    if (val > max_val) return max_val;
    return val;
}

__global__ void scale_boxes_kernel(float* boxes_d, int num_bbox, float output_w, float output_h, float gain, float pad_w, float pad_h) 
{
    int box_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (box_idx >= num_bbox) 
        return; 

    float x1 = boxes_d[box_idx * 8 + 1];
    float y1 = boxes_d[box_idx * 8 + 2];
    float x2 = boxes_d[box_idx * 8 + 3];
    float y2 = boxes_d[box_idx * 8 + 4];

    x1 -= pad_w; 
    y1 -= pad_h; 

    x1 /= gain;
    y1 /= gain;
    x2 /= gain; 
    y2 /= gain; 

    x1 = clamp_dice(x1, 0.0f, output_w);
    y1 = clamp_dice(y1, 0.0f, output_h);
    x2 = clamp_dice(x2, 0.0f, output_w);
    y2 = clamp_dice(y2, 0.0f, output_h);

    boxes_d[box_idx * 8 + 1] = x1;
    boxes_d[box_idx * 8 + 2] = y1;
    boxes_d[box_idx * 8 + 3] = x2;
    boxes_d[box_idx * 8 + 4] = y2;
}

void cuda_scale_boxes(float* boxes_d, int num_bboxes, float output_w, float output_h, float gain, float pad_w, float pad_h)
{
    auto grid = grid_dims(num_bboxes);
	auto block = block_dims(num_bboxes);
    scale_boxes_kernel<<<grid, block>>>(boxes_d, num_bboxes, output_w, output_h, gain, pad_w, pad_h);
}
