/*
 * @Author: taifyang 
 * @Date: 2024-06-12 09:26:41
 * @LastEditTime: 2026-01-12 15:15:02
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
	if(algo_type == YOLOv3 || algo_type == YOLOv4 || algo_type == YOLOv6 || algo_type == YOLOv8 || algo_type == YOLOv9 || algo_type == YOLOv10 || algo_type == YOLOv11 || algo_type == YOLOv12 || algo_type == YOLOv13)
	{
		if(task_type == Detect)
		{
			pitem = predict + (4 + num_classes) * position;
		}
		else if(task_type == Segment)
		{
			pitem = predict + (36 + num_classes) * position;
		}
        else if(task_type == Pose)
        {
            pitem = predict + 56 * position;
        }
		else if(task_type == OBB)
		{
			pitem = predict + (5 + num_classes) * position;
			angle = pitem[4 + num_classes];
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
	for (int i = 1; i < num_classes; ++i, ++class_score)
	{
		if (*class_score > score && algo_type != Pose)
		{
			score = *class_score;
			label = i;
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
		else if (algo_type == YOLOv10)
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

void cuda_decode(float* predict, int num_bboxes, int num_classes, float confidence_threshold, float score_threshold, float* inverse_affine_matrix, float* parray, int max_objects, int num_box_element, cv::Size input_size, cudaStream_t stream, Algo_Type algo_type, Task_Type task_type)
{
	auto grid = grid_dims(num_bboxes);
	auto block = block_dims(num_bboxes);
	decode_kernel << <grid, block, 0, stream >> > (predict, num_bboxes, num_classes, confidence_threshold, score_threshold, inverse_affine_matrix, parray, max_objects, num_box_element, input_size, algo_type, task_type);
}

void cuda_nms(float* parray, float nms_threshold, int max_objects, int num_box_element, cudaStream_t stream)
{
	auto grid = grid_dims(max_objects);
	auto block = block_dims(max_objects);
	nms_kernel << <grid, block, 0, stream >> > (parray, max_objects, nms_threshold, num_box_element);
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

void cuda_decode_mask(float left, float top, float* mask_weights, float* mask_predict, int mask_width, int mask_height, uchar* mask_out,
                    int mask_dim, int out_width, int out_height, cudaStream_t stream)
{
	// mask_weights is mask_dim(32 element) gpu pointer
	dim3 grid((out_width + 31) / 32, (out_height + 31) / 32);
	dim3 block(32, 32);
	cuda_decode_mask_kernel<<<grid, block, 0, stream>>>(left, top, mask_weights, mask_predict, mask_width, mask_height, mask_out, mask_dim, out_width, out_height);
}

__global__ void resize_linear_kernel(const uchar* src, uchar* dst, int src_step, int dst_step,
                                     int src_h, int src_w, int dst_h, int dst_w, float scale_h, float scale_w) 
{
    int dst_y = blockIdx.x * blockDim.x + threadIdx.x;
    int dst_x = blockIdx.y * blockDim.y + threadIdx.y;
    if (dst_y >= dst_h || dst_x >= dst_w) 
        return;
    int src_y = static_cast<int>(dst_y / scale_h);
    int src_x = static_cast<int>(dst_x / scale_w);
    src_y = min(max(src_y, 0), src_h - 1);
    src_x = min(max(src_x, 0), src_w - 1);
    dst[dst_y * dst_step + dst_x] = src[src_y * src_step + src_x];
}

void cuda_resize(uchar* src, uchar* dst, cv::Size src_size, cv::Size dst_size, cudaStream_t stream) 
{
    float scale_h = static_cast<float>(dst_size.height) / src_size.height;
    float scale_w = static_cast<float>(dst_size.width) / src_size.width;

    int src_step = src_size.width;  
    int dst_step = dst_size.width;

    const dim3 block_size(32, 32);  
    const dim3 grid_size((dst_size.height + block_size.x - 1) / block_size.x,(dst_size.width + block_size.y - 1) / block_size.y);

    resize_linear_kernel<<<grid_size, block_size, 0, stream>>>(src, dst, src_step, dst_step, src_size.height, src_size.width, dst_size.height, dst_size.width, scale_h, scale_w);
}

__global__ void extract_colKernel(const float* input_d, float* output_d, int target_col, int rows, int cols) 
{
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    if (row < rows) 
        output_d[row] = input_d[row * cols + target_col];
}

void cuda_extract_col(const float* input_d, float* output_d, int target_col, int rows, int cols, cudaStream_t stream) 
{       
    auto grid = grid_dims(rows);
	auto block = block_dims(rows);  
    extract_colKernel<<<grid, block, 0, stream>>>(input_d, output_d, target_col, rows, cols);
}

void thrust_argsort(float* scores_d, int* sorted_idx_d, int num_bbox)
{
    thrust::device_ptr<float> dev_ptr_scores(scores_d);
    thrust::device_ptr<int> dev_ptr_indices(sorted_idx_d);
    thrust::sequence(thrust::device,  dev_ptr_indices,  dev_ptr_indices + num_bbox);
    thrust::sort_by_key(thrust::device, dev_ptr_scores, dev_ptr_scores + num_bbox, dev_ptr_indices, thrust::greater<float>());
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

void cuda_pick_bbox(const float* ious_d, int* pick_d, float threshold, int num_bboxes, int& pick_size, cudaStream_t stream) 
{
    float* col_max_d = nullptr;
    int* mask_d = nullptr;
    int* prefix_sum_d = nullptr;

    cudaMalloc((void**)&col_max_d, num_bboxes * sizeof(float));
    cudaMalloc((void**)&mask_d, num_bboxes * sizeof(int));
    cudaMalloc((void**)&prefix_sum_d, num_bboxes * sizeof(int)); 
    
    auto grid = grid_dims(num_bboxes);
	auto block = block_dims(num_bboxes);  

    columnMaxKernel<<<num_bboxes, block, 0, stream>>>(ious_d, col_max_d, num_bboxes);

    thresholdMaskKernel<<<grid, block, 0, stream>>>(col_max_d, mask_d, threshold, num_bboxes);

    prefixSumKernel<<<grid, block, 0, stream>>>(mask_d, prefix_sum_d, num_bboxes);

    cudaMemcpy(&pick_size, &prefix_sum_d[num_bboxes - 1], sizeof(int), cudaMemcpyDeviceToHost);

    collectIndicesKernel<<<grid, block, 0, stream>>>(mask_d, prefix_sum_d, pick_d, num_bboxes);
}


__global__ void extract_rowsKernel(const float* input_d, const int* index_d, float* output_d, int num_bbox) 
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_bbox) {
        int input_row = index_d[tid];
        for (int j = 0; j < 8; ++j) 
            output_d[tid * 8 + j] = input_d[input_row * 8 + j];
    }
}

void cuda_extract_rows(const float* input_d, const int* index_d, float* output_d, int num_bboxes, cudaStream_t stream) 
{   
    auto grid = grid_dims(num_bboxes);
	auto block = block_dims(num_bboxes);                  
    extract_rowsKernel<<<grid, block, 0, stream>>>(input_d, index_d, output_d, num_bboxes);
}


__global__ void compute_covariance_matrix_kernel(const float* boxes,  float* out1,  float* out2,  float* out3, int num_bbox) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_bbox)
        return;
    
    float w = boxes[i * 8 + 3];
    float h = boxes[i * 8 + 4];
    float theta = boxes[i * 8 + 7];

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

void cuda_compute_covariance_matrix(const float* boxes_d, float* a_d, float* b_d, float* c_d, int num_bboxes, cudaStream_t stream)
 {
    auto grid = grid_dims(num_bboxes);
	auto block = block_dims(num_bboxes);  
    compute_covariance_matrix_kernel<<<grid, block, 0, stream>>>(boxes_d, a_d, b_d, c_d, num_bboxes);
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
    float* hd_d, int num_bbox, cudaStream_t stream) 
{
    dim3 block_dim(32, 32);
    dim3 grid_dim((num_bbox + block_dim.x - 1) / block_dim.x, (num_bbox + block_dim.y - 1) / block_dim.y);
    compute_obb_pairwise_hd_kernel<<<grid_dim, block_dim, 0, stream>>>(obb1_d, obb2_d, a1_d, b1_d, c1_d, a2_d, b2_d, c2_d, hd_d, num_bbox);
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

void cuda_triu_k1(float* d_mat, int rows, int cols, cudaStream_t stream) 
{
    dim3 block(32, 32); 
    dim3 grid((rows + block.x - 1) / block.x, (cols + block.y - 1) / block.y);
    triu_k1_kernel<<<grid, block, 0, stream>>>(d_mat, rows, cols);
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

void cuda_regularize_bbox(float* rboxes_d, int num_bboxes, cudaStream_t stream)
{
    auto grid = grid_dims(num_bboxes);
	auto block = block_dims(num_bboxes);
    regularize_bboxKernel<<<grid, block, 0, stream>>>(rboxes_d, num_bboxes);
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

void cuda_scale_boxes(float* boxes_d, int num_bboxes, float output_w, float output_h, float gain, float pad_w, float pad_h, cudaStream_t stream)
{
    auto grid = grid_dims(num_bboxes);
	auto block = block_dims(num_bboxes);
    scale_boxes_kernel<<<grid, block, 0, stream>>>(boxes_d, num_bboxes, output_w, output_h, gain, pad_w, pad_h);
}
