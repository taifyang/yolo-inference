/*
 * @Author: taifyang 
 * @Date: 2024-06-12 09:26:41
 * @LastEditTime: 2026-01-03 20:39:23
 * @Description: source file for cuda post-processing decoding
 */

#include "decode.cuh"

dim3 grid_dims(int numJobs) 
{
	int numBlockThreads = numJobs < GPU_BLOCK_THREADS ? numJobs : GPU_BLOCK_THREADS;
	return dim3(((numJobs + numBlockThreads - 1) / (float)numBlockThreads));
}

dim3 block_dims(int numJobs) 
{
	return numJobs < GPU_BLOCK_THREADS ? numJobs : GPU_BLOCK_THREADS;
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

void decode_kernel_invoker(float* predict, int num_bboxes, int num_classes, float confidence_threshold, float score_threshold, float* inverse_affine_matrix, float* parray, int max_objects, int num_box_element, cv::Size input_size, cudaStream_t stream, Algo_Type algo_type, Task_Type task_type)
{
	auto grid = grid_dims(num_bboxes);
	auto block = block_dims(num_bboxes);
	decode_kernel << <grid, block, 0, stream >> > (predict, num_bboxes, num_classes, confidence_threshold, score_threshold, inverse_affine_matrix, parray, max_objects, num_box_element, input_size, algo_type, task_type);
}


void nms_kernel_invoker(float* parray, float nms_threshold, int max_objects, int num_box_element, cudaStream_t stream)
{
	auto grid = grid_dims(max_objects);
	auto block = block_dims(max_objects);
	nms_kernel << <grid, block, 0, stream >> > (parray, max_objects, nms_threshold, num_box_element);
}

 static __global__ void decode_single_mask_kernel(int left, int top, float* mask_weights, float* mask_predict, int mask_width, int mask_height, 
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

void decode_single_mask(float left, float top, float* mask_weights, float* mask_predict,
						int mask_width, int mask_height, unsigned char *mask_out,
                        int mask_dim, int out_width, int out_height, cudaStream_t stream)
{
	// mask_weights is mask_dim(32 element) gpu pointer
	dim3 grid((out_width + 31) / 32, (out_height + 31) / 32);
	dim3 block(32, 32);
	decode_single_mask_kernel<<<grid, block, 0, stream>>>(left, top, mask_weights, mask_predict, mask_width, mask_height, mask_out, mask_dim, out_width, out_height);
}

__global__ void resize_linear_kernel(const uchar* src, uchar* dst,
                                     int src_step, int dst_step,
                                     int src_h, int src_w,
                                     int dst_h, int dst_w,
                                     float scale_h, float scale_w) 
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

void resize_cuda(uchar* src, uchar* dst, cv::Size src_size, cv::Size dst_size, cudaStream_t stream) 
{
    float scale_h = static_cast<float>(dst_size.height) / src_size.height;
    float scale_w = static_cast<float>(dst_size.width) / src_size.width;

    int src_step = src_size.width;  
    int dst_step = dst_size.width;

    const dim3 block_size(32, 32);  
    const dim3 grid_size((dst_size.height + block_size.x - 1) / block_size.x,(dst_size.width + block_size.y - 1) / block_size.y);

    resize_linear_kernel<<<grid_size, block_size, 0, stream>>>(
        src, dst,
        src_step, dst_step,
        src_size.height, src_size.width,
        dst_size.height, dst_size.width,
        scale_h, scale_w
    );
}
