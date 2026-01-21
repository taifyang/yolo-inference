/*
 * @Author: taifyang 
 * @Date: 2024-06-12 09:26:41
 * @LastEditTime: 2026-01-19 18:38:38
 * @Description: source file for cuda pre-processing decoding
 */

#include "preprocess.cuh"

__global__ void crop_kernel(uint8_t* src, uint8_t* dst, int src_w, int src_h, int crop_size, int left, int top, int channel)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= crop_size && y >= crop_size)
        return;

    int src_idx = ((top + y) * src_w + (left + x)) * channel;
    int dst_idx = (y * crop_size + x) * channel;
    for(int c=0; c<channel; ++c)
    {
        dst[dst_idx + c] = src[src_idx + c];  
    }
}

void cuda_crop(uint8_t* src, uint8_t* dst, cv::Size src_size, int crop_size, int left, int top, int channel) 
{
    dim3 block(32, 32); 
    dim3 grid((crop_size + block.x - 1) / block.x, (crop_size + block.y - 1) / block.y); 
    crop_kernel<<<grid, block>>>(src, dst, src_size.width, src_size.height, crop_size, left, top, channel);
}

__global__ void resize_linear_kernel(uint8_t* src, uint8_t* dst, int src_w, int src_h, int dst_w, int dst_h, int channel)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dst_w || y >= dst_h) 
        return;

    float src_x = (x)* (static_cast<float>(src_w-1) ) / (static_cast<float>(dst_w-1));
    float src_y = (y )* (static_cast<float>(src_h-1) ) / (static_cast<float>(dst_h-1)) ;

    int x0 = static_cast<int>(floor(src_x));
    int x1 = min(x0 + 1, src_w - 1);
    int y0 = static_cast<int>(floor(src_y));
    int y1 = min(y0 + 1, src_h - 1);

    float wx = src_x - x0;
    float wy = src_y - y0;

    int idx00 = (y0 * src_w + x0) * channel;
    int idx01 = (y0 * src_w + x1) * channel;
    int idx10 = (y1 * src_w + x0) * channel;
    int idx11 = (y1 * src_w + x1) * channel;

    for (int c = 0; c < channel; c++) 
    {
        float p00 = static_cast<float>(src[idx00 + c]);
        float p01 = static_cast<float>(src[idx01 + c]);
        float p10 = static_cast<float>(src[idx10 + c]);
        float p11 = static_cast<float>(src[idx11 + c]);
        float interpolated = (1 - wx) * (1 - wy) * p00 + wx * (1 - wy) * p01 + (1 - wx) * wy * p10 + wx * wy * p11;
        dst[(y * dst_w + x) * channel + c] = static_cast<uint8_t>(interpolated);
    }
}

void cuda_resize_linear(uint8_t* src, uint8_t* dst, cv::Size src_size, cv::Size dst_size, int channel) 
{
    dim3 block_size(32, 32);  
    dim3 grid_size((dst_size.width + block_size.x - 1) / block_size.x, (dst_size.height + block_size.y - 1) / block_size.y);
    resize_linear_kernel<<<grid_size, block_size>>>(src, dst, src_size.width, src_size.height, dst_size.width, dst_size.height, channel);
}

void cuda_centercrop(uint8_t* src, uint8_t* crop, uint8_t* dst, cv::Size src_size, cv::Size dst_size, int channel)
{
    int crop_size = std::min(src_size.width, src_size.height);
    int left = (src_size.width - crop_size) / 2, top = (src_size.height - crop_size) / 2;
    cuda_crop(src, crop, src_size, crop_size, left, top, channel);
    cuda_resize_linear(crop, dst, cv::Size(crop_size, crop_size), dst_size, channel);
}

__global__ void normalize_kernel(uint8_t* src, float* dst, int width, int height, Algo_Type algo_type) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width && y >= height) 
        return;

    int idx = (y * width + x) * 3;
    float b = static_cast<float>(src[idx]) / 255.0f;  
    float g = static_cast<float>(src[idx + 1]) / 255.0f;
    float r = static_cast<float>(src[idx + 2]) / 255.0f;

    if (algo_type == YOLOv5)
    {
        b = (b - 0.406) / 0.225;
        g = (g - 0.456) / 0.224;
        r = (r - 0.485) / 0.229;
    }

    int chw_base = y * width + x;      
    int chw_size = height * width;    
    dst[0 * chw_size + chw_base] = r;   
    dst[1 * chw_size + chw_base] = g;   
    dst[2 * chw_size + chw_base] = b;  
}

void cuda_normalize(uint8_t* src, float* dst, cv::Size src_size, Algo_Type algo_type)
{
    dim3 block(32, 32);
    dim3 grid((src_size.width + block.x - 1) / block.x, (src_size.height + block.y - 1) / block.y);
    normalize_kernel<<<grid, block>>>(src, dst, src_size.width, src_size.height, algo_type);
}

__global__ void warpaffine_kernel(uint8_t* src, int src_line_size, int src_width, int src_height, 
	float* dst, int dst_width, int dst_height, uint8_t const_value_st, AffineMatrix d2s, int edge) 
{
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= edge) 
		return;

    float m_x1 = d2s.value[0];
    float m_y1 = d2s.value[1];
    float m_z1 = d2s.value[2];
    float m_x2 = d2s.value[3];
    float m_y2 = d2s.value[4];
    float m_z2 = d2s.value[5];

    int dx = position % dst_width;
    int dy = position / dst_width;
    float src_x = m_x1 * dx + m_y1 * dy + m_z1 + 0.5f;
    float src_y = m_x2 * dx + m_y2 * dy + m_z2 + 0.5f;
    float c0, c1, c2;

    if (src_x <= -1 || src_x >= src_width || src_y <= -1 || src_y >= src_height) 
	{
        // out of range
        c0 = const_value_st;
        c1 = const_value_st;
        c2 = const_value_st;
    } 
	else 
	{
        int y_low = floorf(src_y);
        int x_low = floorf(src_x);
        int y_high = y_low + 1;
        int x_high = x_low + 1;

        uint8_t const_value[] = {const_value_st, const_value_st, const_value_st};
        float ly = src_y - y_low;
        float lx = src_x - x_low;
        float hy = 1 - ly;
        float hx = 1 - lx;
        float w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
        uint8_t* v1 = const_value;
        uint8_t* v2 = const_value;
        uint8_t* v3 = const_value;
        uint8_t* v4 = const_value;

        if (y_low >= 0) 
		{
            if (x_low >= 0)
                v1 = src + y_low * src_line_size + x_low * 3;

            if (x_high < src_width)
                v2 = src + y_low * src_line_size + x_high * 3;
        }

        if (y_high < src_height) 
		{
            if (x_low >= 0)
                v3 = src + y_high * src_line_size + x_low * 3;

            if (x_high < src_width)
                v4 = src + y_high * src_line_size + x_high * 3;
        }

        c0 = w1 * v1[0] + w2 * v2[0] + w3 * v3[0] + w4 * v4[0];
        c1 = w1 * v1[1] + w2 * v2[1] + w3 * v3[1] + w4 * v4[1];
        c2 = w1 * v1[2] + w2 * v2[2] + w3 * v3[2] + w4 * v4[2];
    }

    //bgr to rgb 
    float t = c2;
    c2 = c0;
    c0 = t;

    //normalization
    c0 = c0 / 255.0f;
    c1 = c1 / 255.0f;
    c2 = c2 / 255.0f;

    //rgbrgbrgb to rrrgggbbb
    int area = dst_width * dst_height;
    float* pdst_c0 = dst + dy * dst_width + dx;
    float* pdst_c1 = pdst_c0 + area;
    float* pdst_c2 = pdst_c1 + area;
    *pdst_c0 = c0;
    *pdst_c1 = c1;
    *pdst_c2 = c2;
}

void cuda_preprocess_img(uint8_t* src, int src_width, int src_height, float* dst, int dst_width, int dst_height, float* affine_matrix, float* affine_matrix_inverse)
{
    AffineMatrix s2d,d2s;
    float scale = std::min(dst_height / (float)src_height, dst_width / (float)src_width);

    s2d.value[0] = scale;
    s2d.value[1] = 0;
    s2d.value[2] = -scale * src_width  * 0.5  + dst_width * 0.5;
    s2d.value[3] = 0;
    s2d.value[4] = scale;
    s2d.value[5] = -scale * src_height * 0.5 + dst_height * 0.5;

    cv::Mat m2x3_s2d(2, 3, CV_32F, s2d.value);	
    cv::Mat m2x3_d2s(2, 3, CV_32F, d2s.value);
    cv::invertAffineTransform(m2x3_s2d, m2x3_d2s);

	memcpy(affine_matrix, m2x3_d2s.data, 6 * sizeof(affine_matrix));
    memcpy(affine_matrix_inverse, m2x3_s2d.data, 6 * sizeof(affine_matrix_inverse));
    memcpy(d2s.value, m2x3_d2s.ptr<float>(0), sizeof(d2s.value));

    int jobs = dst_height * dst_width;
    int threads = 1024;
    int blocks = ceil(jobs / (float)threads);
    warpaffine_kernel<<<blocks, threads>>>(src, src_width*3, src_width, src_height, dst, dst_width, dst_height, 114, d2s, jobs);
}
