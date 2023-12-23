#pragma once


#include "yolov5.h"


//LetterBox处理
void LetterBox(cv::Mat& input_image, cv::Mat& output_image, cv::Size& shape = cv::Size(640, 640), cv::Scalar& color = cv::Scalar(114, 114, 114));

//NMS
void nms(std::vector<cv::Rect>& boxes, std::vector<float>& scores, float score_threshold, float nms_threshold, std::vector<int>& indices);

//box缩放到原图尺寸
void scale_box(cv::Rect& box, cv::Size size);

//可视化函数
void draw_result(cv::Mat& image, std::string label, cv::Rect box);	

//float32转float16
uint16_t float32_to_float16(float value);

//float16转float32
float float16_to_float32(uint16_t value);
