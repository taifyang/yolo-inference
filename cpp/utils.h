#pragma once


#include "yolo.h"


void LetterBox(cv::Mat& input_image, cv::Mat& output_image, cv::Size shape = cv::Size(640, 640), cv::Scalar color = cv::Scalar(114, 114, 114));

void nms(std::vector<cv::Rect>& boxes, std::vector<float>& scores, float score_threshold, float nms_threshold, std::vector<int>& indices);

void scale_box(cv::Rect& box, cv::Size size);

void draw_result(cv::Mat& image, std::string label, cv::Rect box);	

//https://github.com/Tencent/ncnn/blob/master/src/mat.cpp
uint16_t float32_to_float16(float value);

//https://github.com/Tencent/ncnn/blob/master/src/mat.cpp
float float16_to_float32(uint16_t value);
