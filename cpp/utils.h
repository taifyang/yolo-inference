#pragma once

#include <iostream>

//https://github.com/Tencent/ncnn/blob/master/src/mat.cpp
uint16_t float32_to_float16(float value);

//https://github.com/Tencent/ncnn/blob/master/src/mat.cpp
float float16_to_float32(uint16_t value);