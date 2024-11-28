/*
 * @Author: taifyang 
 * @Date: 2024-06-12 09:26:41
 * @LastEditors: taifyang 58515915+taifyang@users.noreply.github.com
 * @LastEditTime: 2024-06-17 21:32:15
 * @FilePath: \cpp\utils.h
 * @Description: utilities head file
 */

#pragma once

#include <iostream>

/**
 * @description:   float32 to float16, refer to https://github.com/Tencent/ncnn/blob/master/src/mat.cpp
 * @return {*}     float16 
 */
uint16_t float32_to_float16(float value);

/**
 * @description:   float16 to float32, refer to https://github.com/Tencent/ncnn/blob/master/src/mat.cpp
 * @return {*}     float32 
 */
float float16_to_float32(uint16_t value);