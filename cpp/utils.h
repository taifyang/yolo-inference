/*
 * @Author: taifyang 58515915+taifyang@users.noreply.github.com
 * @Date: 2024-06-12 09:26:41
 * @LastEditors: taifyang 58515915+taifyang@users.noreply.github.com
 * @LastEditTime: 2024-06-17 21:32:15
 * @FilePath: \cpp\utils.h
 * @Description: 功能函数头文件
 */

#pragma once

#include <iostream>

/**
 * @description:   32位浮点数转16位浮点数，见https://github.com/Tencent/ncnn/blob/master/src/mat.cpp
 * @return {*}     16位浮点数 
 */
uint16_t float32_to_float16(float value);

/**
 * @description:   16位浮点数转32位浮点数，见https://github.com/Tencent/ncnn/blob/master/src/mat.cpp
 * @return {*}     32位浮点数 
 */
float float16_to_float32(uint16_t value);