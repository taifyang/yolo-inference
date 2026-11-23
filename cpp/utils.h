/* 
 * @Author: taifyang
 * @Date: 2026-08-15 10:43:42
 * @LastEditTime: 2026-11-23 00:10:38
 * @Description: 
 */
/*
 * @Author: taifyang 
 * @Date: 2024-06-12 09:26:41
 * @LastEditTime: 2026-11-23 00:08:37
 * @Description: utilities head file
 */

#pragma once

#include <iostream>
#include <cstdint>

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

/**
 * @description: Perform affine transformation projection
 * @param matrix: 2x3 affine transformation matrix
 * @param x: input source x coordinate
 * @param y: input source y coordinate
 * @param ox: output transformed target x coordinate
 * @param oy: output transformed target y coordinate
 * @return {*}
 */
void affine_project(float* matrix, float x, float y, float* ox, float* oy);