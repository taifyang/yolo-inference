/*
 * @Author: taifyang 
 * @Date: 2024-06-12 09:26:41
 * @LastEditors: taifyang 
 * @LastEditTime: 2024-06-17 21:34:25
 * @FilePath: \cpp\utils.cpp
 * @Description: 功能函数实现
 */

#include "utils.h"

uint16_t float32_to_float16(float value)
{
	// 1 : 8 : 23
	union
	{
		unsigned int u;
		float f;
	} tmp;

	tmp.f = value;

	// 1 : 8 : 23
	uint16_t sign = (tmp.u & 0x80000000) >> 31;
	uint16_t exponent = (tmp.u & 0x7F800000) >> 23;
	unsigned int significand = tmp.u & 0x7FFFFF;

	//     NCNN_LOGE("%d %d %d", sign, exponent, significand);

	// 1 : 5 : 10
	uint16_t fp16;
	if (exponent == 0)
	{
		// zero or denormal, always underflow
		fp16 = (sign << 15) | (0x00 << 10) | 0x00;
	}
	else if (exponent == 0xFF)
	{
		// infinity or NaN
		fp16 = (sign << 15) | (0x1F << 10) | (significand ? 0x200 : 0x00);
	}
	else
	{
		// normalized
		short newexp = exponent + (-127 + 15);
		if (newexp >= 31)
		{
			// overflow, return infinity
			fp16 = (sign << 15) | (0x1F << 10) | 0x00;
		}
		else if (newexp <= 0)
		{
			// Some normal fp32 cannot be expressed as normal fp16
			fp16 = (sign << 15) | (0x00 << 10) | 0x00;
		}
		else
		{
			// normal fp16
			fp16 = (sign << 15) | (newexp << 10) | (significand >> 13);
		}
	}

	return fp16;
}

float float16_to_float32(uint16_t value)
{
	// 1 : 5 : 10
	uint16_t sign = (value & 0x8000) >> 15;
	uint16_t exponent = (value & 0x7c00) >> 10;
	uint16_t significand = value & 0x03FF;

	//     NCNN_LOGE("%d %d %d", sign, exponent, significand);

	// 1 : 8 : 23
	union
	{
		unsigned int u;
		float f;
	} tmp;
	if (exponent == 0)
	{
		if (significand == 0)
		{
			// zero
			tmp.u = (sign << 31);
		}
		else
		{
			// denormal
			exponent = 0;
			// find non-zero bit
			while ((significand & 0x200) == 0)
			{
				significand <<= 1;
				exponent++;
			}
			significand <<= 1;
			significand &= 0x3FF;
			tmp.u = (sign << 31) | ((-exponent + (-15 + 127)) << 23) | (significand << 13);
		}
	}
	else if (exponent == 0x1F)
	{
		// infinity or NaN
		tmp.u = (sign << 31) | (0xFF << 23) | (significand << 13);
	}
	else
	{
		// normalized
		tmp.u = (sign << 31) | ((exponent + (-15 + 127)) << 23) | (significand << 13);
	}

	return tmp.f;
}