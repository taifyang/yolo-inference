/* 
 * @Author: taifyang
 * @Date: 2026-08-05 20:53:59
 * @LastEditTime: 2026-08-29 16:00:35
 * @Description: semantic segmentation algorithm class
 */

#pragma once

#include "yolo_detect.h"

/**
 * @description: segmentation class for YOLO algorithm
 */
class YOLO_Semantic : virtual public YOLO_Detect
{
public:
	void init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path)
	{
		m_input_size = cv::Size(1024, 1024);
		m_input_numel = 1 * 3 * m_input_size.width * m_input_size.height;
		m_output_numdet = 1 * 1 * m_input_size.width * m_input_size.height;
	}

protected:
	/**
	 * @description: draw result
	 * @return {*}
	 */	
	void draw_result()
	{
        m_result = m_image.clone();
        srand(time(0));

        std::vector<std::pair<int, cv::Mat>> masks;
        std::vector<cv::Scalar> colors;
        for(uint8_t c = 0; c < m_class_num; c++)
        {
            cv::Mat bin;
            cv::compare(m_mask, c, bin, cv::CMP_EQ);
            if(cv::countNonZero(bin)) masks.emplace_back(c, std::move(bin));
            colors.emplace_back(rand() % 256, rand() % 256, rand() % 256);
        }
        
        for(auto& p : masks)
        {
            int cls = p.first;
            auto& bin_mask = p.second;
            cv::Scalar color = colors[cls % colors.size()];
            m_result.setTo(color, bin_mask);
        }
        cv::addWeighted(m_result, 0.5, m_image, 0.5, 0, m_result);
	}

    /**
	 * @description: class num 
	 */	
	int m_class_num = 19;

    /**
	 * @description: model output on host
	 */
	uint8_t* m_output0_host;

    /**
	 * @description: output mask
	 */
    cv::Mat m_mask;
};
