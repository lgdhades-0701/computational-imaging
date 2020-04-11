#pragma once

#include "SuperResPipeline.h"

namespace imaging_cpu {

constexpr float PARAM_S = 5.0f;
constexpr float PARAM_T = 0.2f;

static void calculate_robustness_mask(
    const cv::Mat &reference,
    const cv::Mat &aligned,
    cv::Mat *robustness_mask
) {
    robustness_mask->create(reference.rows, reference.cols, CV_32FC1);
    *robustness_mask = cv::Scalar(0.f);

    for (int y = 1; y < reference.rows - 1; y++) {
        for (int x = 1; x < reference.cols - 1; x++) {
            // Calculate local mean and std dev from the reference
            float local_sum = 0.0f;
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    local_sum += reference.at<uint8_t>(y + dy, x + dx);
                }
            }
            float local_mean = local_sum / 9.0f;

            float local_variance = 0.0f;
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    float pixel_val = reference.at<uint8_t>(y + dy, x + dx);
                    float diff_squared = (pixel_val - local_mean) * (pixel_val - local_mean);
                    local_variance += diff_squared / 9.0f;
                }
            }
            float local_stddev = sqrt(local_variance);
            local_stddev = std::max(local_stddev, 0.1f); // Since we divide by stddev^2 later

            // Calculate pixel value difference by comparing to aligned image
            float color_difference = (float)aligned.at<uint8_t>(y, x) - reference.at<uint8_t>(y, x);

            // TODO: refine robustness from motion
            float raw_robustness = PARAM_S * exp(-1.0f * (color_difference * color_difference) / (local_stddev * local_stddev)) - PARAM_T;
            robustness_mask->at<float>(y, x) = std::max(std::min(raw_robustness, 1.0f), 0.0f);
        }
    }
}

}