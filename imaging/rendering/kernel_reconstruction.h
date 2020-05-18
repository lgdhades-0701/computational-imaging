#pragma once

namespace imaging_cpu {

static void merge_images_kernel(
    const Inputs &inputs,
    const std::vector<AlignedImage> &aligned_images,
    cv::Mat *output,
) {
    cv::Mat pixel_weights(reference.cols, reference.rows, CV_32FC1, cv::Scalar(0));

    for (int y = 0; y < inputs.reference->rows; y++) {
        for (int x = 0; x < inputs.reference->cols; x++) {
            float weight = 1.0f;
            float color_sum = reference.at<uint8_t>(y, x);

            for (const AlignedImage &aligned : aligned_images) {
                
            }
        }
    }
}

}