#pragma once

namespace imaging_cpu {

class Inputs {
public:
    const cv::Mat *reference;
    const cv::Mat *alternate;

    Inputs(const cv::Mat *reference, const cv::Mat *alternate)
        : reference(reference),
        alternate(alternate)
    {}
};

class AlignedImage {
public:
    int displacement_tile_size;
    std::unique_ptr<cv::Mat> displacements;
    std::unique_ptr<cv::Mat> aligned;

    AlignedImage(int tile_size, std::unique_ptr<cv::Mat> &&displacements, std::unique_ptr<cv::Mat> &&aligned)
        : displacement_tile_size(tile_size),
        displacements(std::move(displacements)),
        aligned(std::move(aligned))
    {}
};

}