#pragma once

namespace imaging_cpu {

// Returns: tile size of alignment (e.g. 16 means each pixel in out_displacement represents
// a tile of size 16x16)
int align(const cv::Mat &reference, const cv::Mat &unaligned, cv::Mat *out_displacement);

}