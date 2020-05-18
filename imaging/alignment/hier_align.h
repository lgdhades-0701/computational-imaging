#pragma once

#include <opencv2/opencv.hpp>

#include "imaging/alignment/block_aligner.h"
#include "imaging/SuperResPipeline.h"

namespace imaging_cpu {

// Pick the 3 input coords to check for a given expanded coord
template <int ExpansionFactor>
void get_coords_to_check(
    int input_width,
    int input_height,
    int expanded_x,
    int expanded_y,
    cv::Vec2i out_coords[3]
);

template <int TileExpansionFactor, int PixelExpansionFactor>
void transfer_displacements(
    const cv::Mat &input,
    cv::Mat *output,
    const align::BlockAligner *aligner
);

// Returns: tile size of alignment (e.g. 16 means each pixel in out_displacement represents
// a tile of size 16x16)
template <typename BlockAlignerT>
int hierarchical_align(
    const cv::Mat & reference,
    const cv::Mat & unaligned,
    cv::Mat *out_displacement
);

AlignedImage compute_alignment(const Inputs &inputs);

}