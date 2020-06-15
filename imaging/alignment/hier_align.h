#pragma once

#include <opencv2/opencv.hpp>

#include "imaging/alignment/block_aligner.h"
#include "imaging/SuperResPipeline.h"

namespace imaging_cpu {

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

namespace detail {

// For displacement upscaling, pick the 3 input coords to check for a given expanded coord
void get_coords_to_check(
    cv::Size source_size,
    int expansion_factor,
    cv::Vec2i dest_coord,
    cv::Vec2i out_coords[3]
);

struct AlignLevel {
    int downscale;
    int tile_size;
    int search_radius;

    cv::Mat_<float> ref, alt;
    cv::Mat_<cv::Vec2f> initial_disp;
    cv::Mat_<cv::Vec2f> ending_disp;

    AlignLevel(const cv::Mat_<float> &input_ref,
        const cv::Mat_<float> &input_alt,
        int downscale,
        int tile_size,
        int search_radius,
        bool search_subpixel)
        : downscale(downscale),
        tile_size(tile_size),
        search_radius(search_radius)
    {
        // TODO(fyhuang): support unpadded images
        assert(input_ref.rows % downscale == 0);
        assert(input_ref.cols % downscale == 0);

        cv::Size size_after_scale(input_ref.rows / downscale, input_ref.cols / downscale);
        cv::resize(input_ref, ref, size_after_scale);
        cv::resize(input_alt, alt, size_after_scale);

        // TODO(fyhuang): support "jagged" border
        assert(size_after_scale.width % tile_size == 0);
        assert(size_after_scale.height % tile_size == 0);
        cv::Size disp_size(size_after_scale.width / tile_size, size_after_scale.height / tile_size);
        initial_disp.create(disp_size);
        initial_disp = cv::Vec2f(0.0f);

        ending_disp.create(disp_size);
        ending_disp = cv::Vec2f(0.0f);
    }

    void seed_initial_displacements(const AlignLevel &prev_level);
};

}

}