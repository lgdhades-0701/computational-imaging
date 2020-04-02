#include "stdafx.h"

#include "HierarchicalAlignment.h"
#include "AlignNaive.h"

static const cv::utils::logging::LogTag _TAG("HAlign", cv::utils::logging::LOG_LEVEL_INFO);
static const cv::utils::logging::LogTag *TAG = &_TAG;

namespace imaging_cpu {

#define CHECK_DIM(var_name, alignment) \
  if (var_name % alignment != 0) { \
    CV_LOG_INFO(TAG, #var_name << " " << var_name << " not multiple of " << alignment << "; edge pixels may not be covered"); \
  }

int hierarchical_align(const cv::Mat & reference, const cv::Mat & unaligned, cv::Mat * out_displacement)
{
    // Check input sizes
    if (reference.size != unaligned.size) {
        throw std::runtime_error("Matrix sizes not identical");
    }

    CHECK_DIM(reference.rows, 32);
    CHECK_DIM(reference.cols, 32);

    // Create alignment pyramid
    cv::Mat reference_l2, unaligned_l2,
        reference_l3, unaligned_l3,
        reference_l4, unaligned_l4;
    auto size_l2 = cv::Size(reference.cols / 2, reference.rows / 2);
    auto size_l3 = cv::Size(size_l2.width / 4, size_l2.height / 4);
    auto size_l4 = cv::Size(size_l3.width / 4, size_l3.height / 4); // Total downscale: by 32
    cv::resize(reference, reference_l2, size_l2);
    cv::resize(reference_l2, reference_l3, size_l3);
    cv::resize(reference_l3, reference_l4, size_l4);

    cv::resize(unaligned, unaligned_l2, size_l2);
    cv::resize(unaligned_l2, unaligned_l3, size_l3);
    cv::resize(unaligned_l3, unaligned_l4, size_l4);

    // Align at top level of pyramid
    cv::Mat disp_l3;
    align::block_align_naive<align::align_one_naive<2>>(8 /* tile_size */, 4 /* search_radius */, reference_l3, unaligned_l3, &disp_l3);

    out_displacement->create(reference.size(), CV_32FC2);
    return 0;
}

}