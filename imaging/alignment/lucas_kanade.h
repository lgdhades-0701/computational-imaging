#pragma once

#include <opencv2/opencv.hpp>

namespace imaging {
namespace alignment {

struct LucasKanadeRefiner {
    const cv::Mat_<float> &reference, &alternate;
    cv::Mat_<float> ref_norm, alt_norm;
    cv::Mat_<float> ref_blur, alt_blur;
    cv::Mat_<float> ref_gradient_x, ref_gradient_y;
    cv::Mat_<float> alt_gradient_x, alt_gradient_y;

public:
    LucasKanadeRefiner(const cv::Mat_<float> &reference, const cv::Mat_<float> &alternate);

    cv::Vec2f refine(cv::Rect ref_block, cv::Vec2f initial_disp, int num_iterations);

    void visualize(cv::Rect ref_block, cv::Vec2f displacement, const std::string &out_prefix = "");
};


namespace detail {

struct LKRefineIter {
    constexpr static float LK_INVERTIBLE_THRESHOLD = 0.01f;

    const LucasKanadeRefiner *parent;
    cv::Rect ref_block;

    cv::Mat_<float> alt_grad_x_patch, alt_grad_y_patch, alt_img_patch;
    cv::Mat_<float> ata, atb;
    float ata_det;

public:
    LKRefineIter(const LucasKanadeRefiner *parent, cv::Rect ref_block, cv::Vec2f initial_disp);

    void compute_matrices_cpp();
    void compute_matrices_eigen();

    bool can_perform_lk_cpp();
    bool can_perform_lk_eigen();

    cv::Vec2f compute_disp_cpp();
    cv::Vec2f compute_disp_eigen();
};

}
}
}