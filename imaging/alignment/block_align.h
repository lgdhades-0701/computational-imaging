#pragma once

#include <opencv2/opencv.hpp>

#include "imaging/Util.h"
#include "imaging/alignment/block_aligner.h"

namespace imaging {
namespace alignment {

template <int Norm, typename T>
float calculate_residual(
    const cv::Mat_<T> &a,
    const cv::Mat_<T> &b
) {
    static_assert(Norm == 1 || Norm == 2, "Only L1/L2 norm supported");
    assert(a.size() == b.size());

    float distance = 0.0f;
    for (int y = 0; y < a.rows; y++) {
        for (int x = 0; x < a.cols; x++) {
            float d_pixel = (float)a(y, x) - b(y, x);
#pragma warning(push)
#pragma warning(disable : 4127)
            // Conditional expression is supposed to be constant
            if (Norm == 1) {
                d_pixel = std::fabsf(d_pixel);
            }
            else if (Norm == 2) {
                d_pixel *= d_pixel;
            }
#pragma warning(pop)
            distance += d_pixel;
        }
    }
    return distance;
}

template <int Norm, typename T>
cv::Vec2i align_one_block(
    const cv::Mat_<T> &reference_tile,
    const cv::Mat_<T> &unaligned_neighborhood,
    cv::Vec2i disp_correction = cv::Vec2i(0, 0)
) {
    // Return the displacement that minimizes the residual. Displacement is how many pixels the reference tile
    // must be moved to minimize residual with the alternate image.
    struct BestState { float residual; cv::Vec2i corrected_disp; float disp_length; };
    auto best = BestState{ std::numeric_limits<float>::infinity() };

    for (int disp_y = 0; disp_y < unaligned_neighborhood.rows - reference_tile.rows + 1; disp_y++) {
        for (int disp_x = 0; disp_x < unaligned_neighborhood.cols - reference_tile.cols + 1; disp_x++) {

            // Compute residual for this (disp_x, disp_y)
            float residual = calculate_residual<Norm>(
                reference_tile,
                unaligned_neighborhood(cv::Rect(disp_x, disp_y, reference_tile.cols, reference_tile.rows))
            );

            // See if this is the best one so far
            cv::Vec2i corrected_disp = cv::Vec2i(disp_x, disp_y) + disp_correction;
            float this_length = (float)cv::norm(corrected_disp);
            if (residual < best.residual) {
                best = BestState{ residual, corrected_disp, this_length };
            }
            else if (residual == best.residual) {
                // Pick the smaller displacement in case of tie
                if (this_length < best.disp_length) {
                    best = BestState{ residual, corrected_disp, this_length };
                }
            }
        }
    }

    return best.corrected_disp;
}

struct BlockAligner {
    virtual ~BlockAligner() {}
    //virtual int tile_size() const = 0;
    virtual float disp_residual_L1(cv::Vec2i tile_coord, cv::Vec2f disp) const = 0;
    virtual void align(const cv::Mat_<cv::Vec2f> &in_disp, cv::Mat_<cv::Vec2f> *out_disp) = 0;
};

struct CpuBlockAligner : public BlockAligner {
    int align_norm_;
    int tile_size_, search_radius_;
    const cv::Mat_<float> &ref_, &alt_;

public:
    CpuBlockAligner(int align_norm, int tile_size, int search_radius, const cv::Mat_<float> &ref, const cv::Mat_<float> &alt)
        : align_norm_(align_norm),
        tile_size_(tile_size),
        search_radius_(search_radius),
        ref_(ref),
        alt_(alt)
    {
        assert(align_norm == 1 || align_norm == 2);
    }

    float disp_residual_L1(cv::Vec2i tile_coord, cv::Vec2f disp) const override;
    void align(const cv::Mat_<cv::Vec2f> &in_disp, cv::Mat_<cv::Vec2f> *out_disp) override;
};

namespace detail {

void block_align_images(
    int align_norm,
    int tile_size,
    int search_radius,
    const cv::Mat_<float> &reference,
    const cv::Mat_<float> &unaligned,
    const cv::Mat_<cv::Vec2f> &in_displacements,
    cv::Mat_<cv::Vec2f> *out_displacements
);

}

}
}