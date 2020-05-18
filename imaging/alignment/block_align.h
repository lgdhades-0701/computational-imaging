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

struct LucasKanadeRefiner {
    constexpr static float LK_INVERTIBLE_THRESHOLD = 0.01f;
    
    const cv::Mat_<float> &reference, &alternate;
    cv::Mat_<float> ref_gradient_x, ref_gradient_y;
    cv::Mat_<float> alt_gradient_x, alt_gradient_y;

    LucasKanadeRefiner(const cv::Mat_<float> &reference, const cv::Mat_<float> &alternate);

    cv::Vec2f refine(cv::Rect ref_block, cv::Vec2f initial_disp, int num_iterations);
};

}
}

namespace align {

template <typename T>
void block_align_images(
    int tile_size,
    int search_radius,
    const cv::Mat_<T> &reference,
    const cv::Mat_<T> &unaligned,
    cv::Mat_<cv::Vec2f> *inout_displacements,
    std::function<cv::Vec2i(const cv::Mat_<T> &, const cv::Mat_<T> &, cv::Vec2i)> align_func
);

/*class NaiveBlockAligner : public BlockAligner {
private:
    int tile_size_;
    int search_radius_;
    const cv::Mat &reference_;
    const cv::Mat &alternate_;

public:
    NaiveBlockAligner(int tile_size, int search_radius, const cv::Mat &reference, const cv::Mat &alternate)
        : tile_size_(tile_size),
        search_radius_(search_radius),
        reference_(reference),
        alternate_(alternate)
    {}

    int tile_size() const {
        return tile_size_;
    }

    float disp_residual_L1(int ref_tile_x, int ref_tile_y, float disp_x, float disp_y) const override {
        int ref_x = ref_tile_x * tile_size_,
            ref_y = ref_tile_y * tile_size_;
        return imaging::alignment::calculate_residual<1, uint8_t>(
            reference_(cv::Rect(ref_x, ref_y, tile_size_, tile_size_)),
            alternate_(cv::Rect(ref_x + (int)disp_x, ref_y + (int)disp_y, tile_size_, tile_size_))
        );
    }

    void align_L2(cv::Mat *inout_displacements) override {
        block_align_images(
            tile_size_, search_radius_,
            reference_, alternate_,
            inout_displacements,
            &align::align_one_naive<2>
        );
    }
};*/

}