#pragma once

#include "BlockAligner.h"

namespace align {

template <int Norm>
static inline float calculate_residual_naive(
    const cv::Mat &a,
    const cv::Mat &b,
    int a_x, int a_y,
    int b_x, int b_y,
    int width, int height
) {
    float distance = 0.0f;
    for (int y = 0; y < height; y++) {
        const uint8_t *a_row_ptr = a.ptr(y + a_y);
        const uint8_t *b_row_ptr = b.ptr(y + b_y);
        for (int x = 0; x < width; x++) {
            float d_pixel = (float)a_row_ptr[x + a_x] -
                b_row_ptr[x + b_x];
#pragma warning(push)
#pragma warning(disable : 4127)
            // Conditional expression is supposed to be constant
            if (Norm == 2) {
                d_pixel *= d_pixel;
            }
#pragma warning(pop)
            distance += d_pixel;
        }
    }
    return distance;
}

template <int Norm>
static Disp align_one_naive(
    const cv::Mat &reference_tile,
    const cv::Mat &unaligned_neighborhood,
    int disp_offset_x,
    int disp_offset_y
) {
    // Return the displacement that minimizes the residual
    auto best = std::make_tuple(std::numeric_limits<float>::infinity(), 0, 0, 0.0f /* displacement length */);

    for (int disp_y = 0; disp_y < unaligned_neighborhood.rows - reference_tile.rows + 1; disp_y++) {
        for (int disp_x = 0; disp_x < unaligned_neighborhood.cols - reference_tile.cols + 1; disp_x++) {

            // Compute residual for this (disp_x, disp_y)
            float residual = calculate_residual_naive<Norm>(
                reference_tile,
                unaligned_neighborhood,
                0, 0,
                disp_x, disp_y,
                reference_tile.cols,
                reference_tile.rows
            );

            // See if this is the best one so far
            int real_disp_x = disp_x + disp_offset_x,
                real_disp_y = disp_y + disp_offset_y;
            float this_length = (float)cv::norm(cv::Vec2f((float)real_disp_x, (float)real_disp_y), cv::NORM_L1);
            if (residual < std::get<0>(best)) {
                best = std::make_tuple(residual, real_disp_x, real_disp_y, this_length);
            }
            else if (residual == std::get<0>(best)) {
                // Pick the smaller displacement in case of tie
                if (this_length < std::get<3>(best)) {
                    best = std::make_tuple(residual, real_disp_x, real_disp_y, this_length);
                }
            }
        }
    }

    return Disp(
        (float)std::get<1>(best),
        (float)std::get<2>(best)
    );
}

template <typename AlignT>
static void block_align_naive(
    int tile_size,
    int search_radius,
    const cv::Mat &reference,
    const cv::Mat &unaligned,
    cv::Mat *out,
    AlignT align_func
) {
    out->create(reference.rows / tile_size, reference.cols / tile_size, CV_32FC2);
    for (int tile_y = 0; tile_y < out->rows; tile_y++) {
        float *out_row_ptr = reinterpret_cast<float*>(out->ptr(tile_y));
        for (int tile_x = 0; tile_x < out->cols; tile_x++) {
            auto reference_tile_rect = cv::Rect(
                tile_x * tile_size,
                tile_y * tile_size,
                tile_size,
                tile_size
            );
            cv::Mat reference_tile(reference, reference_tile_rect);

            auto neighborhood_top_left = cv::Point(
                std::max(0, tile_x * tile_size - search_radius),
                std::max(0, tile_y * tile_size - search_radius)
            );
            cv::Mat unaligned_neighborhood(unaligned, cv::Rect(
                neighborhood_top_left,
                cv::Point(
                    std::min(unaligned.cols, (tile_x + 1) * tile_size + search_radius),
                    std::min(unaligned.rows, (tile_y + 1) * tile_size + search_radius)
                )
            ));

            auto disp = align_func(
                reference_tile,
                unaligned_neighborhood,
                neighborhood_top_left.x - reference_tile_rect.x,
                neighborhood_top_left.y - reference_tile_rect.y
            );
            
            out_row_ptr[tile_x * 2] = disp.x;
            out_row_ptr[tile_x * 2 + 1] = disp.y;
        }
    }
}

class NaiveBlockAligner : public BlockAligner {
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

    float tile_residual_L1(float ref_tile_x, float ref_tile_y, float alt_tile_x, float alt_tile_y) const override {
        return calculate_residual_naive<1>(
            reference_, alternate_,
            (int)(ref_tile_x * tile_size_), (int)(ref_tile_y * tile_size_),
            (int)(alt_tile_x * tile_size_), (int)(alt_tile_y * tile_size_),
            tile_size_, tile_size_
        );
    }

    void align_L2(cv::Mat *out) override {
        block_align_naive(tile_size_, search_radius_, reference_, alternate_, out, &align::align_one_naive<2>);
    }
};

}