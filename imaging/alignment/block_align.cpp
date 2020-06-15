#include "imaging/stdafx.h"

#include "block_align.h"
#include "imaging/Util.h"

namespace imaging {
namespace alignment {

template <typename T>
void block_align_images(
    int tile_size,
    int search_radius,
    const cv::Mat_<T> &reference,
    const cv::Mat_<T> &unaligned,
    cv::Mat_<cv::Vec2f> *inout_displacements,
    std::function<cv::Vec2i(const cv::Mat_<T> &, const cv::Mat_<T> &, cv::Vec2i)> align_func
) {
    if (inout_displacements->empty()) {
        inout_displacements->create(reference.rows / tile_size, reference.cols / tile_size);
        *inout_displacements = cv::Vec2f(0.0f, 0.0f);
    }

    for (int tile_y = 0; tile_y < inout_displacements->rows; tile_y++) {
        for (int tile_x = 0; tile_x < inout_displacements->cols; tile_x++) {
            auto reference_tile_rect = cv::Rect(
                tile_x * tile_size,
                tile_y * tile_size,
                tile_size,
                tile_size
            );
            cv::Mat reference_tile(reference, reference_tile_rect);

            auto starting_disp_f = inout_displacements->at<cv::Vec2f>(tile_y, tile_x);
            auto starting_disp_i = cv::Vec2i((int)starting_disp_f[0], (int)starting_disp_f[1]);
            auto base_coords = util::clamp_point(cv::Point(
                tile_x * tile_size + starting_disp_i[0],
                tile_y * tile_size + starting_disp_i[1]
            ),
                cv::Rect(
                    0, 0, unaligned.cols - tile_size, unaligned.rows - tile_size
                )
            );
            auto neighborhood_top_left = cv::Point(
                std::max(0, base_coords.x - search_radius),
                std::max(0, base_coords.y - search_radius)
            );
            auto neighborhood_bot_right = cv::Point(
                std::min(unaligned.cols, base_coords.x + tile_size + search_radius),
                std::min(unaligned.rows, base_coords.y + tile_size + search_radius)
            );
            cv::Mat unaligned_neighborhood(unaligned, cv::Rect(
                neighborhood_top_left,
                neighborhood_bot_right
            ));

            auto disp = align_func(
                reference_tile,
                unaligned_neighborhood,
                cv::Vec2i(neighborhood_top_left.x - reference_tile_rect.x,
                    neighborhood_top_left.y - reference_tile_rect.y)
            );

            (*inout_displacements)(tile_y, tile_x) =
                cv::Vec2f((float)disp[0], (float)disp[1]);
        }
    }
}

// Explicit instantiations
//template LucasKanadeRefiner::LucasKanadeRefiner(const cv::Mat_<uint8_t> &reference, const cv::Mat_<uint8_t> &alternate);

template void block_align_images(
    int tile_size,
    int search_radius,
    const cv::Mat_<float> &reference,
    const cv::Mat_<float> &unaligned,
    cv::Mat_<cv::Vec2f> *inout_displacements,
    std::function<cv::Vec2i(const cv::Mat_<float> &, const cv::Mat_<float> &, cv::Vec2i)> align_func
);

}
}