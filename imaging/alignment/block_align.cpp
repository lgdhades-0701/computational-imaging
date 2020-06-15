#include "imaging/stdafx.h"

#include "block_align.h"
#include "imaging/Util.h"

namespace imaging {
namespace alignment {

float CpuBlockAligner::disp_residual_L1(cv::Vec2i tile_coord, cv::Vec2f disp) const
{
    cv::Vec2i pixel_coord = tile_coord * tile_size_;
    return calculate_residual<1>(
        ref_(cv::Rect(pixel_coord[0], pixel_coord[1], tile_size_, tile_size_)),
        alt_(cv::Rect(pixel_coord[0] + (int)disp[0], pixel_coord[1] + (int)disp[1], tile_size_, tile_size_))
        );
}

void CpuBlockAligner::align(const cv::Mat_<cv::Vec2f> &in_disp, cv::Mat_<cv::Vec2f> *out_disp)
{
    detail::block_align_images(
        align_norm_,
        tile_size_, search_radius_,
        ref_, alt_,
        in_disp,
        out_disp
    );
}

namespace detail {

void block_align_images(
    int align_norm,
    int tile_size,
    int search_radius,
    const cv::Mat_<float> &reference,
    const cv::Mat_<float> &unaligned,
    const cv::Mat_<cv::Vec2f> &in_displacements,
    cv::Mat_<cv::Vec2f> *out_displacements
) {
    assert(align_norm == 1 || align_norm == 2);
    assert(!in_displacements.empty());
    assert(!out_displacements->empty());

    for (int tile_y = 0; tile_y < out_displacements->rows; tile_y++) {
        for (int tile_x = 0; tile_x < out_displacements->cols; tile_x++) {
            auto reference_tile_rect = cv::Rect(
                tile_x * tile_size,
                tile_y * tile_size,
                tile_size,
                tile_size
            );
            cv::Mat_<float> reference_tile(reference, reference_tile_rect);

            auto starting_disp_f = in_displacements(tile_y, tile_x);
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
            cv::Mat_<float> unaligned_neighborhood(unaligned, cv::Rect(
                neighborhood_top_left,
                neighborhood_bot_right
            ));
            
            cv::Vec2f disp(0.0f, 0.0f);
            cv::Vec2i disp_correction(neighborhood_top_left.x - reference_tile_rect.x,
                neighborhood_top_left.y - reference_tile_rect.y);
            if (align_norm == 1) {
                disp = align_one_block<1>(
                    reference_tile,
                    unaligned_neighborhood,
                    disp_correction
                );
            } else {
                disp = align_one_block<2>(
                    reference_tile,
                    unaligned_neighborhood,
                    disp_correction
                    );
            }

            (*out_displacements)(tile_y, tile_x) =
                cv::Vec2f((float)disp[0], (float)disp[1]);
        }
    }
}

}

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
}