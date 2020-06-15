#include "imaging/stdafx.h"

#include "imaging/alignment/hier_align.h"
#include "imaging/alignment/block_align.h"

namespace imaging_cpu {

#define CHECK_DIM(var_name, alignment) \
  if ((var_name) % (alignment) != 0) { \
    std::cout << #var_name << " " << var_name << " not multiple of " << alignment << "\n"; \
    exit(1); \
  }



template <int TileExpansionFactor, int PixelExpansionFactor>
void transfer_displacements(
    const cv::Mat &input,
    cv::Mat *output,
    const align::BlockAligner *aligner
) {

}

template <typename BlockAlignerT>
int hierarchical_align(
    const cv::Mat & reference,
    const cv::Mat & unaligned,
    cv::Mat *out_displacement
) {
    // Check input sizes
    if (reference.size != unaligned.size) {
        throw std::runtime_error("Matrix sizes not identical");
    }

    CHECK_DIM(reference.rows, 32 * 8);
    CHECK_DIM(reference.cols, 32 * 8);

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

    // Align at top level of pyramid, with empty starting displacements
    cv::Mat disp_l4;
    auto aligner_l4 = BlockAlignerT(8 /* tile_size */, 4 /* search_radius */, reference_l4, unaligned_l4);
    aligner_l4.align_L2(&disp_l4);
    // TODO(fyhuang): subpixel displacements at level 4

    // Transfer displacements over to level 3 and align
    cv::Mat disp_l3;
    auto aligner_l3 = BlockAlignerT(16 /* tile_size */, 4 /* search_radius */, reference_l3, unaligned_l3);
    transfer_displacements<2 /* 4x resolution, but 2x tile size */, 4>(disp_l4, &disp_l3, &aligner_l3);
    aligner_l3.align_L2(&disp_l3);

    // Transfer displacements over to level 2 and align
    cv::Mat disp_l2;
    auto aligner_l2 = BlockAlignerT(16 /* tile_size */, 4 /* search_radius */, reference_l2, unaligned_l2);
    transfer_displacements<4, 4>(disp_l3, &disp_l2, &aligner_l2);
    aligner_l2.align_L2(&disp_l2);

    // Transfer displacements to level 1 and do final alignment
    auto aligner_l1 = BlockAlignerT(16 /* tile_size */, 1 /* search_radius */, reference, unaligned);
    transfer_displacements<2, 2>(disp_l2, out_displacement, &aligner_l1);
    // TODO(fyhuang): should be L1 residual, not L2
    aligner_l1.align_L2(out_displacement);

    return 16;
}

AlignedImage compute_alignment(const Inputs &inputs) {
    std::unique_ptr<cv::Mat> displacements(new cv::Mat());
    int tile_size = hierarchical_align<align::NaiveBlockAligner>(
        *inputs.reference, *inputs.alternate, displacements.get());

    // Compute the reconstructed aligned image based on the displacements, rounded to pixels
    std::unique_ptr<cv::Mat> aligned(new cv::Mat(
        inputs.reference->rows, inputs.reference->cols, CV_8UC1, cv::Scalar(0)));
    for (int tile_y = 0; tile_y < aligned->rows / tile_size; tile_y++) {
        for (int tile_x = 0; tile_x < aligned->cols / tile_size; tile_x++) {
            // Copy from alternate at tile location + displacement
            int starting_x = tile_x * tile_size,
                starting_y = tile_y * tile_size;
            auto disp_float = displacements->at<cv::Vec2f>(tile_y, tile_x);
            auto disp = cv::Vec2i(
                (int)round(disp_float[0]),
                (int)round(disp_float[1])
            );
            auto alt_tile = (*inputs.alternate)(cv::Rect(
                starting_x + disp[0], starting_y + disp[1], tile_size, tile_size
            ));
            auto aligned_tile = (*aligned)(cv::Rect(
                starting_x, starting_y, tile_size, tile_size
            ));
            alt_tile.copyTo(aligned_tile);
        }
    }

    return AlignedImage(
        tile_size,
        std::move(displacements),
        std::move(aligned)
    );
}

// Explicit template instantiations for unit tests
template void get_coords_to_check<2>(int, int, int, int, cv::Vec2i[]);
template void get_coords_to_check<4>(int, int, int, int, cv::Vec2i[]);
template void transfer_displacements<2, 4>(const cv::Mat &, cv::Mat *, const align::BlockAligner *);
template void transfer_displacements<4, 4>(const cv::Mat &, cv::Mat *, const align::BlockAligner *);

template int hierarchical_align<align::NaiveBlockAligner>(const cv::Mat &, const cv::Mat &, cv::Mat *);

namespace detail {

inline void get_coords_to_check(
    cv::Size source_size,
    int expansion_factor,
    cv::Vec2i dest_coord,
    cv::Vec2i out_coords[3]
) {
    assert(expansion_factor % 2 == 0);
    const int half_expansion = expansion_factor / 2;
    cv::Vec2i source_coord = dest_coord / expansion_factor;
    cv::Vec2i dest_coord_floor = source_coord * expansion_factor;

    int next_nearest_x;
    if (dest_coord[0] - dest_coord_floor[0] >= half_expansion) {
        next_nearest_x = source_coord[0] + 1;
    }
    else {
        next_nearest_x = source_coord[0] - 1;
    }
    next_nearest_x = cv::borderInterpolate(next_nearest_x, source_size.width, cv::BORDER_REPLICATE);

    int next_nearest_y;
    if (dest_coord[1] - dest_coord_floor[1] >= half_expansion) {
        next_nearest_y = source_coord[1] + 1;
    }
    else {
        next_nearest_y = source_coord[1] - 1;
    }
    next_nearest_y = cv::borderInterpolate(next_nearest_y, source_size.height, cv::BORDER_REPLICATE);

    out_coords[0] = source_coord;
    out_coords[1] = cv::Vec2i(next_nearest_x, source_coord[1]);
    out_coords[2] = cv::Vec2i(source_coord[0], next_nearest_y);
}

void AlignLevel::seed_initial_displacements(const AlignLevel &prev_level)
{
    // Upsample the displacements from the previous level of the pyramid to this one
    assert(prev_level.downscale > downscale);
    int pixel_expansion_factor = prev_level.downscale / downscale;
    int tile_expansion_factor = pixel_expansion_factor * tile_size / prev_level.tile_size;

    printf("prev_level = %d (%dx%d), level = %d (%dx%d)\n",
        prev_level.downscale, prev_level.ref.cols, prev_level.ref.rows,
        downscale, ref.cols, ref.rows);

    for (int out_y = 0; out_y < initial_disp.rows; out_y++) {
        for (int out_x = 0; out_x < initial_disp.cols; out_x++) {
            cv::Vec2i coords_to_check[3];
            get_coords_to_check(prev_level.ending_disp.size(), tile_expansion_factor, cv::Vec2i(out_x, out_y), coords_to_check);

            // Check all three inputs
            float best_residual = std::numeric_limits<float>::infinity();
            cv::Vec2f best_disp;
            for (const auto coord : coords_to_check) {
                cv::Vec2f disp = prev_level.ending_disp(coord[1], coord[0]) * pixel_expansion_factor;
                float residual = aligner->disp_residual_L1(out_x, out_y, disp[0], disp[1]);
                if (residual < best_residual) {
                    best_residual = residual;
                    best_disp = disp;
                }
            }

            // Set the output to the best displacement
            output->at<cv::Vec2f>(out_y, out_x) = best_disp;
        }
    }
}

}

}
