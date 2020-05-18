#include "imaging/stdafx.h"

#include "block_align.h"
#include "imaging/Util.h"

namespace imaging {
namespace alignment {

LucasKanadeRefiner::LucasKanadeRefiner(const cv::Mat_<float> &reference, const cv::Mat_<float> &alternate)
    : reference(reference),
      alternate(alternate),
      ref_gradient_x(reference.size(), 0.0f),
      ref_gradient_y(reference.size(), 0.0f),
      alt_gradient_x(alternate.size(), 0.0f),
      alt_gradient_y(alternate.size(), 0.0f)
{
    // TODO(fyhuang): maybe create some kind of struct to hold precomputed gradients?
    cv::Sobel(reference, ref_gradient_x, CV_32F, 1, 0);
    cv::Sobel(reference, ref_gradient_y, CV_32F, 0, 1);

    cv::Sobel(alternate, alt_gradient_x, CV_32F, 1, 0);
    cv::Sobel(alternate, alt_gradient_y, CV_32F, 0, 1);
}

cv::Vec2f LucasKanadeRefiner::refine(
    cv::Rect ref_block,
    cv::Vec2f initial_disp,
    int num_iterations
) {
    // Iteratively refine "initial_disp" to sub-pixel precision using Lucas-Kanade optical flow
    cv::Vec2f curr_disp = initial_disp;

    for (int iter_ix = 0; iter_ix < num_iterations; iter_ix++) {
        // Compute current S^T * S; and S^T * t
        cv::Point2f center(
            curr_disp[0] + ref_block.x + (ref_block.width / 2.0f),
            curr_disp[1] + ref_block.y + (ref_block.height / 2.0f)
        );

        cv::Mat_<float> alt_x_patch, alt_y_patch, alt_img_patch;
        cv::getRectSubPix(alt_gradient_x, cv::Size(ref_block.width, ref_block.height), center, alt_x_patch);
        cv::getRectSubPix(alt_gradient_y, cv::Size(ref_block.width, ref_block.height), center, alt_y_patch);
        cv::getRectSubPix(alternate, cv::Size(ref_block.width, ref_block.height), center, alt_img_patch);

        cv::Mat_<float> sts(cv::Size(2, 2), 0.0f);
        cv::Mat_<float> stt(cv::Size(1, 2), 0.0f);
        for (int y = 0; y < ref_block.height; y++) {
            for (int x = 0; x < ref_block.width; x++) {
                float grad_x = alt_x_patch(y, x);
                float grad_y = alt_y_patch(y, x);

                sts(0, 0) += grad_x * grad_x;
                sts(1, 0) += grad_x * grad_y;
                sts(0, 1) += grad_x * grad_y;
                sts(1, 1) += grad_y * grad_y;

                float ref_pixel = reference(y, x);
                float alt_patch_pixel = alt_img_patch(y, x);
                float grad_t = alt_patch_pixel - ref_pixel; // TODO(fyhuang): negate this???

                stt(0, 0) += grad_x * grad_t;
                stt(1, 0) += grad_y * grad_t;
            }
        }

        std::cout << "Iteration " << iter_ix << ":\n";
        std::cout << curr_disp << std::endl;
        std::cout << alt_x_patch << std::endl;
        std::cout << alt_y_patch << std::endl;
        std::cout << sts << std::endl;
        std::cout << "\n\n";

        // Compute eigenvalues of sts
        float trace = sts(0, 0) + sts(1, 1);
        float det = sts(0, 0) * sts(1, 1) - sts(0, 1) * sts(1, 0);
        float l1 = trace / 2.0f + sqrtf(trace*trace / 4.0f - det);
        float l2 = trace / 2.0f - sqrtf(trace*trace / 4.0f - det);

        // If either eigenvalue is too small, then the inverse matrix doesn't exist/is ill-conditioned, and we cannot continue the iteration
        if (l1 < LK_INVERTIBLE_THRESHOLD || l2 < LK_INVERTIBLE_THRESHOLD) {
            printf("Not invertible! %f, %f\n", l1, l2);
            return curr_disp;
        }

        // Compute inverse of sts
        cv::Mat_<float> sts_inv(cv::Size(2, 2), 0.0f);
        sts_inv(0, 0) = sts(1, 1) / det;
        sts_inv(0, 1) = -sts(0, 1) / det;
        sts_inv(1, 0) = -sts(1, 0) / det;
        sts_inv(1, 1) = sts(0, 0) / det;

        // Compute the displacement update
        cv::Vec2f disp_update = cv::Vec2f(
            sts_inv(0, 0) * stt(0, 0) + sts_inv(0, 1) * stt(1, 0),
            sts_inv(1, 0) * stt(0, 0) + sts_inv(1, 1) * stt(1, 0)
        );

        printf("Lucas-Kanade updating displacement by %f, %f\n", disp_update[0], disp_update[1]);
        curr_disp = curr_disp + disp_update;
    }

    return curr_disp;
}

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