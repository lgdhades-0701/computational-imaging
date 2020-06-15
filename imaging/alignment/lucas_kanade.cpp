#include "imaging/stdafx.h"

#include <Eigen/Dense>

#include "lucas_kanade.h"
#include "imaging/Util.h"

template <typename T>
static void _opencv_to_eigen(const cv::Mat_<T> &opencv, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &eigen) {
    assert(opencv.rows == eigen.rows());
    assert(opencv.cols == eigen.cols());
    for (int y = 0; y < opencv.rows; y++) {
        for (int x = 0; x < opencv.cols; x++) {
            eigen(y, x) = opencv(y, x);
        }
    }
}

namespace imaging {
namespace alignment {

LucasKanadeRefiner::LucasKanadeRefiner(const cv::Mat_<float> &reference, const cv::Mat_<float> &alternate)
    : reference(reference),
    alternate(alternate),
    ref_norm(reference.size(), 0.0f),
    alt_norm(reference.size(), 0.0f),
    ref_blur(reference.size(), 0.0f),
    alt_blur(alternate.size(), 0.0f),
    ref_gradient_x(reference.size(), 0.0f),
    ref_gradient_y(reference.size(), 0.0f),
    alt_gradient_x(alternate.size(), 0.0f),
    alt_gradient_y(alternate.size(), 0.0f)
{
    reference.convertTo(ref_norm, -1, 1.0 / 255.0, 0.0);
    alternate.convertTo(alt_norm, -1, 1.0 / 255.0, 0.0);

    // Create blurred versions of ref and alt
    /*cv::boxFilter(ref_norm, ref_blur, -1, cv::Size(2, 2));
    cv::boxFilter(alt_norm, alt_blur, -1, cv::Size(2, 2));*/
    ref_norm.copyTo(ref_blur);
    alt_norm.copyTo(alt_blur);

    // TODO(fyhuang): maybe create some kind of struct to hold precomputed gradients?
    cv::Sobel(ref_norm, ref_gradient_x, CV_32F, 1, 0);
    cv::Sobel(ref_norm, ref_gradient_y, CV_32F, 0, 1);

    cv::Sobel(alt_norm, alt_gradient_x, CV_32F, 1, 0);
    cv::Sobel(alt_norm, alt_gradient_y, CV_32F, 0, 1);
}

cv::Vec2f LucasKanadeRefiner::refine(
    cv::Rect ref_block,
    cv::Vec2f initial_disp,
    int num_iterations
) {
    // Iteratively refine "initial_disp" to sub-pixel precision using Lucas-Kanade optical flow
    cv::Vec2f curr_disp = initial_disp;

    for (int iter_ix = 0; iter_ix < num_iterations; iter_ix++) {
        detail::LKRefineIter lk_iter(this, ref_block, curr_disp);

        lk_iter.compute_matrices_eigen();
        if (!lk_iter.can_perform_lk_eigen()) {
            return curr_disp;
        }

        std::cout << "Iteration " << iter_ix << ":\n";
        std::cout << curr_disp << std::endl;
        /*std::cout << lk_iter.alt_grad_x_patch << std::endl;
        std::cout << lk_iter.alt_grad_y_patch << std::endl;
        std::cout << lk_iter.ata << std::endl;
        std::cout << "\n\n";*/

        auto disp_update = lk_iter.compute_disp_eigen();
        printf("Lucas-Kanade updating displacement by %f, %f\n", disp_update[0], disp_update[1]);
        curr_disp = curr_disp + disp_update;
    }

    return curr_disp;
}

void LucasKanadeRefiner::visualize(cv::Rect ref_block, cv::Vec2f displacement, const std::string &out_prefix)
{
    cv::Mat_<float> ref_marked;
    ref_norm.copyTo(ref_marked);
    cv::rectangle(ref_marked, ref_block, 1.0f, 1, cv::LINE_AA, 0);
    util::vismat(ref_marked, out_prefix, "ref");

    const int SHIFT = 2;
    cv::Mat_<float> alt_marked;
    alt_norm.copyTo(alt_marked);
    cv::Point pt1(
        (int)((ref_block.x + displacement[0]) * (1 << SHIFT)),
        (int)((ref_block.y + displacement[1]) * (1 << SHIFT))
    );
    cv::Point pt2(
        (int)((ref_block.x + ref_block.width + displacement[0]) * (1 << SHIFT)),
        (int)((ref_block.y + ref_block.height + displacement[1]) * (1 << SHIFT))
    );
    cv::rectangle(alt_marked, pt1, pt2, 1.0f, 1, cv::LINE_AA, SHIFT);
    util::vismat(alt_marked, out_prefix, "alt");
}


////////////////////////////////
// detail

namespace detail {

inline LKRefineIter::LKRefineIter(const LucasKanadeRefiner *parent, cv::Rect ref_block, cv::Vec2f initial_disp)
    : parent(parent),
    ref_block(ref_block),
    ata(cv::Size(2, 2), 0.0f),
    atb(cv::Size(1, 2), 0.0f)
{
    // Initialize the RefineIter
    util::extract_rect(parent->alt_gradient_x, alt_grad_x_patch, ref_block, initial_disp);
    util::extract_rect(parent->alt_gradient_y, alt_grad_y_patch, ref_block, initial_disp);
    util::extract_rect(parent->alt_blur, alt_img_patch, ref_block, initial_disp);
}

void LKRefineIter::compute_matrices_cpp()
{
    for (int y = 0; y < ref_block.height; y++) {
        for (int x = 0; x < ref_block.width; x++) {
            float grad_x = alt_grad_x_patch(y, x);
            float grad_y = alt_grad_y_patch(y, x);

            ata(0, 0) += grad_x * grad_x;
            ata(1, 0) += grad_x * grad_y;
            ata(0, 1) += grad_x * grad_y;
            ata(1, 1) += grad_y * grad_y;

            float ref_pixel = parent->ref_blur(ref_block.y + y, ref_block.x + x);
            float alt_patch_pixel = alt_img_patch(y, x);
            float grad_t = alt_patch_pixel - ref_pixel;

            atb(0, 0) += grad_x * -grad_t;
            atb(1, 0) += grad_y * -grad_t;
        }
    }
}

void LKRefineIter::compute_matrices_eigen()
{
    int num_pix = ref_block.width * ref_block.height;
    Eigen::MatrixXf mat_a(num_pix, 2);
    Eigen::VectorXf vec_b(num_pix);

    for (int y = 0; y < ref_block.height; y++) {
        for (int x = 0; x < ref_block.width; x++) {
            float grad_x = alt_grad_x_patch(y, x);
            float grad_y = alt_grad_y_patch(y, x);

            float ref_pixel = parent->ref_blur(ref_block.y + y, ref_block.x + x);
            float alt_patch_pixel = alt_img_patch(y, x);
            float grad_t = alt_patch_pixel - ref_pixel;
            
            int pixel_id = y * ref_block.width + x;
            mat_a(pixel_id, 0) = grad_x;
            mat_a(pixel_id, 1) = grad_y;
            vec_b(pixel_id) = -grad_t;
        }
    }

    std::cout << "This iter grad_t norm = " << vec_b.norm() << "\n";

    Eigen::MatrixXf eigen_ata = mat_a.transpose() * mat_a;
    Eigen::MatrixXf eigen_atb = mat_a.transpose() * vec_b;
    assert(eigen_ata.rows() == 2);
    assert(eigen_ata.cols() == 2);
    assert(eigen_atb.rows() == 2);
    assert(eigen_atb.cols() == 1);

    for (int y = 0; y < 2; y++) {
        for (int x = 0; x < 2; x++) {
            ata(y, x) = eigen_ata(y, x);
        }
    }

    for (int i = 0; i < 2; i++) {
        atb(i, 0) = eigen_atb(i, 0);
    }
}

bool LKRefineIter::can_perform_lk_cpp()
{
    // Compute eigenvalues of ata
    float trace = ata(0, 0) + ata(1, 1);
    ata_det = ata(0, 0) * ata(1, 1) - ata(0, 1) * ata(1, 0);
    float l1 = trace / 2.0f + sqrtf(trace*trace / 4.0f - ata_det);
    float l2 = trace / 2.0f - sqrtf(trace*trace / 4.0f - ata_det);

    // If either eigenvalue is too small, then the inverse matrix doesn't exist/is ill-conditioned, and we cannot continue the iteration
    if (std::abs(l1) < LK_INVERTIBLE_THRESHOLD || std::abs(l2) < LK_INVERTIBLE_THRESHOLD) {
        printf("Not invertible! %f, %f\n", l1, l2);
        return false;
    }

    return true;
}

bool LKRefineIter::can_perform_lk_eigen()
{
    // Get eigenvalues of ata
    Eigen::MatrixXf eigen_ata(2, 2);
    _opencv_to_eigen(ata, eigen_ata);

    Eigen::VectorXcf eigenvalues = eigen_ata.eigenvalues();
    float l1 = eigenvalues(0).real(), l2 = eigenvalues(1).real();

    ata_det = eigen_ata.determinant();

    // If either eigenvalue is too small, then the inverse matrix doesn't exist/is ill-conditioned, and we cannot continue the iteration
    if (std::abs(l1) < LK_INVERTIBLE_THRESHOLD || std::abs(l2) < LK_INVERTIBLE_THRESHOLD) {
        printf("Not invertible! %f, %f\n", l1, l2);
        return false;
    }

    return true;
}

cv::Vec2f LKRefineIter::compute_disp_cpp()
{
    // Compute inverse of ata
    cv::Mat_<float> ata_inv(cv::Size(2, 2), 0.0f);
    ata_inv(0, 0) = ata(1, 1) / ata_det;
    ata_inv(0, 1) = -ata(0, 1) / ata_det;
    ata_inv(1, 0) = -ata(1, 0) / ata_det;
    ata_inv(1, 1) = ata(0, 0) / ata_det;

    // Compute the displacement update
    return cv::Vec2f(
        ata_inv(0, 0) * atb(0, 0) + ata_inv(0, 1) * atb(1, 0),
        ata_inv(1, 0) * atb(0, 0) + ata_inv(1, 1) * atb(1, 0)
    );
}

cv::Vec2f LKRefineIter::compute_disp_eigen()
{
    // Compute inverse of ata
    Eigen::MatrixXf eigen_ata(2, 2);
    _opencv_to_eigen(ata, eigen_ata);

    Eigen::MatrixXf eigen_atb(2, 1);
    _opencv_to_eigen(atb, eigen_atb);

    Eigen::MatrixXf ata_inv = eigen_ata.inverse();
    Eigen::Vector2f disp_update = ata_inv * eigen_atb;

    assert(disp_update.rows() == 2);
    assert(disp_update.cols() == 1);
    return cv::Vec2f(
        disp_update(0, 0),
        disp_update(1, 0)
    );
}

}
}
}