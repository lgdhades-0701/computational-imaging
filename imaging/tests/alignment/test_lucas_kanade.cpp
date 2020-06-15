#include "imaging/stdafx.h"
#include "imaging/alignment/lucas_kanade.h"
#include "imaging/Util.h"

#include <gtest/gtest.h>

using namespace cv;
using namespace imaging::alignment;

static Mat imread_as_gray_float(const std::string &filename) {
    Mat temp = imread(filename, cv::IMREAD_GRAYSCALE);
    Mat float_mat(temp.size(), CV_32FC1);
    temp.convertTo(float_mat, CV_32FC1);
    return float_mat;
}

template <typename T>
static void expect_mat_float_eq(const Mat_<T> &m1, const Mat_<T> &m2) {
    EXPECT_EQ(m1.rows, m2.rows);
    EXPECT_EQ(m1.cols, m2.cols);

    for (int y = 0; y < m1.rows; y++) {
        for (int x = 0; x < m1.cols; x++) {
            T v1 = m1(y, x);
            T v2 = m2(y, x);
            if (std::abs(v1 - v2) > 0.00001) {
                fprintf(stderr, "Element (%d,%d) of matrices is not equal: %f, %f\n", y, x, v1, v2);
            }
            ASSERT_FLOAT_EQ(v1, v2);
        }
    }
}

TEST(LucasKanadeRefiner, Gradients) {
    float data[] = {
        0, 32, 64, 0,
        32, 64, 96, 32,
        64, 96, 128, 64,
        0, 32, 64, 0,
    };
    Mat_<float> mat(4, 4, data);

    LucasKanadeRefiner refiner(mat, mat);
    EXPECT_FLOAT_EQ(refiner.ref_gradient_x(1, 1), refiner.ref_gradient_x(2, 1));
    EXPECT_TRUE(refiner.ref_gradient_x(1, 2) < 0.0f);

    EXPECT_FLOAT_EQ(refiner.ref_gradient_y(1, 1), refiner.ref_gradient_y(1, 2));
    EXPECT_FLOAT_EQ(refiner.ref_gradient_x(1, 1), refiner.ref_gradient_y(1, 1));
}

TEST(LucasKanadeRefiner, NoDisplacement) {
    // Correct displacement is -2.0, -3.0
    Mat_<float> ref = imread_as_gray_float("testdata/city_ref.png"),
        alt = imread_as_gray_float("testdata/city_alt.png");
    LucasKanadeRefiner refiner(ref, alt);
    cv::Vec2f result = refiner.refine(cv::Rect(10, 10, 16, 16), cv::Vec2f(-2.0, -3.0), 1);
    
    EXPECT_FLOAT_EQ(-2.0f, result[0]);
    EXPECT_FLOAT_EQ(-3.0f, result[1]);
}

TEST(LucasKanadeRefiner, CompareSteps) {
    // Correct displacement is 2.0, 3.0
    Mat_<float> ref = imread_as_gray_float("testdata/desert_ref.png"),
        alt = imread_as_gray_float("testdata/desert_alt.png");
    LucasKanadeRefiner refiner(ref, alt);

    // Compute matrices
    printf("Comparing matrices...\n");
    imaging::alignment::detail::LKRefineIter iter1(&refiner, cv::Rect(2, 2, 4, 4), cv::Vec2f(1.0f, 4.0f));
    iter1.compute_matrices_cpp();

    imaging::alignment::detail::LKRefineIter iter2(&refiner, cv::Rect(2, 2, 4, 4), cv::Vec2f(1.0f, 4.0f));
    iter2.compute_matrices_eigen();

    expect_mat_float_eq(iter1.ata, iter2.ata);
    expect_mat_float_eq(iter1.atb, iter2.atb);

    // Check if LK can be performed
    printf("Comparing determinants...\n");
    bool can1 = iter1.can_perform_lk_cpp();
    bool can2 = iter2.can_perform_lk_eigen();
    EXPECT_EQ(can1, can2);
    EXPECT_FLOAT_EQ(iter1.ata_det, iter2.ata_det);

    // Do the inversion
    cv::Vec2f disp1 = iter1.compute_disp_cpp();
    cv::Vec2f disp2 = iter1.compute_disp_eigen();
    EXPECT_FLOAT_EQ(disp1[0], disp2[0]);
    EXPECT_FLOAT_EQ(disp1[1], disp2[1]);
}

TEST(LucasKanadeRefiner, City) {
    // Correct displacement is -2.0, -3.0
    Mat_<float> ref = imread_as_gray_float("testdata/city_ref.png"),
        alt = imread_as_gray_float("testdata/city_alt.png");

    const cv::Vec2f correct_displacement(-2.0f, -3.0f);
    const cv::Vec2f starting_displacement(-1.0f, -4.0f);

    LucasKanadeRefiner refiner(ref, alt);
    cv::Vec2f result = refiner.refine(cv::Rect(12, 12, 16, 16), starting_displacement, 5);
    EXPECT_LT(cv::norm(result - correct_displacement), cv::norm(starting_displacement - correct_displacement));
}

TEST(LucasKanadeRefiner, Visualize) {
    // Correct displacement is 2.0, 3.0
    Mat_<float> ref = imread_as_gray_float("testdata/city_ref.png"),
        alt = imread_as_gray_float("testdata/city_alt.png");


    LucasKanadeRefiner refiner(ref, alt);
    cv::Rect ref_block(48, 35, 8, 8);

    cv::Vec2f disp(-16, 10);
    imaging::alignment::detail::LKRefineIter iter(&refiner, ref_block, disp);

    refiner.visualize(ref_block, disp, "temp/");
    util::vismat(iter.alt_img_patch, "temp/", "alt_patch");


    /*cv::Vec2f final_result = refiner.refine(ref_block, cv::Vec2f(-1.9f, -3.1f), 100);
    refiner.visualize(ref_block, final_result);*/
    cv::waitKey(0);
}
