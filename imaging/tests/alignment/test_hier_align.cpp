#include "imaging/stdafx.h"
#include "imaging/alignment/hier_align.h"

#include <gtest/gtest.h>

using namespace imaging_cpu;

bool array_contains(const cv::Vec2i coords[3], const cv::Vec2i &to_check) {
    for (int i = 0; i < 3; i++) {
        if (coords[i] == to_check) {
            return true;
        }
    }
    return false;
}

TEST(GetCoordsToCheck, TwoByTwo) {
    cv::Vec2i coords[3];
    get_coords_to_check<2>(2, 2, 0, 0, coords);
    EXPECT_TRUE(array_contains(coords, cv::Vec2i(0, 0)));

    get_coords_to_check<2>(2, 2, 1, 1, coords);
    EXPECT_TRUE(array_contains(coords, cv::Vec2i(0, 0)));
    EXPECT_TRUE(array_contains(coords, cv::Vec2i(1, 0)));
    EXPECT_TRUE(array_contains(coords, cv::Vec2i(0, 1)));

    get_coords_to_check<2>(2, 2, 3, 1, coords);
    EXPECT_TRUE(array_contains(coords, cv::Vec2i(1, 0)));
    EXPECT_TRUE(array_contains(coords, cv::Vec2i(1, 1)));

    get_coords_to_check<2>(2, 2, 2, 2, coords);
    EXPECT_TRUE(array_contains(coords, cv::Vec2i(1, 0)));
    EXPECT_TRUE(array_contains(coords, cv::Vec2i(0, 1)));
    EXPECT_TRUE(array_contains(coords, cv::Vec2i(1, 1)));
}

TEST(GetCoordsToCheck, ExpansionFactor4) {
    cv::Vec2i coords[3];
    get_coords_to_check<4>(4, 4, 2, 2, coords);
    EXPECT_TRUE(array_contains(coords, cv::Vec2i(0, 0)));
    EXPECT_TRUE(array_contains(coords, cv::Vec2i(0, 1)));
    EXPECT_TRUE(array_contains(coords, cv::Vec2i(1, 0)));

    get_coords_to_check<4>(4, 4, 9, 5, coords);
    EXPECT_TRUE(array_contains(coords, cv::Vec2i(2, 0)));
    EXPECT_TRUE(array_contains(coords, cv::Vec2i(2, 1)));
    EXPECT_TRUE(array_contains(coords, cv::Vec2i(1, 1)));

    get_coords_to_check<4>(4, 4, 7, 8, coords);
    EXPECT_TRUE(array_contains(coords, cv::Vec2i(1, 1)));
    EXPECT_TRUE(array_contains(coords, cv::Vec2i(1, 2)));
    EXPECT_TRUE(array_contains(coords, cv::Vec2i(2, 2)));

    get_coords_to_check<4>(4, 4, 5, 11, coords);
    EXPECT_TRUE(array_contains(coords, cv::Vec2i(0, 2)));
    EXPECT_TRUE(array_contains(coords, cv::Vec2i(1, 2)));
    EXPECT_TRUE(array_contains(coords, cv::Vec2i(1, 3)));
}

#include "imaging/alignment/block_align.h"

void myprint(const cv::Mat &mat) {
    for (int y = 0; y < mat.rows; y++) {
        for (int x = 0; x < mat.cols; x++) {
            auto as_vec = mat.at<cv::Vec2f>(y, x);
            auto as_ptr = reinterpret_cast<const float*>(mat.ptr(y));
            printf("(%f %f) ptr(%f %f), ",
                (double)as_vec[0],
                (double)as_vec[1],
                (double)(as_ptr[x * 2]),
                (double)(as_ptr[x * 2 + 1])
            );
        }
        printf("\n");
    }
}

TEST(TransferDisplacements, SimpleUpsample24) {
    /* Ref/alt are set up so that:
        - At 4x2 scale, tile (0, 0)'s (A) best alignment is (1, 0)
        - At 16x4 scale, tile (1, 1) is a subtile of tile A; however, its
          best alignment is (0, 0), not (4, 0)
    */
    cv::Mat large_ref(8, 16, CV_8UC1, cv::Scalar(0));
    cv::rectangle(large_ref, cv::Rect(4, 0, 6, 4), 255, cv::FILLED);
    cv::rectangle(large_ref, cv::Rect(4, 4, 4, 4), 128, cv::FILLED);
    cv::rectangle(large_ref, cv::Rect(8, 4, 2, 4), 255, cv::FILLED);
    cv::Mat large_alt(8, 16, CV_8UC1, cv::Scalar(0));
    cv::rectangle(large_alt, cv::Rect(8, 0, 4, 8), 255, cv::FILLED);
    cv::rectangle(large_alt, cv::Rect(4, 4, 4, 4), 128, cv::FILLED);

    cv::Mat small_ref, small_alt;
    cv::resize(large_ref, small_ref, cv::Size(4, 2));
    cv::resize(large_alt, small_alt, cv::Size(4, 2));

    align::NaiveBlockAligner aligner_l2(2, 2, small_ref, small_alt);
    cv::Mat disp_l2;
    aligner_l2.align_L2(&disp_l2);

    align::NaiveBlockAligner aligner_l1(4, 4, large_ref, large_alt);
    cv::Mat disp_l1;
    transfer_displacements<2, 4>(disp_l2, &disp_l1, &aligner_l1);

    EXPECT_EQ(cv::Vec2f(0, 0), disp_l1.at<cv::Vec2f>(1, 1));
    EXPECT_EQ(cv::Vec2f(4, 0), disp_l1.at<cv::Vec2f>(0, 1));
    
    // Right half displacements should be unchanged
    EXPECT_EQ(cv::Vec2f(0, 0), disp_l1.at<cv::Vec2f>(0, 2));
    EXPECT_EQ(cv::Vec2f(0, 0), disp_l1.at<cv::Vec2f>(1, 2));
    EXPECT_EQ(cv::Vec2f(0, 0), disp_l1.at<cv::Vec2f>(0, 3));
    EXPECT_EQ(cv::Vec2f(0, 0), disp_l1.at<cv::Vec2f>(1, 3));
}