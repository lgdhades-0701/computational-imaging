#include "imaging/stdafx.h"
#include "imaging/HierarchicalAlignment.h"

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

#include "imaging/AlignNaive.h"

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
    cv::Mat large_ref(16, 16, CV_8UC1, cv::Scalar(0));
    cv::rectangle(large_ref, cv::Rect(6, 0, 4, 16), 255, cv::FILLED);
    cv::Mat large_alt(16, 16, CV_8UC1, cv::Scalar(0));
    cv::rectangle(large_alt, cv::Rect(8, 0, 4, 16), 255, cv::FILLED);

    cv::Mat small_ref, small_alt;
    cv::resize(large_ref, small_ref, cv::Size(4, 4));
    cv::resize(large_alt, small_alt, cv::Size(4, 4));

    align::NaiveBlockAligner aligner(2, 2, small_ref, small_alt);
    cv::Mat out;
    aligner.align_L2(&out);
    //std::cout << out << std::endl;
    myprint(out);
}