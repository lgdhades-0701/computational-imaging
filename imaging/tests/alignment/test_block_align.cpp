#include "imaging/stdafx.h"
#include "imaging/alignment/block_align.h"

#include <gtest/gtest.h>

using namespace cv;
using namespace align;
using namespace imaging::alignment;

TEST(CalculateResidual, Zero) {
    Mat_<uint8_t> a(Size(8, 8), 0);
    float residual;
    
    residual = calculate_residual<1>(a, a);
    EXPECT_EQ(0.0f, residual);
    residual = calculate_residual<2>(a, a);
    EXPECT_EQ(0.0f, residual);
}

TEST(CalculateResidual, L1vsL2) {
    Mat_<uint8_t> a(Size(8, 8), 0);
    Mat_<uint8_t> b(Size(8, 8), 0);
    a(1, 1) = 4;

    float residual;
    residual = calculate_residual<1>(a, b);
    EXPECT_FLOAT_EQ(4.0f, residual);
    residual = calculate_residual<2>(a, b);
    EXPECT_FLOAT_EQ(16.0f, residual);
}

TEST(AlignOneBlock, EightByEight) {
    Mat_<uint8_t> reference(Size(8, 8), 255);
    Mat_<uint8_t> neighborhood(Size(16, 16), 0);
    cv::rectangle(neighborhood, cv::Rect(4, 4, 8, 8), 255, cv::FILLED);

    EXPECT_EQ(
        Vec2i(4, 4),
        align_one_block<1>(reference, neighborhood)
    );
}

TEST(AlignOneBlock, EightByEightWithNoise) {
    Mat_<uint8_t> reference(Size(8, 8), 255);
    Mat_<uint8_t> neighborhood(Size(16, 16), 0);
    cv::rectangle(neighborhood, cv::Rect(4, 4, 8, 8), 255, cv::FILLED);

    // Add some noise
    neighborhood.at<uint8_t>(6, 7) = 192;
    neighborhood.at<uint8_t>(8, 10) = 0;
    
    EXPECT_EQ(
        Vec2i(4, 4),
        align_one_block<1>(reference, neighborhood)
    );

    EXPECT_EQ(
        Vec2i(4, 4),
        align_one_block<2>(reference, neighborhood)
    );
}

TEST(AlignOneBlock, EightByEightPickBest) {
    Mat_<uint8_t> reference(Size(8, 8), 255);
    Mat_<uint8_t> neighborhood(Size(16, 16), 240);
    cv::rectangle(neighborhood, cv::Rect(0, 0, 8, 8), 230, cv::FILLED);

    EXPECT_EQ(
        Vec2i(8, 0),
        align_one_block<2>(reference, neighborhood)
    );
}

TEST(AlignOneBlock, BreakTies) {
    Mat_<uint8_t> reference(Size(8, 8), 255);
    Mat_<uint8_t> neighborhood(Size(16, 16), 240);

    EXPECT_EQ(
        Vec2i(0, 0),
        align_one_block<2>(reference, neighborhood)
    );
}

TEST(AlignOneBlock, AdjustByOffset) {
    Mat_<uint8_t> reference(Size(8, 8), 255);
    Mat_<uint8_t> neighborhood(Size(16, 16), 240);
    cv::rectangle(neighborhood, cv::Rect(0, 0, 8, 8), 230, cv::FILLED);

    EXPECT_EQ(
        Vec2i(0, 0),
        align_one_block<2>(reference, neighborhood, cv::Vec2i(-8, 0))
    );
}

TEST(AlignOneBlock, AliasedLine) {
    Mat_<uint8_t> reference(Size(2, 2), 0);
    cv::rectangle(reference, cv::Rect(1, 0, 1, 2), 128, cv::FILLED);
    Mat_<uint8_t> neighborhood(Size(4, 4), 0);
    cv::rectangle(neighborhood, cv::Rect(2, 0, 1, 4), 255, cv::FILLED);

    EXPECT_EQ(
        Vec2i(1, 0),
        align_one_block<2>(reference, neighborhood)
    );
}


TEST(CpuBlockAligner, Basic) {
    cv::Mat_<float> reference(16, 16, 0.0f);
    cv::rectangle(reference, cv::Rect(4, 4, 8, 8), 1.0f, cv::FILLED);
    cv::Mat_<float> alternate(16, 16, 0.0f);
    cv::rectangle(alternate, cv::Rect(6, 6, 8, 8), 1.0f, cv::FILLED);

    CpuBlockAligner aligner(2, 8 /* tile_size */, 4 /* search_radius */, reference, alternate);

    cv::Mat_<cv::Vec2f> in_disp(2, 2, cv::Vec2f());
    cv::Mat_<cv::Vec2f> out_disp(2, 2);
    aligner.align(in_disp, &out_disp);

    cv::Vec2f result = out_disp(0, 0);
    EXPECT_FLOAT_EQ(2.0f, result[0]);
    EXPECT_FLOAT_EQ(2.0f, result[1]);
}

TEST(CpuBlockAligner, WithStartingDisp) {
    cv::Mat_<float> reference(2, 4, 0.0f);
    reference(1, 1) = 1.0f;
    cv::Mat_<float> alternate(2, 4, 0.0f);
    alternate(1, 3) = 1.0f;

    CpuBlockAligner aligner(2, 1 /* tile_size */, 1 /* search_radius */, reference, alternate);

    cv::Mat_<cv::Vec2f> in_disp(2, 4, cv::Vec2f());
    cv::Mat_<cv::Vec2f> out_disp(2, 4);
    aligner.align(in_disp, &out_disp);

    // The true displacement is (2.0f, 0.0f), but with a search_radius of 1, we won't be able to find it
    cv::Vec2f result = out_disp(1, 1);
    EXPECT_FLOAT_EQ(0.0f, result[0]);
    EXPECT_FLOAT_EQ(0.0f, result[1]);

    // Setting the starting displacement to (1.0f, 0.0f) makes it possible to find
    in_disp(1, 1) = cv::Vec2f(1.0f, 0.0f);
    aligner.align(in_disp, &out_disp);
    result = out_disp(1, 1);
    EXPECT_FLOAT_EQ(2.0f, result[0]);
    EXPECT_FLOAT_EQ(0.0f, result[1]);
}

TEST(CpuBlockAligner, LargeStartingDisp) {
    cv::Mat_<float> reference(2, 2, 0.0f);
    reference(0, 0) = 1.0f;
    cv::Mat_<float> alternate(2, 2, 0.0f);
    alternate(1, 1) = 1.0f;

    CpuBlockAligner aligner(2, 1 /* tile_size */, 1 /* search_radius */, reference, alternate);

    // Aligner should clamp out-of-bounds displacements to the image edge
    cv::Mat_<cv::Vec2f> in_disp(2, 2, cv::Vec2f());
    in_disp(0, 0) = cv::Vec2f(-100.0f, -100.0f);

    cv::Mat_<cv::Vec2f> out_disp(2, 2);
    aligner.align(in_disp, &out_disp);

    cv::Vec2f result = out_disp(0, 0);
    EXPECT_FLOAT_EQ(1.0f, result[0]);
    EXPECT_FLOAT_EQ(1.0f, result[1]);
}