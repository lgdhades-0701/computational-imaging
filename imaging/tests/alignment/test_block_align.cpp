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

TEST(AlignOneNaive, EightByEight) {
    Mat_<uint8_t> reference(Size(8, 8), 255);
    Mat_<uint8_t> neighborhood(Size(16, 16), 0);
    cv::rectangle(neighborhood, cv::Rect(4, 4, 8, 8), 255, cv::FILLED);

    EXPECT_EQ(
        Vec2i(4, 4),
        align_one_block<1>(reference, neighborhood)
    );
}

TEST(AlignOneNaive, EightByEightWithNoise) {
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

TEST(AlignOneNaive, EightByEightPickBest) {
    Mat_<uint8_t> reference(Size(8, 8), 255);
    Mat_<uint8_t> neighborhood(Size(16, 16), 240);
    cv::rectangle(neighborhood, cv::Rect(0, 0, 8, 8), 230, cv::FILLED);

    EXPECT_EQ(
        Vec2i(8, 0),
        align_one_block<2>(reference, neighborhood)
    );
}

TEST(AlignOneNaive, BreakTies) {
    Mat_<uint8_t> reference(Size(8, 8), 255);
    Mat_<uint8_t> neighborhood(Size(16, 16), 240);

    EXPECT_EQ(
        Vec2i(0, 0),
        align_one_block<2>(reference, neighborhood)
    );
}

TEST(AlignOneNaive, AdjustByOffset) {
    Mat_<uint8_t> reference(Size(8, 8), 255);
    Mat_<uint8_t> neighborhood(Size(16, 16), 240);
    cv::rectangle(neighborhood, cv::Rect(0, 0, 8, 8), 230, cv::FILLED);

    EXPECT_EQ(
        Vec2i(0, 0),
        align_one_block<2>(reference, neighborhood, cv::Vec2i(-8, 0))
    );
}

TEST(AlignOneNaive, AliasedLine) {
    Mat_<uint8_t> reference(Size(2, 2), 0);
    cv::rectangle(reference, cv::Rect(1, 0, 1, 2), 128, cv::FILLED);
    Mat_<uint8_t> neighborhood(Size(4, 4), 0);
    cv::rectangle(neighborhood, cv::Rect(2, 0, 1, 4), 255, cv::FILLED);

    EXPECT_EQ(
        Vec2i(1, 0),
        align_one_block<2>(reference, neighborhood)
    );
}

/*TEST(BlockAlignNaive, OneTile) {
    cv::Mat reference(8, 8, CV_8UC1, cv::Scalar(255));
    cv::Mat unaligned(8, 8, CV_8UC1, cv::Scalar(240));

    auto test_align = [](const cv::Mat &ref, const cv::Mat &n, int ox, int oy) -> Disp {
        EXPECT_EQ(8, ref.rows);
        EXPECT_EQ(8, ref.cols);
        EXPECT_EQ(8, n.rows);
        EXPECT_EQ(8, n.cols);
        EXPECT_EQ(0, ox);
        EXPECT_EQ(0, oy);
        return Disp(4, 4);
    };

    cv::Mat out;
    block_align_images(8, 0, reference, unaligned, &out, test_align);
    auto disp = out.at<cv::Vec2f>(0, 0);
    EXPECT_EQ(cv::Vec2f(4, 4), disp);
}

TEST(BlockAlignNaive, TilesWithNeighborhood) {
    cv::Mat reference(16, 16, CV_8UC1, cv::Scalar(255));
    cv::Mat unaligned(16, 16, CV_8UC1, cv::Scalar(240));

    auto test_align = [](const cv::Mat &ref, const cv::Mat &n, int, int) -> Disp {
        EXPECT_EQ(8, ref.rows);
        EXPECT_EQ(8, ref.cols);
        EXPECT_EQ(10, n.rows);
        EXPECT_EQ(10, n.cols);
        return Disp(4, 4);
    };

    cv::Mat out;
    block_align_images(8, 2, reference, unaligned, &out, test_align);
    EXPECT_EQ(2, out.cols);
    EXPECT_EQ(2, out.rows);

    EXPECT_EQ(cv::Vec2f(4, 4), out.at<cv::Vec2f>(0, 0));
    EXPECT_EQ(cv::Vec2f(4, 4), out.at<cv::Vec2f>(0, 1));
    EXPECT_EQ(cv::Vec2f(4, 4), out.at<cv::Vec2f>(1, 0));
    EXPECT_EQ(cv::Vec2f(4, 4), out.at<cv::Vec2f>(1, 1));
}

/*TEST(BlockAlignNaive, WithStartingDisp) {
    cv::Mat reference(2, 8, CV_8UC1, cv::Scalar(128));
    cv::rectangle(reference, cv::Rect(0, 0, 2, 2), 255, cv::FILLED);
    cv::Mat unaligned(2, 8, CV_8UC1, cv::Scalar(0));
    cv::rectangle(unaligned, cv::Rect(4, 0, 2, 2), 255, cv::FILLED);

    NaiveBlockAligner aligner(2, 1, reference, unaligned);

    // With empty starting displacement, aligner cannot find the optimal alignment (4, 0)
    cv::Mat disp;
    aligner.align_L2(&disp);
    EXPECT_EQ(cv::Vec2f(0, 0), disp.at<cv::Vec2f>(0, 0));

    // With starting displacement of 3, aligner should be able to find optimal alignment
    disp.at<cv::Vec2f>(0, 0) = cv::Vec2f(3, 0);
    aligner.align_L2(&disp);
    EXPECT_EQ(cv::Vec2f(4, 0), disp.at<cv::Vec2f>(0, 0));
}

TEST(BlockAlignNaive, StartingDispDoesntExceedBounds) {
    cv::Mat reference(4, 4, CV_8UC1, cv::Scalar(0));
    cv::Mat alternate(4, 4, CV_8UC1, cv::Scalar(0));
    alternate.at<uint8_t>(1, 1) = 128;
    alternate.at<uint8_t>(3, 3) = 64;

    auto test_align = [](const cv::Mat &ref, const cv::Mat &n, int, int) -> Disp {
        EXPECT_EQ(2, ref.rows);
        EXPECT_EQ(2, ref.cols);
        EXPECT_EQ(3, n.rows);
        EXPECT_EQ(3, n.cols);

        if (n.at<uint8_t>(1, 1) == 128) {
            // Top left
            SUCCEED();
        }
        else if (n.at<uint8_t>(2, 2) == 64) {
            // Bottom right
            SUCCEED();
        }
        else if (n.at<uint8_t>(0, 1) == 128) {
            // Bottom left
            SUCCEED();
        }
        else if (n.at<uint8_t>(1, 0) == 128) {
            // Top right
            SUCCEED();
        }
        else {
            std::cout << n << std::endl;
            ADD_FAILURE();
        }

        return Disp(0, 0);
    };

    // Even if the starting displacement is out of bounds, the aligner should clamp it
    // to the edges of the alternate image.
    cv::Mat disp(2, 2, CV_32FC2, cv::Scalar(0.f));
    disp.at<cv::Vec2f>(0, 0) = cv::Vec2f(-100, 0);
    disp.at<cv::Vec2f>(1, 1) = cv::Vec2f(2, 87);
    block_align_naive(2, 1, reference, alternate, &disp, test_align);
}*/