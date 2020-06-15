#include "imaging/Util.h"

#include <gtest/gtest.h>

TEST(UtilExtractRect, WholePixel) {
    cv::Mat_<float> input(4, 4, 0.0f);
    for (int i = 0; i < 16; i++) {
        input(i/4, i%4) = (float)i;
    }

    cv::Mat_<float> output;
    util::extract_rect(input, output, cv::Rect(0, 0, 4, 4), cv::Vec2f(0, 0));

    for (int i = 0; i < 16; i++) {
        EXPECT_FLOAT_EQ(input(i/4, i%4), output(i/4, i%4));
    }
}

TEST(UtilExtractRect, QuarterPixel) {
    cv::Mat_<float> input(4, 4, 0.0f);
    for (int i = 0; i < 16; i++) {
        input(i / 4, i % 4) = (float)i;
    }

    cv::Mat_<float> output;
    util::extract_rect(input, output, cv::Rect(0, 0, 3, 3), cv::Vec2f(0.25f, 0.25f));

    EXPECT_FLOAT_EQ(1.25f, output(0, 0));
    EXPECT_FLOAT_EQ(11.25f, output(2, 2));
}

TEST(UtilExtractRect, Boundaries) {
    cv::Mat_<float> input(4, 4, 0.0f);
    for (int i = 0; i < 16; i++) {
        input(i / 4, i % 4) = (float)i;
    }

    cv::Mat_<float> output;
    util::extract_rect(input, output, cv::Rect(0, 0, 2, 2), cv::Vec2f(3.5f, 0.0f));

    EXPECT_FLOAT_EQ(3.0f, output(0, 0));
    EXPECT_FLOAT_EQ(7.0f, output(1, 0));
    EXPECT_FLOAT_EQ(2.5f, output(0, 1));
    EXPECT_FLOAT_EQ(6.5f, output(1, 1));
}
