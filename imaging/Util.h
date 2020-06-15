#pragma once

#include <opencv2/opencv.hpp>

namespace util {

// Clamp a point inside a rectangle
static cv::Point clamp_point(const cv::Point &input, const cv::Rect &rect) {
    return cv::Point(
        std::min(std::max(input.x, rect.x), rect.x + rect.width),
        std::min(std::max(input.y, rect.y), rect.y + rect.height)
    );
}

// Visualize an intermediate matrix
static void vismat(const cv::Mat &mat, const std::string &out_prefix, const std::string &name) {
    // If out_prefix is empty, show the image on screen
    // Otherwise, write the image to a file
    if (out_prefix == "") {
        cv::imshow(name, mat);
    }
    else {
        std::stringstream filename;
        filename << out_prefix;
        filename << name;
        filename << ".png";

        cv::Mat out_mat;
        switch (mat.type()) {
        case CV_8UC1:
        case CV_8UC3:
            mat.copyTo(out_mat);
            break;

        case CV_32FC1:
            mat.convertTo(out_mat, CV_8UC1, 255.0);
            break;

        default:
            throw std::exception("not sure how to write matrix type");
        }
        cv::imwrite(filename.str(), out_mat);
    }
}

template <typename T>
void extract_rect(const cv::Mat_<T> &input, cv::Mat_<T> &output, cv::Rect ref_block, cv::Vec2f disp) {
    const int borderType = cv::BORDER_REFLECT;

    output.create(ref_block.height, ref_block.width);
    for (int iy = 0; iy < ref_block.height; iy++) {
        for (int ix = 0; ix < ref_block.width; ix++) {
            float pos_x = ref_block.x + disp[0] + ix;
            float pos_y = ref_block.y + disp[1] + iy;

            int x1 = (int)std::floor(pos_x), x2 = x1 + 1;
            int y1 = (int)std::floor(pos_y), y2 = y1 + 1;
            float x_interp = pos_x - x1,
                y_interp = pos_y - y1;

            x1 = cv::borderInterpolate(x1, input.rows, borderType);
            x2 = cv::borderInterpolate(x2, input.rows, borderType);
            y1 = cv::borderInterpolate(y1, input.rows, borderType);
            y2 = cv::borderInterpolate(y2, input.rows, borderType);


            T row1_value = input(y1, x1) * (1.0f - x_interp) + input(y1, x2) * x_interp;
            T row2_value = input(y2, x1) * (1.0f - x_interp) + input(y2, x2) * x_interp;
            output(iy, ix) = row1_value * (1.0f - y_interp) + row2_value * y_interp;
        }
    }
}

}