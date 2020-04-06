#pragma once

namespace util {

// Clamp a point inside a rectangle
static cv::Point clamp_point(const cv::Point &input, const cv::Rect &rect) {
    return cv::Point(
        std::min(std::max(input.x, rect.x), rect.x + rect.width),
        std::min(std::max(input.y, rect.y), rect.y + rect.height)
    );
}

}