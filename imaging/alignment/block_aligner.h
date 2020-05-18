#pragma once

namespace align {

class Disp {
public:
    float x, y;
    Disp(float x, float y) : x(x), y(y) {}
    bool operator ==(const Disp &o) const {
        return x == o.x && y == o.y;
    }
    // for gtest
    friend void PrintTo(const Disp &disp, std::ostream *os) {
        *os << "Disp(" << disp.x << ", " << disp.y << ")";
    }
};

class BlockAligner {
public:
    virtual int tile_size() const = 0;
    virtual float disp_residual_L1(int ref_tile_x, int ref_tile_y, float disp_x, float disp_y) const = 0;
    virtual void align_L2(cv::Mat *out) = 0;
};

}