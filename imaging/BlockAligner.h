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
    virtual float tile_residual_L1(float ref_tile_x, float ref_tile_y, float alt_tile_x, float alt_tile_y) const = 0;
    virtual void align_L2(cv::Mat *out) = 0;
};

}