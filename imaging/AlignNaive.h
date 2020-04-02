#pragma once

namespace align {

class Disp {
public:
    float x, y;
    Disp(float x, float y) : x(x), y(y) {}
};

typedef Disp(*align_func)(const cv::Mat &, const cv::Mat &, int, int);

template <int Norm>
static Disp align_one_naive(
    const cv::Mat &reference_tile,
    const cv::Mat &unaligned_neighborhood,
    int zero_disp_x,
    int zero_disp_y
) {
    // Return the displacement that minimizes the distance measure
    auto best = std::make_tuple(std::numeric_limits<float>::infinity(), 0, 0);

    for (int disp_y = 0; disp_y < unaligned_neighborhood.rows - reference_tile.rows + 1; disp_y++) {
        const uint8_t *unaligned_row_ptr = unaligned_neighborhood.ptr(disp_y);
        for (int disp_x = 0; disp_x < unaligned_neighborhood.cols - reference_tile.cols + 1; disp_x++) {

            // Compute distance measure for this (disp_x, disp_y)
            float distance = 0.0f;
            for (int y = 0; y < reference_tile.rows; y++) {
                const uint8_t *tile_row_ptr = reference_tile.ptr(y);
                for (int x = 0; x < reference_tile.cols; x++) {
                    float d_pixel = (float)tile_row_ptr[x] -
                        unaligned_row_ptr[disp_x + x];
                    if (Norm == 2) {
                        d_pixel *= d_pixel;
                    }
                    distance += d_pixel;
                }
            }

            // See if this is the best one so far
            if (distance <= std::get<0>(best)) {
                best = std::make_tuple(distance, disp_y, disp_x);
            }
        }
    }

    // Adjust the raw displacement to the center of the neighborhood
    return Disp(
        (float)std::get<1>(best) - zero_disp_x,
        (float)std::get<2>(best) - zero_disp_y
    );
}

template <align_func AlignF>
static void block_align_naive(
    int tile_size,
    int search_radius,
    const cv::Mat &reference,
    const cv::Mat &unaligned,
    cv::Mat *out
) {
    out->create(reference.rows / tile_size, reference.cols / tile_size, CV_32FC2);
    for (int tile_y = 0; tile_y < out->rows; tile_y++) {
        float *out_row_ptr = reinterpret_cast<float*>(out->ptr(tile_y));
        for (int tile_x = 0; tile_x < out->cols; tile_x++) {
            cv::Mat reference_tile(reference, cv::Rect(
                tile_x * tile_size,
                tile_y * tile_size,
                tile_size,
                tile_size
            ));
            cv::Mat unaligned_neighborhood(unaligned, cv::Rect(
                cv::Point(
                    std::max(0, tile_x * tile_size - search_radius),
                    std::max(0, tile_y * tile_size - search_radius)
                ),
                cv::Point(
                    std::min(unaligned.cols - 1, (tile_x + 1) * tile_size + search_radius),
                    std::min(unaligned.rows - 1, (tile_y + 1) * tile_size + search_radius)
                )
            ));

            auto disp = AlignF(
                reference_tile,
                unaligned_neighborhood,
                search_radius,
                search_radius
            );
            out_row_ptr[tile_x * 2] = disp.x;
            out_row_ptr[tile_x * 2 + 1] = disp.y;
        }
    }
}

}