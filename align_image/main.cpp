#include <gflags/gflags.h>

#include "imaging/stdafx.h"
#include "imaging/AlignNaive.h"
#include "imaging/HierarchicalAlignment.h"

DEFINE_string(ref, "", "Filename of reference image");
DEFINE_string(alt, "", "Filename of alternate image");
//DEFINE_string(aligned_out_file, "", "Filename of output aligned image");

int main(int argc, char *argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    auto reference_raw = cv::imread(FLAGS_ref, cv::IMREAD_GRAYSCALE);
    auto alternate_raw = cv::imread(FLAGS_alt, cv::IMREAD_GRAYSCALE);

    cv::Mat reference;
    cv::copyMakeBorder(reference_raw, reference, 0, 224, 0, 128, cv::BORDER_CONSTANT, cv::Scalar(0));

    cv::Mat alternate;
    cv::copyMakeBorder(alternate_raw, alternate, 0, 224, 0, 128, cv::BORDER_CONSTANT, cv::Scalar(0));

    cv::Mat displacements;
    int tile_size = imaging_cpu::hierarchical_align<align::NaiveBlockAligner>(reference, alternate, &displacements);
    printf("tile_size = %d, d_width = %d, d_height = %d\n",
        tile_size,
        displacements.cols,
        displacements.rows);

    // Create a reconstructed aligned image based on the displacements
    cv::Mat aligned(reference.rows, reference.cols, CV_8UC1, cv::Scalar(0));
    for (int tile_y = 0; tile_y < aligned.rows / tile_size; tile_y++) {
        for (int tile_x = 0; tile_x < aligned.cols / tile_size; tile_x++) {
            // Copy from alternate at tile location + displacement
            int starting_x = tile_x * tile_size,
                starting_y = tile_y * tile_size;
            auto disp = cv::Vec2i(displacements.at<cv::Vec2f>(tile_y, tile_x));
            auto alt_tile = alternate(cv::Rect(
                starting_x + disp[0], starting_y + disp[1], tile_size, tile_size
            ));
            auto aligned_tile = aligned(cv::Rect(
                starting_x, starting_y, tile_size, tile_size
            ));
            alt_tile.copyTo(aligned_tile);
        }
    }

    cv::imshow("Reference", reference);
    cv::imshow("Aligned", aligned);
    cv::waitKey(0);
    
    return 0;
}