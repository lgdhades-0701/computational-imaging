#include <gflags/gflags.h>

#include "imaging/stdafx.h"
#include "imaging/AlignNaive.h"
#include "imaging/HierarchicalAlignment.h"

DEFINE_string(ref, "", "Filename of reference image");
DEFINE_string(alt, "", "Filename of alternate image");
//DEFINE_string(aligned_out_file, "", "Filename of output aligned image");

int main(int argc, char *argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    auto reference = cv::imread(FLAGS_ref, cv::IMREAD_GRAYSCALE);
    auto alternate = cv::imread(FLAGS_alt, cv::IMREAD_GRAYSCALE);

    cv::Mat displacements;
    int tile_size = imaging_cpu::hierarchical_align<align::NaiveBlockAligner>(reference, alternate, &displacements);
    (void)tile_size;

    // Create a reconstructed aligned image based on the displacements
    cv::Mat aligned(reference.rows, reference.cols, CV_8UC1, cv::Scalar(0));

    cv::imshow("Aligned", aligned);
    cv::waitKey(0);
    
    return 0;
}