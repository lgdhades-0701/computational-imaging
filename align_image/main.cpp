#include <gflags/gflags.h>

#include "imaging/stdafx.h"
#include "imaging/SuperResPipeline.h"
#include "imaging/HierarchicalAlignment.h"
#include "imaging/RobustnessMask.h"

DEFINE_string(ref, "", "Filename of reference image");
DEFINE_string(alt, "", "Filename of alternate image");
//DEFINE_string(aligned_out_file, "", "Filename of output aligned image");

static int round_up(int input, int divisor) {
    return ((input + divisor - 1) / divisor) * divisor;
}

static void resize_for_alignment(const cv::Mat &input, cv::Mat &output) {
    int min_width = round_up(input.cols, 256);
    int border_w = (min_width - input.cols) / 2;
    int min_height = round_up(input.rows, 256);
    int border_h = (min_height - input.rows) / 2;
    cv::copyMakeBorder(input, output,
        border_h, border_h,
        border_w, border_w,
        cv::BORDER_CONSTANT, cv::Scalar(0)
    );
}

int main(int argc, char *argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    auto reference_raw = cv::imread(FLAGS_ref, cv::IMREAD_GRAYSCALE);
    if (!reference_raw.data) {
        fprintf(stderr, "Couldn't load image \"%s\"\n", FLAGS_ref.c_str());
        exit(1);
    }

    auto alternate_raw = cv::imread(FLAGS_alt, cv::IMREAD_GRAYSCALE);
    if (!alternate_raw.data) {
        fprintf(stderr, "Couldn't load image \"%s\"\n", FLAGS_alt.c_str());
        exit(1);
    }

    cv::Mat reference;
    resize_for_alignment(reference_raw, reference);
    cv::Mat alternate;
    resize_for_alignment(alternate_raw, alternate);
    auto inputs = imaging_cpu::Inputs(&reference, &alternate);

    auto aligned_image = imaging_cpu::compute_alignment(inputs);
    cv::Mat robustness;
    imaging_cpu::calculate_robustness_mask(reference, *aligned_image.aligned, &robustness);

    // TODO: replace with more accurate kernel reconstruction
    cv::Mat merged(reference.rows, reference.cols, CV_8UC1, cv::Scalar(0));
    for (int y = 0; y < merged.rows; y++) {
        for (int x = 0; x < merged.cols; x++) {
            float weight = 1.0f;
            float color_sum = reference.at<uint8_t>(y, x);

            float pixel_robustness = robustness.at<float>(y, x);
            weight += pixel_robustness;
            color_sum += aligned_image.aligned->at<uint8_t>(y, x) * pixel_robustness;

            merged.at<uint8_t>(y, x) = (uint8_t)(color_sum / weight);
        }
    }

    cv::imshow("Reference", reference);
    cv::imshow("Merged", merged);
    cv::waitKey(0);
    
    return 0;
}