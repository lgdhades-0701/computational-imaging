
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <opencv2/opencv.hpp>

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}

cv::Mat disparity_opencv(const cv::Mat &left_image, const cv::Mat &right_image) {
    cv::Mat disparity(left_image.rows, left_image.cols, CV_16U);
    cv::Ptr<cv::StereoBM> stereo_bm =
        cv::StereoBM::create(0 /* num_disparities */, 21 /* block_size */);
    stereo_bm->compute(left_image, right_image, disparity);

    cv::Mat disparity_8bit(left_image.rows, left_image.cols, CV_8U);
    disparity.convertTo(disparity_8bit, CV_8U, 0.2, 0.0);
    return disparity_8bit;
}

float match_block_cpu(int num_pixels,
    const cv::Mat &left_image, int left_x, int left_y,
    const cv::Mat &right_image, int right_x, int right_y) {

    const uint8_t *left_ptr = left_image.ptr(left_y) + left_x;
    const uint8_t *right_ptr = right_image.ptr(right_y) + right_x;

    // Use the SSD cost for now
    // TODO: try out the normalized cross-correlation
    float cost = 0.0f;
    for (int x = 0; x < num_pixels; x++) {
        float diff = (float)left_ptr[x] - right_ptr[x];
        cost += diff * diff;
    }

    return cost;
}

cv::Mat disparity_cpu(const cv::Mat &left_image, const cv::Mat &right_image) {
    cv::Mat output(left_image.rows, left_image.cols, CV_8U);

    const int HALF_BLOCK_SIZE = 10;

    for (int y = 0; y < left_image.rows; y++) {
        // Ignore the left/right borders for now
        for (int x = HALF_BLOCK_SIZE; x < left_image.cols; x++) {
            // Find best matching block in right image
            auto best_d = std::make_tuple(0, std::numeric_limits<float>::infinity());

            for (int potential_d = 0; potential_d < 32; potential_d++) {
                if (x + potential_d + HALF_BLOCK_SIZE >= left_image.cols) {
                    break;
                }
                int block_start = x - HALF_BLOCK_SIZE;
                float cost = match_block_cpu(HALF_BLOCK_SIZE * 2 + 1,
                    left_image, block_start, y,
                    right_image, block_start + potential_d, y);
                cost += potential_d * 0.1f; // Bias toward smaller disparity

                if (cost < std::get<1>(best_d)) {
                    best_d = std::make_tuple(potential_d, cost);
                }
            }

            output.at<uint8_t>(y, x) = std::get<0>(best_d);
        }
    }

    cv::Mat scaled;
    output.convertTo(scaled, -1, 6.0, 0.0);
    return scaled;
}

int main()
{
    /*const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }*/

    // Try out OpenCV
    //cv::Mat left_image = cv::imread("tsukuba/scene1.row3.col1.png", cv::IMREAD_GRAYSCALE);
    cv::Mat left_image = cv::imread("easy_stereo/left.png", cv::IMREAD_GRAYSCALE);
    if (left_image.empty()) {
        throw std::runtime_error("Couldn't load image (left_image)");
    }

    //cv::Mat right_image = cv::imread("tsukuba/scene1.row3.col5.png", cv::IMREAD_GRAYSCALE);
    cv::Mat right_image = cv::imread("easy_stereo/right.png", cv::IMREAD_GRAYSCALE);
    if (right_image.empty()) {
        throw std::runtime_error("Couldn't load image (right_image)");
    }

    /*cv::imshow("Left image", left_image);
    cv::waitKey(0);
    cv::imshow("Right image", right_image);
    cv::waitKey(0);*/

    // Compute stereo disparity
    cv::Mat disparity =
        disparity_cpu(left_image, right_image);
    cv::imshow("Disparity", disparity);
    cv::waitKey(0);

    cv::destroyAllWindows();

    return 0;
}