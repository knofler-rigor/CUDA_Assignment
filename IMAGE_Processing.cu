#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

// Kernel function to convert RGB image to grayscale
__global__ void rgbToGrayscale(unsigned char* d_input, unsigned char* d_output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int rgbOffset = (y * width + x) * 3;
        int grayOffset = y * width + x;
        unsigned char r = d_input[rgbOffset];
        unsigned char g = d_input[rgbOffset + 1];
        unsigned char b = d_input[rgbOffset + 2];

        // Convert RGB to grayscale using the luminance formula
        d_output[grayOffset] = 0.299f * r + 0.587f * g + 0.114f * b;
    }
}

int main() {
    // Load the image using OpenCV
    cv::Mat img = cv::imread("image.jpg");
    if (img.empty()) {
        std::cerr << "Error: Could not open image." << std::endl;
        return -1;
    }

    // Convert the image to RGB format if it's not already
    if (img.channels() == 1) {
        cv::cvtColor(img, img, cv::COLOR_GRAY2RGB);
    }

    int width = img.cols;
    int height = img.rows;
    int channels = img.channels();
    
    // Allocate host memory for the output grayscale image
    cv::Mat gray_img(height, width, CV_8UC1);

    // Allocate device memory
    unsigned char* d_input;
    unsigned char* d_output;
    size_t input_size = width * height * channels * sizeof(unsigned char);
    size_t output_size = width * height * sizeof(unsigned char);

    cudaMalloc(&d_input, input_size);
    cudaMalloc(&d_output, output_size);

    // Copy the input image to the device
    cudaMemcpy(d_input, img.data, input_size, cudaMemcpyHostToDevice);

    // Define block and grid sizes
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // Launch the kernel
    rgbToGrayscale<<<gridSize, blockSize>>>(d_input, d_output, width, height);

    // Copy the result back to the host
    cudaMemcpy(gray_img.data, d_output, output_size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);

    // Save the processed image
    cv::imwrite("gray_image.jpg", gray_img);

    // Display the original and processed images
    cv::imshow("Original Image", img);
    cv::imshow("Grayscale Image", gray_img);
    cv::waitKey(0);

    return 0;
}
