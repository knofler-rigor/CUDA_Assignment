**ABOUT THE CODE/PROGRAM**

This CUDA C++ code demonstrates a simple yet powerful way to utilize GPU parallelism for image processing tasks. By converting an RGB image to grayscale using CUDA, 
the program showcases how to set up CUDA kernel functions, manage device memory, and integrate CUDA with OpenCV for efficient image processing.

**ABOUT THE CODE AND THE WORKING**

Kernel Definition: __global__ qualifier indicates that the function runs on the GPU.
Thread Index Calculation: Each thread computes its unique index x and y based on the block and thread indices.
Boundary Check: Ensures threads do not access out-of-bounds memory.
Grayscale Conversion: Each thread computes the grayscale value using the luminance formula and stores it in the output array.
Load Image: cv::imread loads the image into a cv::Mat object.
Check Image: Ensures the image is loaded correctly.
Convert to RGB: If the image is grayscale, it is converted to RGB to ensure compatibility.
Image Dimensions and Channels: Get the dimensions (width and height) and number of channels of the image.
Allocate Host Memory: Allocate memory for the output grayscale image.
Allocate Device Memory: Use cudaMalloc to allocate memory on the GPU for the input and output images.
Copy Data to Device: Use cudaMemcpy to transfer image data from the host to the device.
Define Execution Configuration: dim3 is used to define the block size (16x16 threads per block) and grid size.
Launch Kernel: Call the kernel with the execution configuration.
Copy Data Back to Host: Transfer the processed image data back from the device to the host.
Free Device Memory: Release the allocated GPU memory.
Save Processed Image: Use cv::imwrite to save the grayscale image.
Display Images: Use cv::imshow to display the original and grayscale images. cv::waitKey waits for a key press before closing the windows.
