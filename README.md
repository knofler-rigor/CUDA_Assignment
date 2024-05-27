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

**HOW THE CODE WORKS**

The CUDA C++ code orchestrates the conversion of an input image from RGB to grayscale utilizing the parallel processing capabilities of the GPU. Upon loading the image via OpenCV, the code verifies its format, ensuring consistency by converting grayscale images to RGB. Subsequently, memory allocation occurs on the GPU for both input and output images, followed by data transfer from the CPU to the GPU. A CUDA kernel function, rgbToGrayscale, is then invoked to process each pixel of the image concurrently, employing a luminance formula to compute grayscale values from RGB components. Post-processing, the resulting grayscale image is transferred back to the CPU, and GPU memory is deallocated. Finally, utilizing OpenCV, the grayscale image is saved to disk, and both the original and grayscale images are displayed for visualization. Through these orchestrated steps, the code harnesses GPU parallelism to expedite the conversion process, offering a performance boost over traditional CPU-based implementations.

**GOING INTO THE EXECUTION OF THE CODE:**

This provides the image that was suppose to be the input for the file processing
![image](https://github.com/knofler-rigor/CUDA_Assignment/assets/76225148/93d98106-88f0-43ea-b2c6-31cfb7ad4f65)
This provides the required output for the file processing:
![image](https://github.com/knofler-rigor/CUDA_Assignment/assets/76225148/e568f5dc-8f9d-49da-94e4-ec1c6f8afa5e)
