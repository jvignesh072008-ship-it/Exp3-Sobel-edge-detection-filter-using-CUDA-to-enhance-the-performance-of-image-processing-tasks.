# Exp3-Sobel-edge-detection-filter-using-CUDA-to-enhance-the-performance-of-image-processing-tasks.

### AIM:
To implement the Sobel edge detection filter on the GPU using CUDA, handling color images (grayscale conversion and boundary checks inside the kernel), and to analyse its performance for different image sizes and block sizes and compare it with a CPU-based Sobel filter implemented using OpenCV.

### NAME: Vignesh J
### REGISTER NO: 212225230297
### EX. NO: 3
### DATE: 23-11-2026

# Sobel edge detection filter using CUDA

Implement Sobel edge detection filter using GPU.

## EXPERIMENT DETAILS:

The Sobel operator is a popular edge detection method that computes the gradient of the image intensity at each pixel. It uses convolution with two 3x3 kernels to determine the gradient in the x and y directions:

```
Gx = [-1  0  1]      Gy = [-1 -2 -1]
     [-2  0  2]           [ 0  0  0]
     [-1  0  1]           [ 1  2  1]

Edge strength = sqrt(Gx^2 + Gy^2)
```

This lab uses CUDA to parallelize the Sobel filter. Each GPU thread computes one output pixel. The code reads an input image, applies the Sobel filter in parallel on the GPU, and writes the result to an output image.

## EQUIPMENTS REQUIRED:
- Google Colab with NVIDIA T4 GPU and NVCC compiler
- CUDA Toolkit
- OpenCV 4.6.0 (C++ for the CUDA program, Python `cv2` for the CPU comparison)
- A sample color image for testing (3840x2160)

## PROCEDURE:

**Tasks**

a. **Modify the kernel:** The kernel converts the color (BGR) image to grayscale and then applies the Sobel filter. Boundary checks avoid reading out of bounds for pixels on the image edges.

b. **Performance analysis:** Measure the execution time of the Sobel filter for 256x256, 512x512 and 1024x1024 images, and analyse how the block size (8x8, 16x16, 32x32) affects execution time and output quality.

c. **Comparison:** Compare the CUDA output with a CPU-based Sobel filter implemented with OpenCV, and discuss the differences in execution time and output quality.

**Steps followed**
1. Set the Colab runtime to T4 GPU and check it with `nvidia-smi` and `nvcc --version`.
2. Install the OpenCV development files and upload the test image as `/content/images.jpeg`.
3. Write the CUDA program with `%%writefile`, compile it with `nvcc`, and run it.
4. Run the program for each image size and block size, read the kernel time, and time the OpenCV CPU Sobel for the same image.
5. Compare the CUDA and CPU outputs pixel by pixel.

**Changes made to the starter code**
- Kernel: added RGB to grayscale conversion, the Sobel Gx/Gy computation, gradient magnitude with clamping to 255, and boundary handling (out-of-range neighbours are mirrored back into the image with `reflectIdx()`, and threads outside the image return immediately).
- Fixed the buffer sizes: the color input needs `width*height*3` bytes and the output `width*height` bytes.
- Grid size now uses `(width + B - 1) / B`, so any block size works.
- Added command-line arguments for image size and block size.
- Timing uses a warm-up launch and the average of 100 launches (CUDA events).
- Output is saved as PNG (lossless) so that the comparison with the CPU result is not affected by JPEG compression.

## PROGRAM:

### Setup (Colab cells)
```
!nvidia-smi
!nvcc --version
!apt-get update -qq
!apt-get install -y libopencv-dev pkg-config
!pkg-config --modversion opencv4
```
```python
from google.colab import files
import os
uploaded = files.upload()
os.rename(list(uploaded.keys())[0], '/content/images.jpeg')
```

### CUDA program (`sobelEdgeDetectionFilter.cu`)
```cpp
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

using namespace cv;

// Boundary check: mirror an out-of-range index back into [0, n-1]
__device__ __forceinline__ int reflectIdx(int i, int n) {
    if (i < 0)  i = -i;
    if (i >= n) i = 2 * n - 2 - i;
    return i;
}

// Grayscale value of one pixel from an interleaved BGR image
__device__ __forceinline__ float grayAt(const unsigned char *bgr, int x, int y,
                                        int w, int h) {
    x = reflectIdx(x, w);
    y = reflectIdx(y, h);
    const unsigned char *p = bgr + (y * w + x) * 3;
    // OpenCV's BGR2GRAY weights, rounded to 8 bit
    return (float)(unsigned char)(0.114f * p[0] + 0.587f * p[1] + 0.299f * p[2] + 0.5f);
}

__global__ void sobelFilter(unsigned char *srcImage, unsigned char *dstImage,
                            unsigned int width, unsigned int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int w = (int)width, h = (int)height;
    if (x >= w || y >= h) return;               // outside the image

    float a  = grayAt(srcImage, x - 1, y - 1, w, h);
    float b  = grayAt(srcImage, x,     y - 1, w, h);
    float c  = grayAt(srcImage, x + 1, y - 1, w, h);
    float d  = grayAt(srcImage, x - 1, y,     w, h);
    float f  = grayAt(srcImage, x + 1, y,     w, h);
    float g  = grayAt(srcImage, x - 1, y + 1, w, h);
    float hh = grayAt(srcImage, x,     y + 1, w, h);
    float i  = grayAt(srcImage, x + 1, y + 1, w, h);

    // Gx = [-1 0 1; -2 0 2; -1 0 1]    Gy = [-1 -2 -1; 0 0 0; 1 2 1]
    float gx = -a + c - 2.f * d + 2.f * f - g + i;
    float gy = -a - 2.f * b - c + g + 2.f * hh + i;

    float mag = sqrtf(gx * gx + gy * gy);
    dstImage[y * w + x] = (unsigned char)fminf(mag + 0.5f, 255.f);
}

void checkCudaErrors(cudaError_t r) {
    if (r != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(r));
        exit(EXIT_FAILURE);
    }
}

// Usage: ./sobelEdgeDetectionFilter [size] [blockSize]
//   size      : resize image to size x size (0 or omitted = original size)
//   blockSize : 8, 16 or 32 (default 16)
int main(int argc, char **argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 0;
    int B = (argc > 2) ? atoi(argv[2]) : 16;

    // Read input image (color, BGR)
    Mat image = imread("/content/images.jpeg", IMREAD_COLOR);
    if (image.empty()) {
        printf("Error: Image not found.\n");
        return -1;
    }
    if (N > 0) resize(image, image, Size(N, N));

    int width = image.cols;
    int height = image.rows;
    size_t inputSize  = (size_t)width * height * 3;   // 3 channels in
    size_t outputSize = (size_t)width * height;       // 1 channel out

    unsigned char *h_outputImage = (unsigned char *)malloc(outputSize);
    if (h_outputImage == nullptr) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return -1;
    }

    unsigned char *d_inputImage, *d_outputImage;
    checkCudaErrors(cudaMalloc(&d_inputImage, inputSize));
    checkCudaErrors(cudaMalloc(&d_outputImage, outputSize));
    checkCudaErrors(cudaMemcpy(d_inputImage, image.data, inputSize, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dim3 blockSize(B, B);
    dim3 gridSize((width + B - 1) / B, (height + B - 1) / B);

    // Warm-up launch (first launch includes one-time startup cost)
    sobelFilter<<<gridSize, blockSize>>>(d_inputImage, d_outputImage, width, height);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Timed launches: average of 100 runs
    const int ITERS = 100;
    cudaEventRecord(start);
    for (int k = 0; k < ITERS; k++)
        sobelFilter<<<gridSize, blockSize>>>(d_inputImage, d_outputImage, width, height);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    milliseconds /= ITERS;

    checkCudaErrors(cudaMemcpy(h_outputImage, d_outputImage, outputSize, cudaMemcpyDeviceToHost));

    // Save as PNG (lossless) so the CPU comparison is not affected by JPEG compression
    Mat outputImage(height, width, CV_8UC1, h_outputImage);
    std::string name = "output_sobel_" + std::to_string(width) + "_" + std::to_string(B) + ".png";
    imwrite(name, outputImage);

    free(h_outputImage);
    cudaFree(d_inputImage);
    cudaFree(d_outputImage);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("Image %dx%d, block %dx%d\n", width, height, B, B);
    printf("KERNEL_MS=%f\n", milliseconds);
    printf("Total time taken: %f milliseconds\n", milliseconds);
    return 0;
}
```

### Compile and run
```
!nvcc -diag-suppress 611 -o sobelEdgeDetectionFilter sobelEdgeDetectionFilter.cu `pkg-config --cflags --libs opencv4`
!./sobelEdgeDetectionFilter
```

### Display the input and the CUDA output
```python
import cv2
from matplotlib import pyplot as plt

out = cv2.imread('/content/output_sobel_3840_16.png', cv2.IMREAD_GRAYSCALE)
inp = cv2.cvtColor(cv2.imread('/content/images.jpeg'), cv2.COLOR_BGR2RGB)

fig, ax = plt.subplots(1, 2, figsize=(16, 5))
ax[0].imshow(inp); ax[0].set_title('Input image'); ax[0].axis('off')
ax[1].imshow(out, cmap='gray'); ax[1].set_title('CUDA Sobel edge detection output'); ax[1].axis('off')
plt.show()
```

### Performance analysis (image sizes, block sizes, CPU timing)
```python
import subprocess, re, time, cv2, numpy as np, pandas as pd
import matplotlib.pyplot as plt

img = cv2.imread('/content/images.jpeg')

def cpu_sobel(im):
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return np.clip(cv2.magnitude(gx, gy), 0, 255).round().astype(np.uint8)

rows = []
for N in [256, 512, 1024]:
    small = cv2.resize(img, (N, N))
    cpu_out = cpu_sobel(small)
    t = time.perf_counter()
    for _ in range(20):
        cpu_sobel(small)
    cpu_ms = (time.perf_counter() - t) / 20 * 1000

    for B in [8, 16, 32]:
        out = subprocess.run(['./sobelEdgeDetectionFilter', str(N), str(B)],
                             capture_output=True, text=True).stdout
        gpu_ms = float(re.search(r'KERNEL_MS=([\d.]+)', out).group(1))
        gpu_out = cv2.imread(f'output_sobel_{N}_{B}.png', cv2.IMREAD_GRAYSCALE)
        d = cv2.absdiff(gpu_out, cpu_out)
        rows.append(dict(size=N, block=f'{B}x{B}', gpu_ms=gpu_ms, cpu_ms=cpu_ms,
                         speedup=cpu_ms / gpu_ms, mean_diff=d.mean(), max_diff=d.max()))

df = pd.DataFrame(rows)
display(df)

for blk, g in df.groupby('block'):
    plt.plot(g['size'], g['gpu_ms'], marker='o', label=f'GPU {blk}')
plt.plot(df['size'].unique(), df.groupby('size')['cpu_ms'].first(), 'k--', marker='s', label='CPU (OpenCV)')
plt.xlabel('Image size (N x N)'); plt.ylabel('Time (ms)'); plt.legend(); plt.grid(True)
plt.title('Sobel execution time'); plt.show()
```

### CUDA vs CPU (OpenCV) comparison
```python
import cv2, numpy as np
from matplotlib import pyplot as plt

out = cv2.imread('/content/output_sobel_3840_16.png', cv2.IMREAD_GRAYSCALE)
img = cv2.imread('/content/images.jpeg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
cpu = np.clip(cv2.magnitude(gx, gy), 0, 255).round().astype(np.uint8)

diff = cv2.absdiff(out, cpu)
print("Mean abs difference :", diff.mean())
print("Max difference      :", diff.max())
print("Pixels that differ  : %.4f %%" % (100 * np.count_nonzero(diff) / diff.size))

bright = lambda a, k=3: np.clip(a.astype(np.float32) * k, 0, 255).astype(np.uint8)
fig, ax = plt.subplots(2, 2, figsize=(16, 9))
ax[0][0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); ax[0][0].set_title('Input image')
ax[0][1].imshow(bright(out), cmap='gray'); ax[0][1].set_title('CUDA Sobel (brightened x3)')
ax[1][0].imshow(bright(cpu), cmap='gray'); ax[1][0].set_title('CPU Sobel, OpenCV (brightened x3)')
ax[1][1].imshow(bright(diff, 50), cmap='gray'); ax[1][1].set_title('|CUDA - CPU| (amplified x50)')
for a in ax.ravel(): a.axis('off')
plt.tight_layout(); plt.show()
```

## OUTPUT:

### 1. Compilation, execution and CUDA Sobel output (3840x2160 image, 16x16 blocks)
The program runs on the GPU and takes about 1.14 ms per Sobel pass on the full 3840x2160 image. The edge map shows the outlines of the hair, face, eyes, hand and cloak, while the smooth background stays dark.

![Run output and CUDA Sobel result](output1_run_and_cuda.png)

### 2. Performance analysis: image size and block size
Kernel time is the average of 100 launches (GPU) and the average of 20 runs (CPU, including the grayscale conversion). GPU time excludes host-to-device and device-to-host copies.

![Timing table and graph](output2_timing_table_graph.png)

| Image size | Block | GPU kernel (ms) | CPU OpenCV (ms) | Speedup | Mean diff | Max diff |
|-----------:|:-----:|----------------:|----------------:|--------:|----------:|---------:|
| 256x256   | 8x8   | 0.018821 | 0.268956 | 14.3x | 0.00177 | 2 |
| 256x256   | 16x16 | 0.010957 | 0.268956 | 24.5x | 0.00177 | 2 |
| 256x256   | 32x32 | 0.013230 | 0.268956 | 20.3x | 0.00177 | 2 |
| 512x512   | 8x8   | 0.053658 | 1.934521 | 36.1x | 0.00280 | 3 |
| 512x512   | 16x16 | 0.027716 | 1.934521 | 69.8x | 0.00280 | 3 |
| 512x512   | 32x32 | 0.035917 | 1.934521 | 53.9x | 0.00280 | 3 |
| 1024x1024 | 8x8   | 0.165543 | 6.533671 | 39.5x | 0.00302 | 4 |
| 1024x1024 | 16x16 | 0.081924 | 6.533671 | 79.8x | 0.00302 | 4 |
| 1024x1024 | 32x32 | 0.108810 | 6.533671 | 60.0x | 0.00302 | 4 |

### 3. Comparison of CUDA output with the CPU (OpenCV) output (3840x2160)
- Mean absolute difference: 9.65e-07 gray levels
- Max difference: 2
- Pixels that differ: 0.0001 %

The CUDA and CPU edge maps look identical, and the difference image (amplified 50 times) is completely black.

![CUDA vs CPU comparison](output3_cuda_vs_cpu.png)

## RESULT:
Thus the program has been executed by using CUDA to perform Sobel edge detection on a color image, with the grayscale conversion and the Sobel filter done on the GPU. The GPU kernel was 14x to 80x faster than the OpenCV CPU implementation (kernel time only), the 16x16 block size gave the best performance, and the CUDA output matched the CPU output to within a mean difference of less than 0.004 gray levels.

## QUESTIONS AND ANSWERS:

**1. What challenges did you face while implementing the Sobel filter for color images?**
- OpenCV loads color images in BGR order, so the grayscale weights had to be applied as 0.114B + 0.587G + 0.299R, the same as OpenCV, to get matching results.
- The color input has 3 bytes per pixel, so the input buffer is three times the size of the output buffer. Using the single-channel size for both copies reads and writes the wrong amount of memory.
- Pixels on the image border have missing neighbours. These reads had to be handled (mirrored back into the image) so that no thread reads out of bounds, and so that the borders match OpenCV's default behaviour.
- Computing the grayscale value inside the kernel means each pixel is converted up to 9 times (once for each 3x3 neighbourhood that contains it), which is extra work.
- The gradient magnitude has to be rounded and clamped to 0-255 before it is stored as an 8-bit pixel.

**2. How did changing the block size influence the performance of your CUDA implementation?**
- 16x16 was the fastest at every image size: 0.0110 ms (256), 0.0277 ms (512) and 0.0819 ms (1024).
- 32x32 was next (0.0132, 0.0359 and 0.1088 ms), and 8x8 was the slowest (0.0188, 0.0537 and 0.1655 ms). At 512 and 1024, 8x8 took about twice as long as 16x16.
- A typical explanation: 8x8 blocks have only 64 threads (2 warps), so the GPU cannot keep enough threads resident per multiprocessor. 32x32 blocks use the maximum 1024 threads, which leaves less flexibility for the scheduler and wastes more threads on edge blocks. 16x16 (256 threads) balances the two.
- Block size did not change the output quality: for each image size the mean and max difference are identical for all three block sizes, because each thread computes its own pixel independently.
- For the 256x256 image the kernel takes only about 0.01 ms, so launch overhead dominates and the differences between block sizes are small and noisy.
- Kernel time grows with image size, and the speedup over the CPU also grows (14-24x at 256x256, up to about 80x at 1024x1024) because larger images keep more of the GPU busy.

**3. What were the differences in output between the CUDA and CPU implementations? Discuss any discrepancies.**
- On the full 3840x2160 image the outputs are practically identical: mean absolute difference 9.65e-07, maximum difference 2 gray levels, and only 0.0001 % of pixels differ. The difference image is black even when amplified 50 times, and there are no differences along the image borders, so the boundary handling matches OpenCV.
- On the resized images the mean difference is 0.0018 to 0.0030 and the maximum is 2 to 4 gray levels, which is not visible to the eye. The likely causes are floating-point rounding (the square root and the rounding of values that fall exactly between two integers) and, for the resized images, the fact that the CUDA program resizes with the C++ OpenCV library while the CPU reference uses the Python `cv2` build, which can produce slightly different resized pixels.
- Execution time differs a lot: the GPU kernel takes 0.011 to 0.166 ms, while the CPU takes 0.27 to 6.53 ms. These are not a perfectly fair comparison, because the GPU time is kernel only (no data transfer to and from the GPU), while the CPU time includes the grayscale conversion. With the transfers included, the speedup would be smaller, especially for small images.

**4. Suggest potential optimizations for improving the performance of the Sobel filter.**
- Use shared memory: load a tile of the image plus a 1-pixel border (halo) into shared memory once, so each pixel is read from global memory once instead of up to 9 times.
- Convert the image to grayscale once (in a separate kernel or while loading the tile) instead of recomputing the grayscale of every neighbour.
- Improve memory access with coalesced and vectorized loads (for example `uchar4`), and mark pointers `const __restrict__`.
- Use the separable form of the Sobel kernels (a smoothing [1 2 1] pass and a difference [-1 0 1] pass) to reduce the number of operations per pixel.
- Process several pixels per thread to reduce per-thread overhead.
- Use pinned host memory and CUDA streams to overlap data transfers with kernel execution, since for small images the copies cost more than the kernel.
- Use a faster magnitude, for example |Gx| + |Gy| or a fast square root intrinsic, if exact Euclidean magnitude is not required.

## DELIVERABLES:
- Modified CUDA code with comments explaining the changes (above)
- Report with timing table, graph and comparison of outputs (Output section)
- Answers to the questions (above)
