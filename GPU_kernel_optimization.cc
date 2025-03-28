/**
 * CUDA Kernels for Image Processing with Optimizations
 * 
 * This file contains optimized CUDA kernels for various image processing operations:
 * - RGB to Grayscale conversion
 * - Gaussian Blur
 * - Sobel Edge Detection
 * - Image Sharpening
 * - Fused operations for better performance
 */

 #include <cuda_runtime.h>

 // Constants for shared memory optimization
 #define BLOCK_SIZE 16
 #define FILTER_RADIUS 2  // For 5x5 filters
 #define TILE_SIZE (BLOCK_SIZE + 2 * FILTER_RADIUS)
 
 // Simple RGB to grayscale conversion
 __global__ void rgbToGrayscaleKernel(
     const unsigned char* inputImage,
     unsigned char* outputImage,
     int width, 
     int height)
 {
     // Calculate pixel coordinates
     int x = blockIdx.x * blockDim.x + threadIdx.x;
     int y = blockIdx.y * blockDim.y + threadIdx.y;
     
     // Check if within image bounds
     if (x < width && y < height) {
         // Calculate input/output indices
         int outputIdx = y * width + x;
         int inputIdx = outputIdx * 3; // 3 channels for RGB
         
         // Standard RGB to grayscale conversion weights (ITU-R BT.601)
         float r = inputImage[inputIdx];
         float g = inputImage[inputIdx + 1];
         float b = inputImage[inputIdx + 2];
         
         // Calculate grayscale
         outputImage[outputIdx] = static_cast<unsigned char>(0.299f * r + 0.587f * g + 0.114f * b);
     }
 }
 
 // Optimized RGB to grayscale with memory coalescing
 __global__ void rgbToGrayscaleOptimizedKernel(
     const unsigned char* inputImage,
     unsigned char* outputImage,
     int width, 
     int height,
     int pitch)
 {
     // Calculate pixel coordinates
     int x = blockIdx.x * blockDim.x + threadIdx.x;
     int y = blockIdx.y * blockDim.y + threadIdx.y;
     
     // Check if within image bounds
     if (x < width && y < height) {
         // Calculate input/output indices with proper memory alignment
         int outputIdx = y * pitch + x;
         int inputIdx = y * pitch * 3 + x * 3; // 3 channels for RGB
         
         // Standard RGB to grayscale conversion weights
         float r = inputImage[inputIdx];
         float g = inputImage[inputIdx + 1];
         float b = inputImage[inputIdx + 2];
         
         // Calculate grayscale
         outputImage[outputIdx] = static_cast<unsigned char>(0.299f * r + 0.587f * g + 0.114f * b);
     }
 }
 
 // Generate Gaussian filter kernel
 __global__ void generateGaussianKernel(
     float* filter, 
     int filterWidth, 
     float sigma)
 {
     int x = blockIdx.x * blockDim.x + threadIdx.x;
     int y = blockIdx.y * blockDim.y + threadIdx.y;
     
     if (x < filterWidth && y < filterWidth) {
         int filterRadius = filterWidth / 2;
         float dx = x - filterRadius;
         float dy = y - filterRadius;
         
         // Gaussian function
         filter[y * filterWidth + x] = expf(-(dx*dx + dy*dy) / (2.0f * sigma * sigma));
     }
 }
 
 // Gaussian Blur with Shared Memory Optimization
 __global__ void gaussianBlurSharedKernel(
     const unsigned char* inputImage,
     unsigned char* outputImage,
     int width, 
     int height,
     float sigma,
     const float* filter,
     int filterWidth)
 {
     extern __shared__ unsigned char sharedMem[];
     
     // Calculate pixel coordinates
     int tx = threadIdx.x;
     int ty = threadIdx.y;
     int x = blockIdx.x * blockDim.x + tx;
     int y = blockIdx.y * blockDim.y + ty;
     
     int filterRadius = filterWidth / 2;
     
     // Load data into shared memory including halo cells (apron)
     // This cooperative loading pattern ensures coalesced memory access
     int sharedIdx = ty * (blockDim.x + 2 * filterRadius) + tx;
     
     // Initialize shared memory
     if (x < width && y < height) {
         // Center point
         sharedMem[sharedIdx] = inputImage[y * width + x];
         
         // Load additional halo cells if thread is on the border of the block
         if (tx < filterRadius) {
             // Left halo
             int srcX = max(x - filterRadius, 0);
             sharedMem[sharedIdx - filterRadius] = inputImage[y * width + srcX];
             
             // Right halo
             int srcX2 = min(x + blockDim.x, width - 1);
             sharedMem[sharedIdx + blockDim.x] = inputImage[y * width + srcX2];
         }
         
         if (ty < filterRadius) {
             // Top halo
             int srcY = max(y - filterRadius, 0);
             sharedMem[(ty - filterRadius) * (blockDim.x + 2 * filterRadius) + tx] = 
                 inputImage[srcY * width + x];
             
             // Bottom halo
             int srcY2 = min(y + blockDim.y, height - 1);
             sharedMem[(ty + blockDim.y) * (blockDim.x + 2 * filterRadius) + tx] = 
                 inputImage[srcY2 * width + x];
         }
     }
     
     // Ensure all threads have finished loading shared memory
     __syncthreads();
     
     // Perform Gaussian blur from shared memory
     if (x < width && y < height) {
         float sum = 0.0f;
         float filterSum = 0.0f;
         
         // Apply filter
         for (int fy = -filterRadius; fy <= filterRadius; fy++) {
             for (int fx = -filterRadius; fx <= filterRadius; fx++) {
                 int sharedX = tx + fx + filterRadius;
                 int sharedY = ty + fy + filterRadius;
                 
                 int filterIdx = (fy + filterRadius) * filterWidth + (fx + filterRadius);
                 float filterVal = filter[filterIdx];
                 
                 sum += sharedMem[sharedY * (blockDim.x + 2 * filterRadius) + sharedX] * filterVal;
                 filterSum += filterVal;
             }
         }
         
         // Normalize and write output
         outputImage[y * width + x] = static_cast<unsigned char>(sum / filterSum);
     }
 }
 
 // Image sharpening kernel
 __global__ void sharpenKernel(
     const unsigned char* inputImage,
     unsigned char* outputImage,
     int width,
     int height)
 {
     __shared__ unsigned char sharedMem[TILE_SIZE][TILE_SIZE];
     
     // Sharpening kernel
     const int sharpen[3][3] = {
         { 0, -1,  0},
         {-1,  5, -1},
         { 0, -1,  0}
     };
     
     int tx = threadIdx.x;
     int ty = threadIdx.y;
     int x = blockIdx.x * blockDim.x + tx;
     int y = blockIdx.y * blockDim.y + ty;
     
     // Load shared memory including apron region
     int x_shared = tx + 1;  // +1 for apron region
     int y_shared = ty + 1;
     
     // Initialize shared memory
     sharedMem[y_shared][x_shared] = (x < width && y < height) ? 
         inputImage[y * width + x] : 0;
     
     // Load apron region
     if (tx == 0) {
         // Left edge
         sharedMem[y_shared][0] = (x > 0 && y < height) ? 
             inputImage[y * width + (x-1)] : 0;
     }
     else if (tx == blockDim.x - 1 || x == width - 1) {
         // Right edge
         sharedMem[y_shared][x_shared+1] = (x < width-1 && y < height) ? 
             inputImage[y * width + (x+1)] : 0;
     }
     
     if (ty == 0) {
         // Top edge
         sharedMem[0][x_shared] = (x < width && y > 0) ? 
             inputImage[(y-1) * width + x] : 0;
     }
     else if (ty == blockDim.y - 1 || y == height - 1) {
         // Bottom edge
         sharedMem[y_shared+1][x_shared] = (x < width && y < height-1) ? 
             inputImage[(y+1) * width + x] : 0;
     }
     
     // Load corners of apron
     if (tx == 0 && ty == 0) {
         // Top-left
         sharedMem[0][0] = (x > 0 && y > 0) ? 
             inputImage[(y-1) * width + (x-1)] : 0;
     }
     else if (tx == blockDim.x - 1 && ty == 0) {
         // Top-right
         sharedMem[0][x_shared+1] = (x < width-1 && y > 0) ? 
             inputImage[(y-1) * width + (x+1)] : 0;
     }
     else if (tx == 0 && ty == blockDim.y - 1) {
         // Bottom-left
         sharedMem[y_shared+1][0] = (x > 0 && y < height-1) ? 
             inputImage[(y+1) * width + (x-1)] : 0;
     }
     else if (tx == blockDim.x - 1 && ty == blockDim.y - 1) {
         // Bottom-right
         sharedMem[y_shared+1][x_shared+1] = (x < width-1 && y < height-1) ? 
             inputImage[(y+1) * width + (x+1)] : 0;
     }
     
     __syncthreads();
     
     // Apply sharpening filter
     if (x < width && y < height) {
         int sum = 0;
         
         for (int j = -1; j <= 1; j++) {
             for (int i = -1; i <= 1; i++) {
                 int pixel = sharedMem[y_shared + j][x_shared + i];
                 sum += pixel * sharpen[j+1][i+1];
             }
         }
         
         // Clamp to valid range
         outputImage[y * width + x] = static_cast<unsigned char>(min(max(sum, 0), 255));
     }
 }
 
 // Sobel Edge Detection with Shared Memory
 __global__ void sobelEdgeDetectionKernel(
     const unsigned char* inputImage,
     unsigned char* outputImage,
     int width, 
     int height)
 {
     __shared__ unsigned char sharedMem[TILE_SIZE][TILE_SIZE];
     
     // Sobel operators
     const int sobel_x[3][3] = {
         {-1, 0, 1},
         {-2, 0, 2},
         {-1, 0, 1}
     };
     
     const int sobel_y[3][3] = {
         {-1, -2, -1},
         { 0,  0,  0},
         { 1,  2,  1}
     };
     
     int tx = threadIdx.x;
     int ty = threadIdx.y;
     int x = blockIdx.x * blockDim.x + tx;
     int y = blockIdx.y * blockDim.y + ty;
     
     // Load shared memory including apron region
     int x_shared = tx + 1;  // +1 for apron region
     int y_shared = ty + 1;
     
     // Initialize shared memory
     sharedMem[y_shared][x_shared] = (x < width && y < height) ? 
         inputImage[y * width + x] : 0;
     
     // Load apron region
     if (tx == 0) {
         // Left edge
         sharedMem[y_shared][0] = (x > 0 && y < height) ? 
             inputImage[y * width + (x-1)] : 0;
     }
     else if (tx == blockDim.x - 1 || x == width - 1) {
         // Right edge
         sharedMem[y_shared][x_shared+1] = (x < width-1 && y < height) ? 
             inputImage[y * width + (x+1)] : 0;
     }
     
     if (ty == 0) {
         // Top edge
         sharedMem[0][x_shared] = (x < width && y > 0) ? 
             inputImage[(y-1) * width + x] : 0;
     }
     else if (ty == blockDim.y - 1 || y == height - 1) {
         // Bottom edge
         sharedMem[y_shared+1][x_shared] = (x < width && y < height-1) ? 
             inputImage[(y+1) * width + x] : 0;
     }
     
     // Load corners of apron
     if (tx == 0 && ty == 0) {
         // Top-left
         sharedMem[0][0] = (x > 0 && y > 0) ? 
             inputImage[(y-1) * width + (x-1)] : 0;
     }
     else if (tx == blockDim.x - 1 && ty == 0) {
         // Top-right
         sharedMem[0][x_shared+1] = (x < width-1 && y > 0) ? 
             inputImage[(y-1) * width + (x+1)] : 0;
     }
     else if (tx == 0 && ty == blockDim.y - 1) {
         // Bottom-left
         sharedMem[y_shared+1][0] = (x > 0 && y < height-1) ? 
             inputImage[(y+1) * width + (x-1)] : 0;
     }
     else if (tx == blockDim.x - 1 && ty == blockDim.y - 1) {
         // Bottom-right
         sharedMem[y_shared+1][x_shared+1] = (x < width-1 && y < height-1) ? 
             inputImage[(y+1) * width + (x+1)] : 0;
     }
     
     __syncthreads();
     
     // Compute Sobel
     if (x < width && y < height) {
         int gx = 0;
         int gy = 0;
         
         for (int j = -1; j <= 1; j++) {
             for (int i = -1; i <= 1; i++) {
                 int pixel = sharedMem[y_shared + j][x_shared + i];
                 gx += pixel * sobel_x[j+1][i+1];
                 gy += pixel * sobel_y[j+1][i+1];
             }
         }
         
         // Calculate gradient magnitude
         float magnitude = sqrtf(gx*gx + gy*gy);
         
         // Normalize to 0-255
         outputImage[y * width + x] = static_cast<unsigned char>(min(255.0f, magnitude));
     }
 }
 
 // Fused kernel for RGB to Edge Detection (kernel fusion optimization)
 __global__ void fusedRgbToEdgeKernel(
     const unsigned char* inputImage,
     unsigned char* outputImage,
     int width, 
     int height)
 {
     __shared__ unsigned char sharedGray[TILE_SIZE][TILE_SIZE];
     
     // Sobel operators
     const int sobel_x[3][3] = {
         {-1, 0, 1},
         {-2, 0, 2},
         {-1, 0, 1}
     };
     
     const int sobel_y[3][3] = {
         {-1, -2, -1},
         { 0,  0,  0},
         { 1,  2,  1}
     };
     
     int tx = threadIdx.x;
     int ty = threadIdx.y;
     int x = blockIdx.x * blockDim.x + tx;
     int y = blockIdx.y * blockDim.y + ty;
     
     // Load and convert to grayscale in one step
     int x_shared = tx + 1;  // +1 for apron region
     int y_shared = ty + 1;
     
     // Direct RGB to grayscale conversion into shared memory
     if (x < width && y < height) {
         int inputIdx = 3 * (y * width + x);
         float r = inputImage[inputIdx];
         float g = inputImage[inputIdx + 1];
         float b = inputImage[inputIdx + 2];
         
         // Store grayscale in shared memory
         sharedGray[y_shared][x_shared] = static_cast<unsigned char>(0.299f * r + 0.587f * g + 0.114f * b);
     } else {
         sharedGray[y_shared][x_shared] = 0;
     }
     
     // Load grayscale apron region
     if (tx == 0) {
         // Left edge
         if (x > 0 && y < height) {
             int inputIdx = 3 * (y * width + (x-1));
             float r = inputImage[inputIdx];
             float g = inputImage[inputIdx + 1];
             float b = inputImage[inputIdx + 2];
             sharedGray[y_shared][0] = static_cast<unsigned char>(0.299f * r + 0.587f * g + 0.114f * b);
         } else {
             sharedGray[y_shared][0] = 0;
         }
     }
     
     if (tx == blockDim.x - 1 || x == width - 1) {
         // Right edge
         if (x < width-1 && y < height) {
             int inputIdx = 3 * (y * width + (x+1));
             float r = inputImage[inputIdx];
             float g = inputImage[inputIdx + 1];
             float b = inputImage[inputIdx + 2];
             sharedGray[y_shared][x_shared+1] = static_cast<unsigned char>(0.299f * r + 0.587f * g + 0.114f * b);
         } else {
             sharedGray[y_shared][x_shared+1] = 0;
         }
     }
     
     if (ty == 0) {
         // Top edge
         if (x < width && y > 0) {
             int inputIdx = 3 * ((y-1) * width + x);
             float r = inputImage[inputIdx];
             float g = inputImage[inputIdx + 1];
             float b = inputImage[inputIdx + 2];
             sharedGray[0][x_shared] = static_cast<unsigned char>(0.299f * r + 0.587f * g + 0.114f * b);
         } else {
             sharedGray[0][x_shared] = 0;
         }
     }
     
     if (ty == blockDim.y - 1 || y == height - 1) {
         // Bottom edge
         if (x < width && y < height-1) {
             int inputIdx = 3 * ((y+1) * width + x);
             float r = inputImage[inputIdx];
             float g = inputImage[inputIdx + 1];
             float b = inputImage[inputIdx + 2];
             sharedGray[y_shared+1][x_shared] = static_cast<unsigned char>(0.299f * r + 0.587f * g + 0.114f * b);
         } else {
             sharedGray[y_shared+1][x_shared] = 0;
         }
     }
     
     // Also handle corners
     if (tx == 0 && ty == 0) {
         // Top-left corner
         if (x > 0 && y > 0) {
             int inputIdx = 3 * ((y-1) * width + (x-1));
             float r = inputImage[inputIdx];
             float g = inputImage[inputIdx + 1];
             float b = inputImage[inputIdx + 2];
             sharedGray[0][0] = static_cast<unsigned char>(0.299f * r + 0.587f * g + 0.114f * b);
         } else {
             sharedGray[0][0] = 0;
         }
     }
     
     __syncthreads();
     
     // Apply Sobel directly from shared memory
     if (x < width && y < height) {
         int gx = 0;
         int gy = 0;
         
         for (int j = -1; j <= 1; j++) {
             for (int i = -1; i <= 1; i++) {
                 int pixel = sharedGray[y_shared + j][x_shared + i];
                 gx += pixel * sobel_x[j+1][i+1];
                 gy += pixel * sobel_y[j+1][i+1];
             }
         }
         
         // Calculate gradient magnitude
         float magnitude = sqrtf(gx*gx + gy*gy);
         
         // Normalize to 0-255
         outputImage[y * width + x] = static_cast<unsigned char>(min(255.0f, magnitude));
     }
 }
 
 // Fused Blur and Sharpen kernel
 __global__ void fusedBlurSharpenKernel(
     const unsigned char* inputImage,
     unsigned char* outputImage,
     const float* gaussianFilter,
     int width,
     int height,
     int filterWidth)
 {
     extern __shared__ unsigned char sharedMem[];
     
     int tx = threadIdx.x;
     int ty = threadIdx.y;
     int x = blockIdx.x * blockDim.x + tx;
     int y = blockIdx.y * blockDim.y + ty;
     
     int filterRadius = filterWidth / 2;
     int sharedDimX = blockDim.x + 2 * filterRadius;
     
     // First part: Load data for Gaussian blur
     if (x < width && y < height) {
         // Center point
         sharedMem[ty * sharedDimX + tx + filterRadius] = inputImage[y * width + x];
         
         // Load apron region cooperatively
         if (tx < filterRadius) {
             // Left edge
             int srcX = max(x - filterRadius, 0);
             sharedMem[ty * sharedDimX + tx] = inputImage[y * width + srcX];
             
             // Right edge
             if (tx + blockDim.x < sharedDimX) {
                 int srcX2 = min(x + blockDim.x, width - 1);
                 sharedMem[ty * sharedDimX + tx + blockDim.x + filterRadius] = inputImage[y * width + srcX2];
             }
         }
         
         if (ty < filterRadius) {
             // Top edge
             int srcY = max(y - filterRadius, 0);
             sharedMem[(ty) * sharedDimX + tx + filterRadius] = inputImage[srcY * width + x];
             
             // Bottom edge
             if (ty + blockDim.y < sharedDimX) {
                 int srcY2 = min(y + blockDim.y, height - 1);
                 sharedMem[(ty + blockDim.y) * sharedDimX + tx + filterRadius] = inputImage[srcY2 * width + x];
             }
         }
     }
     
     __syncthreads();
     
     // Perform Gaussian blur in shared memory
     float blurredPixel = 0.0f;
     float filterSum = 0.0f;
     
     if (x < width && y < height) {
         for (int fy = -filterRadius; fy <= filterRadius; fy++) {
             for (int fx = -filterRadius; fx <= filterRadius; fx++) {
                 int sharedX = tx + fx + filterRadius;
                 int sharedY = ty + fy + filterRadius;
                 
                 if (sharedX >= 0 && sharedX < sharedDimX && 
                     sharedY >= 0 && sharedY < sharedDimX) {
                     int filterIdx = (fy + filterRadius) * filterWidth + (fx + filterRadius);
                     float filterVal = gaussianFilter[filterIdx];
                     
                     blurredPixel += sharedMem[sharedY * sharedDimX + sharedX] * filterVal;
                     filterSum += filterVal;
                 }
             }
         }
         
         // Store blurred result back to shared memory for sharpening
         blurredPixel = blurredPixel / filterSum;
         sharedMem[ty * sharedDimX + tx + filterRadius] = static_cast<unsigned char>(blurredPixel);
     }
     
     __syncthreads();
     
     // Now apply sharpening directly
     if (x < width && y < height) {
         // Sharpening kernel
         const int sharpen[3][3] = {
             { 0, -1,  0},
             {-1,  5, -1},
             { 0, -1,  0}
         };
         
         int sum = 0;
         for (int j = -1; j <= 1; j++) {
             for (int i = -1; i <= 1; i++) {
                 int sharedX = tx + i + filterRadius;
                 int sharedY = ty + j + filterRadius;
                 
                 if (sharedX >= 0 && sharedX < sharedDimX && 
                     sharedY >= 0 && sharedY < sharedDimX) {
                     int pixel = sharedMem[sharedY * sharedDimX + sharedX];
                     sum += pixel * sharpen[j+1][i+1];
                 }
             }
         }
         
         // Clamp result and write output
         outputImage[y * width + x] = static_cast<unsigned char>(min(max(sum, 0), 255));
     }
 }
 
 // Main launcher function for the image processing pipeline
 extern "C" void processImage(
     const unsigned char* inputImage,
     unsigned char* outputGrayscale,
     unsigned char* outputBlurred,
     unsigned char* outputEdges,
     unsigned char* outputSharpened,
     int width,
     int height,
     bool optimize)
 {
     // Define block and grid dimensions
     dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
     dim3 gridDim((width + blockDim.x - 1) / blockDim.x, 
                  (height + blockDim.y - 1) / blockDim.y);
     
     // Create CUDA streams for overlapping operations
     cudaStream_t stream1, stream2, stream3;
     cudaStreamCreate(&stream1);
     cudaStreamCreate(&stream2);
     cudaStreamCreate(&stream3);
     
     // Create and initialize Gaussian filter
     int filterWidth = 5;
     float* d_gaussianFilter;
     cudaMalloc(&d_gaussianFilter, filterWidth * filterWidth * sizeof(float));
     
     dim3 filterBlockDim(16, 16);
     dim3 filterGridDim((filterWidth + filterBlockDim.x - 1) / filterBlockDim.x,
                       (filterWidth + filterBlockDim.y - 1) / filterBlockDim.y);
     
     // Generate Gaussian filter
     generateGaussianKernel<<<filterGridDim, filterBlockDim>>>(d_gaussianFilter, filterWidth, 1.0f);
     
     if (optimize) {
         // Use optimized kernels with streams
         
         // Stream 1: RGB to Grayscale
         rgbToGrayscaleOptimizedKernel<<<gridDim, blockDim, 0, stream1>>>(
             inputImage, outputGrayscale, width, height, width); // Using width as pitch for simplicity
         
         // Stream 2: Fused RGB to Edge Detection
         fusedRgbToEdgeKernel<<<gridDim, blockDim, 0, stream2>>>(
             inputImage, outputEdges, width, height);
         
         // Stream 3: Fused Blur and Sharpen
         // Calculate shared memory size
         int sharedMemSize = (BLOCK_SIZE + 2 * (filterWidth/2)) * (BLOCK_SIZE + 2 * (filterWidth/2)) * sizeof(unsigned char);
         fusedBlurSharpenKernel<<<gridDim, blockDim, sharedMemSize, stream3>>>(
             inputImage, outputSharpened, d_gaussianFilter, width, height, filterWidth);
         
         // For the blur-only result, we use a separate kernel
         // This could be optimized further by extracting the intermediate result from fusedBlurSharpenKernel
         gaussianBlurSharedKernel<<<gridDim, blockDim, sharedMemSize, stream3>>>(
             inputImage, outputBlurred, width, height, 1.0f, d_gaussianFilter, filterWidth);
     } 
     else {
         // Non-optimized sequential execution
         
         // RGB to Grayscale
         rgbToGrayscaleKernel<<<gridDim, blockDim>>>(
             inputImage, outputGrayscale, width, height);
         
         // Wait for grayscale to complete
         cudaDeviceSynchronize();
         
         // Gaussian Blur
         int sharedMemSize = (BLOCK_SIZE + 2 * (filterWidth/2)) * (BLOCK_SIZE + 2 * (filterWidth/2)) * sizeof(unsigned char);
         gaussianBlurSharedKernel<<<gridDim, blockDim, sharedMemSize>>>(
             inputImage, outputBlurred, width, height, 1.0f, d_gaussianFilter, filterWidth);
         
         // Wait for blur to complete
         cudaDeviceSynchronize();
         
         // Edge Detection (uses grayscale output)
         sobelEdgeDetectionKernel<<<gridDim, blockDim>>>(
             outputGrayscale, outputEdges, width, height);
         
         // Sharpen (uses blurred output)
         sharpenKernel<<<gridDim, blockDim>>>(
             outputBlurred, outputSharpened, width, height);
     }
     
     // Synchronize all streams
     cudaStreamSynchronize(stream1);
     cudaStreamSynchronize(stream2);
     cudaStreamSynchronize(stream3);
     
     // Clean up
     cudaFree(d_gaussianFilter);
     cudaStreamDestroy(stream1);
     cudaStreamDestroy(stream2);
     cudaStreamDestroy(stream3);
 }
 
// Benchmark function to measure performance of each operation
extern "C" void benchmarkImageProcessing(
    const unsigned char* inputImage,
    int width,
    int height,
    float* timings,    // Array to hold timing results
    bool optimize)
{
    // Define block and grid dimensions
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, 
                (height + blockDim.y - 1) / blockDim.y);
    
    // Allocate device memory for output images
    unsigned char* d_outputGrayscale;
    unsigned char* d_outputBlurred;
    unsigned char* d_outputEdges;
    unsigned char* d_outputSharpened;
    
    cudaMalloc(&d_outputGrayscale, width * height * sizeof(unsigned char));
    cudaMalloc(&d_outputBlurred, width * height * sizeof(unsigned char));
    cudaMalloc(&d_outputEdges, width * height * sizeof(unsigned char));
    cudaMalloc(&d_outputSharpened, width * height * sizeof(unsigned char));
    
    // Create and initialize Gaussian filter
    int filterWidth = 5;
    float* d_gaussianFilter;
    cudaMalloc(&d_gaussianFilter, filterWidth * filterWidth * sizeof(float));
    
    dim3 filterBlockDim(16, 16);
    dim3 filterGridDim((filterWidth + filterBlockDim.x - 1) / filterBlockDim.x,
                      (filterWidth + filterBlockDim.y - 1) / filterBlockDim.y);
    
    // Generate Gaussian filter
    generateGaussianKernel<<<filterGridDim, filterBlockDim>>>(d_gaussianFilter, filterWidth, 1.0f);
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Setup shared memory size for blur operations
    int sharedMemSize = (BLOCK_SIZE + 2 * (filterWidth/2)) * (BLOCK_SIZE + 2 * (filterWidth/2)) * sizeof(unsigned char);
    
    // Create streams for overlapping operations if using optimized version
    cudaStream_t stream1, stream2, stream3;
    if (optimize) {
        cudaStreamCreate(&stream1);
        cudaStreamCreate(&stream2);
        cudaStreamCreate(&stream3);
    }
    
    // Warm up the GPU
    rgbToGrayscaleKernel<<<gridDim, blockDim>>>(inputImage, d_outputGrayscale, width, height);
    cudaDeviceSynchronize();
    
    float elapsedTime;
    int timingIdx = 0;
    
    // ================ Benchmark Individual Operations ================
    
    // 1. RGB to Grayscale
    cudaEventRecord(start);
    if (optimize) {
        rgbToGrayscaleOptimizedKernel<<<gridDim, blockDim, 0, stream1>>>(
            inputImage, d_outputGrayscale, width, height, width);
    } else {
        rgbToGrayscaleKernel<<<gridDim, blockDim>>>(
            inputImage, d_outputGrayscale, width, height);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    timings[timingIdx++] = elapsedTime;  // Store RGB to Grayscale time
    
    // 2. Gaussian Blur
    cudaEventRecord(start);
    if (optimize) {
        gaussianBlurSharedKernel<<<gridDim, blockDim, sharedMemSize, stream2>>>(
            inputImage, d_outputBlurred, width, height, 1.0f, d_gaussianFilter, filterWidth);
    } else {
        gaussianBlurSharedKernel<<<gridDim, blockDim, sharedMemSize>>>(
            inputImage, d_outputBlurred, width, height, 1.0f, d_gaussianFilter, filterWidth);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    timings[timingIdx++] = elapsedTime;  // Store Gaussian Blur time
    
    // 3. Edge Detection
    cudaEventRecord(start);
    if (optimize) {
        fusedRgbToEdgeKernel<<<gridDim, blockDim, 0, stream3>>>(
            inputImage, d_outputEdges, width, height);
    } else {
        sobelEdgeDetectionKernel<<<gridDim, blockDim>>>(
            d_outputGrayscale, d_outputEdges, width, height);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    timings[timingIdx++] = elapsedTime;  // Store Edge Detection time
    
    // 4. Sharpen
    cudaEventRecord(start);
    if (optimize) {
        fusedBlurSharpenKernel<<<gridDim, blockDim, sharedMemSize>>>(
            inputImage, d_outputSharpened, d_gaussianFilter, width, height, filterWidth);
    } else {
        sharpenKernel<<<gridDim, blockDim>>>(
            d_outputBlurred, d_outputSharpened, width, height);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    timings[timingIdx++] = elapsedTime;  // Store Sharpen time
    
    // ================ Benchmark Full Pipeline ================
    
    // 5. Full pipeline (end-to-end)
    cudaEventRecord(start);
    
    if (optimize) {
        // Process in parallel using streams
        rgbToGrayscaleOptimizedKernel<<<gridDim, blockDim, 0, stream1>>>(
            inputImage, d_outputGrayscale, width, height, width);
            
        fusedRgbToEdgeKernel<<<gridDim, blockDim, 0, stream2>>>(
            inputImage, d_outputEdges, width, height);
            
        fusedBlurSharpenKernel<<<gridDim, blockDim, sharedMemSize, stream3>>>(
            inputImage, d_outputSharpened, d_gaussianFilter, width, height, filterWidth);
            
        gaussianBlurSharedKernel<<<gridDim, blockDim, sharedMemSize, stream3>>>(
            inputImage, d_outputBlurred, width, height, 1.0f, d_gaussianFilter, filterWidth);
            
        // Synchronize all streams
        cudaStreamSynchronize(stream1);
        cudaStreamSynchronize(stream2);
        cudaStreamSynchronize(stream3);
    } 
    else {
        // Sequential processing
        rgbToGrayscaleKernel<<<gridDim, blockDim>>>(
            inputImage, d_outputGrayscale, width, height);
        cudaDeviceSynchronize();
        
        gaussianBlurSharedKernel<<<gridDim, blockDim, sharedMemSize>>>(
            inputImage, d_outputBlurred, width, height, 1.0f, d_gaussianFilter, filterWidth);
        cudaDeviceSynchronize();
        
        sobelEdgeDetectionKernel<<<gridDim, blockDim>>>(
            d_outputGrayscale, d_outputEdges, width, height);
        cudaDeviceSynchronize();
        
        sharpenKernel<<<gridDim, blockDim>>>(
            d_outputBlurred, d_outputSharpened, width, height);
        cudaDeviceSynchronize();
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    timings[timingIdx++] = elapsedTime;  // Store Full Pipeline time
    
    // ================ Measure Memory Transfer Overhead ================
    
    // 6. Host to Device Transfer
    unsigned char* h_inputImage = new unsigned char[width * height * 3];
    unsigned char* d_inputImage;
    cudaMalloc(&d_inputImage, width * height * 3 * sizeof(unsigned char));
    
    cudaEventRecord(start);
    cudaMemcpy(d_inputImage, h_inputImage, width * height * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    timings[timingIdx++] = elapsedTime;  // Store Host to Device time
    
    // 7. Device to Host Transfer
    unsigned char* h_outputImage = new unsigned char[width * height];
    
    cudaEventRecord(start);
    cudaMemcpy(h_outputImage, d_outputGrayscale, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    timings[timingIdx++] = elapsedTime;  // Store Device to Host time
    
    // Calculate memory bandwidth
    double totalSize = width * height * 3 * sizeof(unsigned char); // Size in bytes
    double transferRate = (totalSize / 1e6) / (timings[5] / 1000.0); // MB/s
    timings[timingIdx++] = transferRate;  // Store memory bandwidth
    
    // Clean up
    cudaFree(d_outputGrayscale);
    cudaFree(d_outputBlurred);
    cudaFree(d_outputEdges);
    cudaFree(d_outputSharpened);
    cudaFree(d_gaussianFilter);
    cudaFree(d_inputImage);
    
    delete[] h_inputImage;
    delete[] h_outputImage;
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    if (optimize) {
        cudaStreamDestroy(stream1);
        cudaStreamDestroy(stream2);
        cudaStreamDestroy(stream3);
    }
}

// Profile memory access patterns to detect coalescing issues
extern "C" void profileMemoryAccess(
    const unsigned char* inputImage,
    int width,
    int height,
    bool optimized)
{
    // This function would typically be used with NVIDIA profiling tools
    // such as Nsight Compute or nvprof
    
    // For illustrative purposes, we'll just run the kernels here
    // and rely on external profiling tools to capture the metrics
    
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, 
                (height + blockDim.y - 1) / blockDim.y);
    
    unsigned char* d_outputGrayscale;
    cudaMalloc(&d_outputGrayscale, width * height * sizeof(unsigned char));
    
    // Run the kernel that we want to profile
    if (optimized) {
        rgbToGrayscaleOptimizedKernel<<<gridDim, blockDim>>>(
            inputImage, d_outputGrayscale, width, height, width);
    } else {
        rgbToGrayscaleKernel<<<gridDim, blockDim>>>(
            inputImage, d_outputGrayscale, width, height);
    }
    
    cudaDeviceSynchronize();
    cudaFree(d_outputGrayscale);
    
    // In practice, you would run this with commands like:
    // nvprof --metrics gld_efficiency,gst_efficiency ./your_application
    // to measure global memory load/store efficiency
}

// Helper function to print benchmark results
extern "C" void printBenchmarkResults(
    float* timings,
    bool optimized)
{
    printf("\n=== Benchmark Results (%s) ===\n", 
           optimized ? "Optimized" : "Non-optimized");
    printf("RGB to Grayscale:     %.3f ms\n", timings[0]);
    printf("Gaussian Blur:        %.3f ms\n", timings[1]);
    printf("Edge Detection:       %.3f ms\n", timings[2]);
    printf("Image Sharpening:     %.3f ms\n", timings[3]);
    printf("Full Pipeline:        %.3f ms\n", timings[4]);
    printf("Host to Device:       %.3f ms\n", timings[5]);
    printf("Device to Host:       %.3f ms\n", timings[6]);
    printf("Memory Bandwidth:     %.2f MB/s\n", timings[7]);
    
    float computeTime = timings[0] + timings[1] + timings[2] + timings[3];
    float transferTime = timings[5] + timings[6];
    
    printf("\nComputation Time:     %.3f ms\n", computeTime);
    printf("Transfer Time:        %.3f ms\n", transferTime);
    printf("Transfer Overhead:    %.1f%%\n", (transferTime / (computeTime + transferTime)) * 100.0f);
    
    if (optimized) {
        float speedup = (timings[0] + timings[1] + timings[2] + timings[3]) / timings[4];
        printf("Pipeline Speedup:     %.2fx\n", speedup);
    }
}

// Main function to demonstrate usage
int main(int argc, char** argv) {
    int width = 1920;
    int height = 1080;
    
    // Allocate host memory
    unsigned char* h_inputImage = new unsigned char[width * height * 3];
    
    // Fill with sample data (or load from file)
    for (int i = 0; i < width * height * 3; i++) {
        h_inputImage[i] = rand() % 256;
    }
    
    // Allocate device memory
    unsigned char* d_inputImage;
    cudaMalloc(&d_inputImage, width * height * 3 * sizeof(unsigned char));
    
    // Copy input to device
    cudaMemcpy(d_inputImage, h_inputImage, width * height * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);
    
    // Allocate memory for timing results
    // [0]: RGB to Grayscale, [1]: Gaussian Blur, [2]: Edge Detection, [3]: Sharpen
    // [4]: Full Pipeline, [5]: Host to Device, [6]: Device to Host, [7]: Memory Bandwidth
    float* timingsOptimized = new float[8];
    float* timingsNonOptimized = new float[8];
    
    // Run optimized benchmarks
    benchmarkImageProcessing(d_inputImage, width, height, timingsOptimized, true);
    
    // Run non-optimized benchmarks
    benchmarkImageProcessing(d_inputImage, width, height, timingsNonOptimized, false);
    
    // Print results
    printBenchmarkResults(timingsOptimized, true);
    printBenchmarkResults(timingsNonOptimized, false);
    
    // Calculate overall speedup
    float optimizedTotal = timingsOptimized[4];  // Full pipeline time
    float nonOptimizedTotal = timingsNonOptimized[4];  // Full pipeline time
    
    printf("\n=== Overall Performance ===\n");
    printf("Optimized Pipeline:   %.3f ms\n", optimizedTotal);
    printf("Non-optimized:        %.3f ms\n", nonOptimizedTotal);
    printf("Overall Speedup:      %.2fx\n", nonOptimizedTotal / optimizedTotal);
    
    // Profile memory access patterns
    profileMemoryAccess(d_inputImage, width, height, true);
    
    // Clean up
    cudaFree(d_inputImage);
    delete[] h_inputImage;
    delete[] timingsOptimized;
    delete[] timingsNonOptimized;
    
    return 0;
}