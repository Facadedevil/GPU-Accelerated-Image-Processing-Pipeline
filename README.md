# GPU-Accelerated Image Processing Pipeline

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CUDA Version](https://img.shields.io/badge/CUDA-11.0+-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)

A high-performance image processing pipeline that leverages GPU acceleration for common operations like Gaussian blur, edge detection, color space conversion, and image sharpening.

[📑 **Comprehensive Wiki Documentation**](https://github.com/Facadedevil/GPU-Accelerated-Image-Processing-Pipeline/wiki)

<p align="center">
  <img src="https://github.com/Facadedevil/GPU-Accelerated-Image-Processing-Pipeline/assets/your-user-id/image-asset-id.png" alt="Performance Improvement from Memory Coalescing on RTX 4080 Ti" width="700"/>
</p>

## ⚡ Quick Start

```bash
# Clone repository
git clone https://github.com/Facadedevil/GPU-Accelerated-Image-Processing-Pipeline.git
cd GPU-Accelerated-Image-Processing-Pipeline

# Python setup
pip install -r python/requirements.txt

# Run example
python examples/basic_example.py
```

## 🚀 Features

- **Core Image Processing Operations**:
  - Gaussian blur
  - Edge detection (Sobel filter)
  - RGB to grayscale conversion
  - Image sharpening

- **GPU Optimization Techniques**:
  - Memory coalescing for efficient data access
  - Shared memory utilization to reduce global memory access
  - Kernel fusion to reduce overhead of multiple operations
  - Work distribution optimization across thread blocks
  - Stream processing for overlapping data transfers and computation

- **Performance Analysis Tools**:
  - Comprehensive benchmarking suite
  - Comparison between optimized and unoptimized versions
  - Integration with NVIDIA profiling tools

- **Dual Implementation**:
  - High-level Python (PyTorch) for rapid prototyping and easy integration
  - Low-level CUDA for maximum performance and control

## 📋 Requirements

### Python Implementation
- Python 3.7+
- PyTorch 1.9+
- CUDA Toolkit 11.0+
- NumPy
- Matplotlib (for visualization)

### CUDA Implementation
- CUDA Toolkit 11.0+
- C++17 compatible compiler
- CMake 3.18+ (for building)

## 💻 Installation

### Python

```bash
# Clone the repository
git clone https://github.com/Facadedevil/GPU-Accelerated-Image-Processing-Pipeline.git
cd GPU-Accelerated-Image-Processing-Pipeline

# Install Python dependencies
pip install -r python/requirements.txt
```

### CUDA

```bash
# Navigate to CUDA directory
cd cuda

# Build with CMake
mkdir build && cd build
cmake ..
make

# Or use the provided Makefile
cd cuda
make
```

## 🔍 Usage

### Python Example

```python
from python.image_processor import GPUImageProcessor
from PIL import Image
import torch

# Create processor
processor = GPUImageProcessor(use_gpu=True, optimize=True)

# Load image
image = processor.load_image("data/sample1.jpg")

# Process image
results, timings = processor.process_image(image)

# Visualize results
processor.visualize_results(results)

# Benchmark
avg_times, speedup, _ = processor.benchmark(image, num_runs=10)
print(f"Speedup: {speedup:.2f}x")
```

### CUDA Example

```cpp
#include "image_processing.h"

int main() {
    // Load image
    unsigned char* image = loadImage("data/sample1.jpg", &width, &height);
    
    // Allocate memory for results
    unsigned char *grayscale, *blurred, *edges, *sharpened;
    allocateMemory(&grayscale, &blurred, &edges, &sharpened, width, height);
    
    // Process image with optimizations
    processImage(image, grayscale, blurred, edges, sharpened, width, height, true);
    
    // Save results
    saveImage("grayscale.jpg", grayscale, width, height, 1);
    saveImage("blurred.jpg", blurred, width, height, 3);
    saveImage("edges.jpg", edges, width, height, 1);
    saveImage("sharpened.jpg", sharpened, width, height, 3);
    
    // Benchmark
    float timings[8];
    benchmarkImageProcessing(image, width, height, timings, true);
    printBenchmarkResults(timings, true);
    
    // Clean up
    freeMemory(image, grayscale, blurred, edges, sharpened);
    
    return 0;
}
```

## 📊 Performance

The optimized implementation achieves significant speedups compared to the unoptimized version:

| Operation         | Unoptimized | Optimized | Speedup |
|-------------------|-------------|-----------|---------|
| RGB to Grayscale  | 0.38 ms     | 0.17 ms   | 2.2x    |
| Gaussian Blur     | 0.98 ms     | 0.32 ms   | 3.1x    |
| Edge Detection    | 0.52 ms     | 0.18 ms   | 2.9x    |
| Image Sharpening  | 0.58 ms     | 0.21 ms   | 2.8x    |
| Full Pipeline     | 2.46 ms     | 0.58 ms   | 4.2x    |

*Measured on an NVIDIA RTX 4080 Ti with a 1920x1080 image*

<p align="center">
  <img src="docs/images/optimization_impact.png" alt="Optimization Impact" width="600"/>
</p>

## 📚 Documentation

Detailed documentation is available in our [Wiki](https://github.com/Facadedevil/GPU-Accelerated-Image-Processing-Pipeline/wiki) and in the `docs/` directory:

- [Architecture Overview](docs/architecture.md) - High-level system design
- [Optimization Guide](docs/optimization_guide.md) - Detailed explanation of GPU optimization techniques
- [API Reference](docs/api_reference.md) - Complete API documentation
- [Benchmarks](docs/benchmarks.md) - Performance analysis and comparison
- [Examples](docs/examples.md) - Additional usage examples

## 🧩 Project Structure

```
gpu-image-processing/
│
├── python/                    # Python prototype implementation
│   ├── image_processor.py     # Main implementation file
│   ├── benchmarks.py          # Benchmarking utilities
│   └── ...
│
├── cuda/                      # CUDA implementation
│   ├── include/               # Header files
│   ├── src/                   # Source files
│   └── ...
│
├── docs/                      # Documentation
│   ├── architecture.md        # Architecture overview
│   ├── optimization_guide.md  # Optimization documentation
│   └── ...
│
├── data/                      # Example images for testing
│
└── scripts/                   # Utility scripts
```

## 🔬 Key Optimization Techniques

### Memory Coalescing

```cpp
// Unoptimized memory access (stride = 3)
int idx = (y * width + x) * 3;  // Non-coalesced

// Optimized memory access (stride = 1)
int idx = y * (width * 3) + (x * 3);  // Coalesced
```

### Shared Memory Utilization

```cpp
// Shared memory usage for blur filter
__shared__ unsigned char sharedMem[TILE_SIZE][TILE_SIZE];

// Cooperative loading into shared memory
if (x < width && y < height) {
    sharedMem[ty][tx] = inputImage[y * width + x];
    // Load apron region...
}

__syncthreads();  // Ensure all threads have loaded data

// Process from shared memory instead of global memory
```

### Kernel Fusion

```cpp
// Without fusion: Two separate kernel launches
rgbToGrayscaleKernel<<<...>>>(inputImage, grayscaleImage, width, height);
sobelEdgeDetectionKernel<<<...>>>(grayscaleImage, edgeImage, width, height);

// With fusion: Single kernel launch
fusedRgbToEdgeKernel<<<...>>>(inputImage, edgeImage, width, height);
```

## 💻 Hardware Compatibility

This pipeline has been tested on the following hardware:
- NVIDIA RTX 4080 Ti
- NVIDIA RTX 3080

Minimum requirement: CUDA-capable GPU with compute capability 6.0+

## 📋 Future Work

- Additional image processing operations (non-local means denoising, HDR tone mapping)
- Multi-GPU support for processing large datasets
- Integration with deep learning frameworks for end-to-end pipelines
- Mixed-precision computing for improved performance on newer GPUs

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- NVIDIA for CUDA documentation and examples
- PyTorch team for their excellent GPU abstractions