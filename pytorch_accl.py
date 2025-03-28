import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

class GPUImageProcessor:
    def __init__(self, use_gpu=True, optimize=True, stream_processing=True):
        """
        Initialize the GPU-accelerated image processor
        
        Args:
            use_gpu: Whether to use GPU acceleration
            optimize: Whether to use optimized kernels (kernel fusion, etc.)
            stream_processing: Whether to use CUDA streams for overlapping operations
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
        self.optimize = optimize
        self.stream_processing = stream_processing
        
        # Define Sobel filters for edge detection
        self.sobel_x = torch.tensor([[-1, 0, 1], 
                                     [-2, 0, 2], 
                                     [-1, 0, 1]], dtype=torch.float32).to(self.device)
        
        self.sobel_y = torch.tensor([[-1, -2, -1], 
                                     [0, 0, 0], 
                                     [1, 2, 1]], dtype=torch.float32).to(self.device)
        
        # Define sharpening kernel
        self.sharpen_kernel = torch.tensor([[0, -1, 0],
                                           [-1, 5, -1],
                                           [0, -1, 0]], dtype=torch.float32).to(self.device)
        
        # Create streams for overlapping operations if enabled
        self.streams = None
        if self.stream_processing and torch.cuda.is_available():
            self.streams = [torch.cuda.Stream() for _ in range(3)]  # One stream for each major operation
    
    def load_image(self, image_path):
        """Load an image and convert to tensor on the appropriate device"""
        img = Image.open(image_path)
        img_tensor = transforms.ToTensor()(img).to(self.device)
        return img_tensor
    
    def save_image(self, tensor, path):
        """Save a tensor as an image"""
        # Convert to CPU, then numpy, then PIL Image
        img = transforms.ToPILImage()(tensor.cpu())
        img.save(path)
    
    def rgb_to_grayscale(self, image):
        """Convert RGB image to grayscale"""
        # Method matches ITU-R BT.601 standard for luminance
        weights = torch.tensor([0.299, 0.587, 0.114]).to(self.device)
        return (image * weights.view(3, 1, 1)).sum(dim=0, keepdim=True)
    
    def gaussian_blur(self, image, kernel_size=5, sigma=1.0):
        """Apply Gaussian blur to an image"""
        # Use PyTorch's built-in 2D Gaussian blur
        padding = kernel_size // 2
        return F.gaussian_blur(image, kernel_size=(kernel_size, kernel_size), sigma=(sigma, sigma))
    
    def edge_detection(self, image):
        """Apply Sobel edge detection"""
        # Make sure we're working with grayscale
        if image.shape[0] == 3:
            image = self.rgb_to_grayscale(image)
        
        # Reshape Sobel filters for 2D convolution (add dimensions for batch and channels)
        sobel_x = self.sobel_x.view(1, 1, 3, 3)
        sobel_y = self.sobel_y.view(1, 1, 3, 3)
        
        # Apply Sobel filters
        grad_x = F.conv2d(image, sobel_x, padding=1)
        grad_y = F.conv2d(image, sobel_y, padding=1)
        
        # Compute gradient magnitude
        grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2)
        
        # Normalize to [0, 1]
        if grad_magnitude.max() > 0:
            grad_magnitude = grad_magnitude / grad_magnitude.max()
            
        return grad_magnitude
    
    def sharpen(self, image):
        """Apply sharpening filter to image"""
        sharpen_kernel = self.sharpen_kernel.view(1, 1, 3, 3)
        
        # Apply separately to each channel if it's an RGB image
        if image.shape[0] == 3:
            result = torch.zeros_like(image)
            for c in range(3):
                channel = image[c:c+1]
                result[c:c+1] = F.conv2d(channel, sharpen_kernel, padding=1)
            return result
        else:
            return F.conv2d(image, sharpen_kernel, padding=1)
    
    def fused_rgb_to_edge(self, image):
        """Optimized version that fuses RGB-to-grayscale and edge detection in one kernel"""
        # In a real CUDA implementation, this would be a custom kernel
        # Here we're simulating kernel fusion by making sure operations stay on GPU
        with torch.no_grad():
            gray = self.rgb_to_grayscale(image)
            return self.edge_detection(gray)
            
    def fused_blur_sharpen(self, image):
        """Optimized version that fuses blur and sharpen operations"""
        # In a real CUDA implementation, this would be a custom kernel
        # Here we're simulating kernel fusion
        with torch.no_grad():
            blurred = self.gaussian_blur(image)
            return self.sharpen(blurred)
    
    def process_image(self, image_tensor):
        """
        Process an image through the entire pipeline
        
        Returns a dictionary with the results of each processing step
        """
        results = {}
        timings = {}
        
        # Warmup GPU
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Ensure we're working with the right device
        if image_tensor.device != self.device:
            image_tensor = image_tensor.to(self.device)
        
        # Regular (non-optimized) processing
        if not self.optimize:
            # Record times for each operation
            start = time.time()
            results['grayscale'] = self.rgb_to_grayscale(image_tensor)
            timings['grayscale'] = time.time() - start
            
            start = time.time()
            results['blur'] = self.gaussian_blur(image_tensor)
            timings['blur'] = time.time() - start
            
            start = time.time()
            results['edge'] = self.edge_detection(image_tensor)
            timings['edge'] = time.time() - start
            
            start = time.time()
            results['sharpen'] = self.sharpen(image_tensor)
            timings['sharpen'] = time.time() - start
        
        # Optimized processing with kernel fusion
        else:
            if self.stream_processing and self.device.type == 'cuda':
                # Stream 1: RGB to Edge (fused)
                with torch.cuda.stream(self.streams[0]):
                    start = time.time()
                    results['edge'] = self.fused_rgb_to_edge(image_tensor)
                    timings['edge'] = time.time() - start
                
                # Stream 2: Grayscale conversion
                with torch.cuda.stream(self.streams[1]):
                    start = time.time()
                    results['grayscale'] = self.rgb_to_grayscale(image_tensor)
                    timings['grayscale'] = time.time() - start
                
                # Stream 3: Blur and sharpen (fused)
                with torch.cuda.stream(self.streams[2]):
                    start = time.time()
                    results['sharpen'] = self.fused_blur_sharpen(image_tensor)
                    # Blur is a byproduct of the fused operation
                    blurred = self.gaussian_blur(image_tensor)
                    results['blur'] = blurred
                    timings['blur_sharpen'] = time.time() - start
                
                # Synchronize streams
                torch.cuda.synchronize()
            else:
                # Fused operations without streaming
                start = time.time()
                results['edge'] = self.fused_rgb_to_edge(image_tensor)
                timings['edge'] = time.time() - start
                
                start = time.time()
                results['grayscale'] = self.rgb_to_grayscale(image_tensor)
                timings['grayscale'] = time.time() - start
                
                start = time.time()
                results['blur'] = self.gaussian_blur(image_tensor)
                timings['blur'] = time.time() - start
                
                start = time.time()
                results['sharpen'] = self.sharpen(image_tensor)
                timings['sharpen'] = time.time() - start
        
        return results, timings
    
    def benchmark(self, image_tensor, num_runs=10):
        """Benchmark the processing pipeline with multiple runs"""
        total_times = {
            'optimized': [],
            'unoptimized': []
        }
        
        # Benchmark optimized version
        self.optimize = True
        for _ in range(num_runs):
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            start = time.time()
            _, _ = self.process_image(image_tensor)
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            total_times['optimized'].append(time.time() - start)
        
        # Benchmark unoptimized version
        self.optimize = False
        for _ in range(num_runs):
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            start = time.time()
            _, _ = self.process_image(image_tensor)
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            total_times['unoptimized'].append(time.time() - start)
        
        # Calculate average times
        avg_times = {
            'optimized': np.mean(total_times['optimized']),
            'unoptimized': np.mean(total_times['unoptimized'])
        }
        
        # Calculate speedup
        speedup = avg_times['unoptimized'] / avg_times['optimized']
        
        return avg_times, speedup, total_times
    
    def visualize_results(self, results):
        """Visualize the processing results"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Original image
        if 'original' in results:
            axes[0, 0].imshow(transforms.ToPILImage()(results['original'].cpu()))
            axes[0, 0].set_title('Original')
        
        # Grayscale
        if 'grayscale' in results:
            gray = results['grayscale'].cpu()
            if gray.shape[0] == 1:
                axes[0, 1].imshow(transforms.ToPILImage()(gray))
            else:
                axes[0, 1].imshow(gray.squeeze().numpy(), cmap='gray')
            axes[0, 1].set_title('Grayscale')
        
        # Edge detection
        if 'edge' in results:
            edge = results['edge'].cpu()
            if edge.shape[0] == 1:
                axes[1, 0].imshow(edge.squeeze().numpy(), cmap='gray')
            else:
                axes[1, 0].imshow(transforms.ToPILImage()(edge))
            axes[1, 0].set_title('Edge Detection')
        
        # Sharpened
        if 'sharpen' in results:
            axes[1, 1].imshow(transforms.ToPILImage()(results['sharpen'].cpu()))
            axes[1, 1].set_title('Sharpened')
        
        for ax in axes.flat:
            ax.axis('off')
        
        plt.tight_layout()
        return fig

    def visualize_benchmark(self, avg_times, speedup):
        """Visualize benchmark results"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        operations = list(avg_times.keys())
        times = list(avg_times.values())
        
        bars = ax.bar(operations, times, color=['#1f77b4', '#ff7f0e'])
        ax.set_ylabel('Time (seconds)')
        ax.set_title(f'Processing Time Comparison (Speedup: {speedup:.2f}x)')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.5f}s',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
        
        return fig

# Example usage
if __name__ == "__main__":
    # Create processor
    processor = GPUImageProcessor(use_gpu=True, optimize=True, stream_processing=True)
    
    # Load an example image
    try:
        # Try to load from a standard location
        image = processor.load_image("sample_image.jpg")
    except:
        # If it doesn't exist, generate a sample tensor
        # Create a sample 3-channel image
        image = torch.randn(3, 512, 512, device=processor.device)
        # Normalize to [0, 1]
        image = (image - image.min()) / (image.max() - image.min())
    
    # Process the image
    results, timings = processor.process_image(image)
    results['original'] = image
    
    # Benchmark
    avg_times, speedup, all_times = processor.benchmark(image, num_runs=10)
    
    print(f"Processing time (optimized): {avg_times['optimized']:.5f} seconds")
    print(f"Processing time (unoptimized): {avg_times['unoptimized']:.5f} seconds")
    print(f"Speedup: {speedup:.2f}x")
    
    # Visualize results
    result_fig = processor.visualize_results(results)
    result_fig.savefig("processing_results.png")
    
    # Visualize benchmark
    benchmark_fig = processor.visualize_benchmark(avg_times, speedup)
    benchmark_fig.savefig("benchmark_results.png")