import time
import numpy as np
import scipy.fft
import pyfftw

# Function to benchmark a given FFT function
def benchmark_fft(func, data):
    start_time = time.time()
    func(data)
    end_time = time.time()
    return end_time - start_time

# Data sizes for 1D, 2D, and 3D FFTs
sizes = [1024, (64, 64), (16, 16, 16)]

# Number of iterations for averaging
iterations = 10

# Benchmark results
results = {
    'numpy': {'1D': [], '2D': [], '3D': []},
    'scipy': {'1D': [], '2D': [], '3D': []},
    'pyfftw': {'1D': [], '2D': [], '3D': []}
}

# Run benchmarks
for size in sizes:
    for _ in range(iterations):
        # Generate random data
        data = np.random.random(size)

        # Numpy FFT
        if isinstance(size, int):
            results['numpy']['1D'].append(benchmark_fft(np.fft.fft, data))
        elif len(size) == 2:
            results['numpy']['2D'].append(benchmark_fft(np.fft.fft2, data))
        elif len(size) == 3:
            results['numpy']['3D'].append(benchmark_fft(np.fft.fftn, data))

        # Scipy FFT
        if isinstance(size, int):
            results['scipy']['1D'].append(benchmark_fft(scipy.fft.fft, data))
        elif len(size) == 2:
            results['scipy']['2D'].append(benchmark_fft(scipy.fft.fft2, data))
        elif len(size) == 3:
            results['scipy']['3D'].append(benchmark_fft(scipy.fft.fftn, data))

        # pyFFTW FFT
        if isinstance(size, int):
            results['pyfftw']['1D'].append(benchmark_fft(pyfftw.interfaces.numpy_fft.fft, data))
        elif len(size) == 2:
            results['pyfftw']['2D'].append(benchmark_fft(pyfftw.interfaces.numpy_fft.fft2, data))
        elif len(size) == 3:
            results['pyfftw']['3D'].append(benchmark_fft(pyfftw.interfaces.numpy_fft.fftn, data))

# Calculate average times
average_times = {lib: {dim: np.mean(times) for dim, times in dims.items()} for lib, dims in results.items()}

# Print results
for lib, dims in average_times.items():
    for dim, avg_time in dims.items():
        print(f"Average time for {lib} {dim} FFT: {avg_time:.6f} seconds")
