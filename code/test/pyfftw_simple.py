import pyfftw
import numpy as np

a = pyfftw.empty_aligned(1024, dtype='complex128')
b = pyfftw.empty_aligned(1024, dtype='complex128')

fft = pyfftw.FFTW(a, b)
print("FFT successfully performed with pyFFTW and FFTW libraries!")

