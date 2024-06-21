import pyfftw
import numpy as np
import matplotlib.pyplot as plt

# Create a 2D array representing an image (e.g., a simple gradient)
image = np.linspace(0, 255, num=256*256).reshape((256, 256))

# Perform 2D FFT using pyFFTW
fft_image = pyfftw.interfaces.numpy_fft.fft2(image)

# Perform inverse 2D FFT to reconstruct the original image
reconstructed_image = pyfftw.interfaces.numpy_fft.ifft2(fft_image)

# Take the real part of the reconstructed image (should be very close to the original)
reconstructed_image = np.real(reconstructed_image)

# Plot original and reconstructed images for visual comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

ax1.imshow(image, cmap='gray')
ax1.set_title('Original Image')

ax2.imshow(reconstructed_image, cmap='gray')
ax2.set_title('Reconstructed Image')

plt.show()

# Verify the difference
difference = np.abs(image - reconstructed_image)
max_difference = np.max(difference)

print(f"Max difference between original and reconstructed image: {max_difference}")

# Asserting the max difference to ensure correctness
assert max_difference < 1e-10, "Reconstructed image is not accurate enough!"

