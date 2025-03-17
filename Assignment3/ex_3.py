import matplotlib.pyplot as plt
import h5py
import numpy as np

from reconstruction_functions import fourier_transform 

# Load k-space data
with h5py.File("file061.h5", "r") as f:
    print("Keys in the dataset:", list(f.keys()))  
    kspace = f["kspace"][:]  # Shape: (num_slices, num_coils, H, W)

# Select the center slice
center_slice_idx = kspace.shape[0] // 2
kspace_center_slice = kspace[center_slice_idx]  # Shape: (num_coils, H, W)

print("K-space shape:", kspace.shape)
print("Center slice index:", center_slice_idx)
print("Center slice shape:", kspace_center_slice.shape)

# Perform inverse Fourier transform
image = fourier_transform(kspace_center_slice)


# Plot K-space (log scale for better visualization)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(np.log1p(np.abs(kspace_center_slice[0])), cmap="gray")
plt.title("K-Space (Log Scale)")
plt.axis("off")

# Plot the magnitude image (Image domain)
plt.subplot(1, 2, 2)
plt.imshow(image, cmap="gray")
plt.title("Image Space (Magnitude)")
plt.axis("off")

plt.show()
