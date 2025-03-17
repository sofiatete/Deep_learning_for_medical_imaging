import matplotlib.pyplot as plt
import h5py
import numpy as np

from fastmri.data.subsample import create_mask_for_mask_type

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


# perform inverse fourier transform and shifting
def fourier_transform(kspace):
    """Reconstructs an image using the Fourier transform with proper shifting."""
    dim1, dim2 = -2, -1  # Last two dimensions (H, W)

    # Shift k-space center to middle before performing IFT
    kspace_shifted = np.fft.ifftshift(kspace, axes=(dim1, dim2))

    # Perform the inverse 2D Fourier Transform
    image = np.fft.ifft2(kspace_shifted, axes=(dim1, dim2))

    # Shift the final image to center it correctly
    image = np.fft.fftshift(image, axes=(dim1, dim2))

    return image

# Perform inverse Fourier transform
image = fourier_transform(kspace_center_slice)

# Take the magnitude to get a real-valued image
image_magnitude = np.abs(image).squeeze()

# POINT A)
# Plot K-space (log scale for better visualization)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(np.log1p(np.abs(kspace_center_slice[0])), cmap="gray")
plt.title("K-Space (Log Scale)")
plt.axis("off")

# Plot the magnitude image (Image domain)
plt.subplot(1, 2, 2)
plt.imshow(image_magnitude, cmap="gray")  # Use magnitude
plt.title("Image Space (Magnitude)")
plt.axis("off")

# plt.show()

# POINT B)
# Reconstruct images for all slices
num_slices = kspace.shape[0]
reconstructed_images = np.abs(fourier_transform(kspace))  # Take magnitude

# Plot images in a tiled display
cols = 6  # Number of columns in subplot grid
rows = int(np.ceil(num_slices / cols))  # Compute number of rows needed

fig, axes = plt.subplots(rows, cols, figsize=(15, 15))

for i, ax in enumerate(axes.flat):
    if i < num_slices:
        ax.imshow(reconstructed_images[i].squeeze(), cmap="gray")
        ax.set_title(f"Slice {i}")
    ax.axis("off")  # Hide axes

plt.tight_layout()
# plt.show()

# POINT C)

# Extract components
image_magnitude = np.abs(image).squeeze()  # Magnitude
image_phase = np.angle(image).squeeze()    # Phase
image_real = np.real(image).squeeze()      # Real part
image_imag = np.imag(image).squeeze()      # Imaginary part

# Plot the four components in a 2x2 grid
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Magnitude Image
axes[0, 0].imshow(image_magnitude, cmap="gray")
axes[0, 0].set_title("Magnitude")
axes[0, 0].axis("off")

# Phase Image
axes[0, 1].imshow(image_phase, cmap="twilight")
axes[0, 1].set_title("Phase")
axes[0, 1].axis("off")

# Real Component
axes[1, 0].imshow(image_real, cmap="gray")
axes[1, 0].set_title("Real Component")
axes[1, 0].axis("off")

# Imaginary Component
axes[1, 1].imshow(image_imag, cmap="gray")
axes[1, 1].set_title("Imaginary Component")
axes[1, 1].axis("off")

plt.tight_layout()
# plt.show()

print("Min phase value:", np.min(image_phase))
print("Max phase value:", np.max(image_phase))

# MASKING
# Load k-space data
with h5py.File("file135.h5", "r") as f:
    print("Keys in the dataset:", list(f.keys()))  
    kspace = f["kspace"][:]  # Shape: (num_slices, num_coils, H, W)

# Select one test slice (e.g., center slice)
center_slice_idx = kspace.shape[0] // 2
kspace_center_slice = kspace[center_slice_idx]  # Shape: (num_coils, H, W)

# Create the k-space mask
mask_func = create_mask_for_mask_type(
    mask_type_str="random",  # Type of undersampling (can be "uniform", "random", etc.)
    center_fractions=[0.08],  # Keep 8% of low-frequency k-space data
    accelerations=[4]  # Acceleration factor of 4x
)

# Apply the mask to k-space
mask, acc = mask_func(kspace_center_slice.shape)
mask = mask[0,0] # you need to to this because the resulting mask is 4D, and you need it to be 2D 

kspace_masked = kspace_center_slice * mask.numpy().astype(kspace_center_slice.dtype)

# Reconstruct the masked image
image_masked = np.abs(fourier_transform(kspace_masked))

# Plot original and masked k-space
plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
plt.imshow(np.log1p(np.abs(kspace_center_slice[0])), cmap="gray")
plt.title("Original K-Space (Log Scale)")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(np.log1p(np.abs(kspace_masked[0])), cmap="gray")
plt.title(f"Masked K-Space (Acceleration = 4x)")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(np.abs(fourier_transform(kspace_center_slice)).squeeze(), cmap="gray")
plt.title("Original Image (Magnitude)")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(image_masked.squeeze(), cmap="gray")
plt.title("Masked Image (Magnitude)")
plt.axis("off")

plt.show()