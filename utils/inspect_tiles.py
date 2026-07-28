import numpy as np
import matplotlib.pyplot as plt
import random
import os
import glob

img_folder = 'data/patches/images/'
mask_folder = 'data/patches/masks/'
img_files = glob.glob(os.path.join(img_folder, '*.npy'))

patch = random.choice(img_files)
fname = os.path.basename(patch)

img_tensor = np.load(patch) # (T, C, 128, 128)
mask_tensor = np.load(os.path.join(mask_folder, fname.replace('patch_', 'mask_'))) # (128, 128)
print("Shape:", img_tensor.shape)
print("dtype:", img_tensor.dtype)
print("Min:", img_tensor.min())
print("Max:", img_tensor.max())
print("NaNs:", np.isnan(img_tensor).sum())
print("Total elements", img_tensor.size)

img_float = img_tensor / 10000.0

img_t0 = img_float[0]

# ['B2', 'B3', 'B4', 'B8', 'B5', 'B6', 'B7', 'B8A', 'B11', 'B12']
rgb = np.stack([img_t0[2], img_t0[1], img_t0[0]], axis=-1)

rgb = np.clip(rgb * 3.5, 0, 1)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].imshow(rgb)
axes[0].set_title(f"RGB Image at T0 - {fname}", fontsize=14)
axes[0].axis('off')

axes[1].imshow(mask_tensor, cmap='gray')
axes[1].set_title("Ground Truth", fontsize=14)
axes[1].axis('off')

plt.tight_layout()
plt.show()
