import glob
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from ultralytics import SAM

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
TEST_CITY   = "Minas_Novas"
IMG_FOLDER  = "data/patches/images/"
MASK_FOLDER = "data/patches/masks/"
SAVE_DIR    = Path(f"preds/sam_zeroshot_{TEST_CITY}")

SAVE_DIR.mkdir(parents=True, exist_ok=True)

model = SAM('sam2_b.pt') 

img_files  = sorted(glob.glob(os.path.join(IMG_FOLDER, f'*_{TEST_CITY}_*.npy')))
mask_files = sorted(glob.glob(os.path.join(MASK_FOLDER, f'*_{TEST_CITY}_*.npy')))

TOTAL_PIXELS = 128 * 128
MIN_SILVICULTURE_RATIO = 0.30 # 30%

processed_count = 0

for img_path, mask_path in tqdm(zip(img_files, mask_files), total=len(img_files), desc=f"Inference: {TEST_CITY}"):
    gt_mask = np.load(mask_path)
    
    area_ratio = gt_mask.sum() / TOTAL_PIXELS
    if area_ratio < MIN_SILVICULTURE_RATIO:
        continue
        
    img_np  = np.load(img_path)

    # Dry period (t=0)
    img_t0 = img_np[0]

    # SAM-> RGB in HWC & uint8 (0-255)
    # Sentinel-2 RGB: B4 (idx 2), B3 (idx 1), B2 (idx 0)
    rgb = np.stack([img_t0[2], img_t0[1], img_t0[0]], axis=-1)

    rgb = np.clip((rgb / 10000.0) * 3.5, 0, 1)
    rgb_uint8 = (rgb * 255).astype(np.uint8)

    # Zero-Shot
    results   = model(rgb_uint8, device=DEVICE, verbose=False)
    sam_masks = results[0].masks.data.cpu().numpy() if results[0].masks is not None else []

    combined_sam_mask = np.zeros((128, 128))
    for i, mask in enumerate(sam_masks):
        # Diff value for each instance
        combined_sam_mask[mask > 0] = i + 1 

    fig, axes  = plt.subplots(1, 3, figsize=(15, 5))
    patch_name = os.path.basename(img_path)
    fig.suptitle(f"SAM Zero-Shot - {patch_name}", fontsize=14)

    axes[0].imshow(rgb)
    axes[0].set_title("Sentinel-2 RGB (T=0)")
    axes[0].axis("off")

    axes[1].imshow(gt_mask, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    # SAM Masks
    axes[2].imshow(combined_sam_mask, cmap="jet")
    axes[2].set_title(f"SAM Masks ({len(sam_masks)} objects)")
    axes[2].axis("off")

    plt.tight_layout()
    
    save_filename = patch_name.replace('.npy', '.png')
    plt.savefig(SAVE_DIR / save_filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    processed_count += 1
