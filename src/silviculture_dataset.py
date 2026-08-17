import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class SilvicultureDataset(Dataset):
    def __init__(self, img_folder, mask_folder, allowed_cities=None, augment=False, mode="3d"):
        super().__init__()
        self.img_folder = img_folder
        self.mask_folder = mask_folder
        self.augment = augment
        self.mode = mode
        self.T = 10 # Steps in each tensor
        
        all_imgs  = sorted(glob.glob(os.path.join(img_folder, '*.npy')))
        all_masks = sorted(glob.glob(os.path.join(mask_folder, '*.npy')))
        
        if allowed_cities is not None:
            self.img_files  = [f for f in all_imgs if any(city in os.path.basename(f) for city in allowed_cities)]
            self.mask_files = [f for f in all_masks if any(city in os.path.basename(f) for city in allowed_cities)]
        else:
            self.img_files  = all_imgs
            self.mask_files = all_masks
            
        assert len(self.img_files) == len(self.mask_files), "Mismatch between images and masks"
        
    def __len__(self):
        # 2D = dataset 10x bigger
        if self.mode == "2d":
            return len(self.img_files) * self.T
        return len(self.img_files)

    def __getitem__(self, idx):
        if self.mode == "2d":
            file_idx = idx // self.T
            t_idx = idx % self.T
        else:
            file_idx = idx
            
        img_path  = self.img_files[file_idx]
        mask_path = self.mask_files[file_idx]
        
        img_np  = np.load(img_path)
        mask_np = np.load(mask_path)
        
        nodata_mask = np.all(img_np == 0, axis=(0, 1)) 
        mask_np[nodata_mask] = 0
        
        img_np = img_np.astype(np.float32) / 10000.0
        
        if self.augment:
            if np.random.rand() > 0.5:
                img_np  = np.flip(img_np, axis=3)
                mask_np = np.flip(mask_np, axis=1)
            if np.random.rand() > 0.5:
                img_np  = np.flip(img_np, axis=2)
                mask_np = np.flip(mask_np, axis=0)

        img_tensor  = torch.from_numpy(img_np.copy())
        mask_tensor = torch.from_numpy(mask_np.copy()).long()

        if self.mode == "2d":
            # Isolates timestamp
            img_tensor_2d = img_tensor[t_idx]
            return img_tensor_2d, mask_tensor
        else:
            days = [0, 180, 365, 545, 730, 910, 1095, 1275, 1460, 1640]
            dates_tensor = torch.tensor(days, dtype=torch.long)
            return (img_tensor, dates_tensor), mask_tensor

train_ds = SilvicultureDataset(
    img_folder='data/patches/images/',
    mask_folder='data/patches/masks/',
    augment=True
)

train_dl = DataLoader(
    train_ds, 
    batch_size=16, 
    shuffle=True, 
    num_workers=4, 
    pin_memory=True 
)


if __name__ == "__main__":
    (imgs, dates), masks = next(iter(train_dl))
    
    print(f"Img Batch: {imgs.shape} -> (B, T, C, H, W)")
    print(f"Dates Batch: {dates.shape} -> (B, T)")
    print(f"Mask Batch: {masks.shape} -> (B, H, W)")
    print(f"Img Type: {imgs.dtype}")
    print(f"Mask Type: {masks.dtype}")

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    for i in range(4):
        img_t0 = imgs[i, 0].numpy() # Shape: (C, H, W)
        mask = masks[i].numpy()     # Shape: (H, W)
        
        rgb = np.stack([img_t0[2], img_t0[1], img_t0[0]], axis=-1)
        rgb = np.clip(rgb * 3.5, 0, 1)
        
        axes[0, i].imshow(rgb)
        axes[0, i].set_title(f"Patch {i} (RGB)")
        axes[0, i].axis('off')
        
        axes[1, i].imshow(mask, cmap='gray', vmin=0, vmax=1)
        axes[1, i].set_title(f"Mask {i}")
        axes[1, i].axis('off')
        
    plt.tight_layout()
    plt.show()
