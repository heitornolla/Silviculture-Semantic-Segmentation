import argparse

import numpy as np
import matplotlib.pyplot as plt
import torch

from torch.utils.data import DataLoader

from train import get_model, TEST_CITIES
from src.silviculture_dataset import SilvicultureDataset 
from src.loss import DiceFocalLoss

parser = argparse.ArgumentParser(description="Semantic Segmentation Model Test")
parser.add_argument("--model", type=str, default="utae", help="Model Architecture")
parser.add_argument("--weights", type=str, required=True, help="Path to pth file")
parser.add_argument("--batch_size", type=int, default=8)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_intersection_union(preds, labels):
    preds = preds.argmax(dim=1)
    intersection = ((preds == 1) & (labels == 1)).float().sum()
    union = ((preds == 1) | (labels == 1)).float().sum()
    return intersection.item(), union.item()

def evaluate_model(model, dataloader, criterion):
    model.eval()
    test_loss = 0.0
    total_intersection = 0.0
    total_union = 0.0
    
    with torch.no_grad():
        for (x, dates), y in dataloader:
            x, dates, y = x.to(DEVICE), dates.to(DEVICE), y.to(DEVICE)
            
            out = model(x, batch_positions=dates)
            loss = criterion(out, y)
            
            test_loss += loss.item()
            
            i, u = get_intersection_union(out, y)
            total_intersection += i
            total_union += u
            
    test_loss /= len(dataloader)
    test_iou = total_intersection / (total_union + 1e-6)
    
    print(f"Loss: {test_loss:.4f} | IoU: {test_iou:.4f}")
    
    return test_loss, test_iou

def visualize_predictions(model, dataloader, num_samples=4):
    model.eval()
    
    (x, dates), y = next(iter(dataloader))
    x, dates, y = x.to(DEVICE), dates.to(DEVICE), y.to(DEVICE)
    
    with torch.no_grad():
        out = model(x, batch_positions=dates)
        preds = out.argmax(dim=1) 

    x_cpu = x.cpu().numpy()
    y_cpu = y.cpu().numpy()
    preds_cpu = preds.cpu().numpy()
    
    fig, axes = plt.subplots(3, num_samples, figsize=(4 * num_samples, 10))
    fig.suptitle(f"Qualitative Evaluation", fontsize=16)
    
    for i in range(min(num_samples, x.shape[0])):
        img_t0 = x_cpu[i, 0] 
        rgb = np.stack([img_t0[2], img_t0[1], img_t0[0]], axis=-1)
        rgb = np.clip(rgb * 3.5, 0, 1)
        
        axes[0, i].imshow(rgb)
        axes[0, i].set_title(f"Amostra {i} (RGB)")
        axes[0, i].axis("off")
        
        axes[1, i].imshow(y_cpu[i], cmap='gray', vmin=0, vmax=1)
        axes[1, i].set_title("Ground Truth")
        axes[1, i].axis("off")
        
        axes[2, i].imshow(preds_cpu[i], cmap='gray', vmin=0, vmax=1)
        axes[2, i].set_title("Model Prediction")
        axes[2, i].axis("off")
        
    plt.tight_layout()
    plt.show()

def main():
    args = parser.parse_args()
    
    model = get_model(args.model).to(DEVICE)
    model.load_state_dict(torch.load(args.weights, map_location=DEVICE))

    dataset = SilvicultureDataset(
        img_folder='data/patches/images/', 
        mask_folder='data/patches/masks/', 
        augment=False, 
        allowed_cities=TEST_CITIES
    )
    
    test_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    criterion = DiceFocalLoss(
        alpha=0.50,
        gamma=2.0,
        dice_weight=1.0,
        focal_weight=0.5,
    )
    
    evaluate_model(model, test_loader, criterion)
    visualize_predictions(model, test_loader)

if __name__ == "__main__":
    main()