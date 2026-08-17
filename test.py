import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import tqdm

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

def evaluate_and_predict(model, dataloader, criterion, model_name, is_2d, save_predictions=False):
    model.eval()

    test_loss = 0.0
    total_intersection = 0.0
    total_union = 0.0

    if save_predictions:
        save_dir = Path("preds") / model_name
        save_dir.mkdir(parents=True, exist_ok=True)
        img_id = 1

    with torch.no_grad():
        for batch_data, y in tqdm.tqdm(dataloader, desc=f"Evaluating {model_name.upper()}"):
            y = y.to(DEVICE)

            if is_2d:
                x = batch_data.to(DEVICE)
                out = model(x)
            else:
                x, dates = batch_data
                x, dates = x.to(DEVICE), dates.to(DEVICE)
                out = model(x, batch_positions=dates)

            loss = criterion(out, y)
            test_loss += loss.item()

            i, u = get_intersection_union(out, y)
            total_intersection += i
            total_union += u

            if save_predictions:
                preds = out.argmax(dim=1)
                x_cpu = x.cpu().numpy()
                y_cpu = y.cpu().numpy()
                preds_cpu = preds.cpu().numpy()

                for b in range(x.shape[0]):
                    # 2D -> (C, H, W) // 3D -> (T, C, H, W).
                    img_t0 = x_cpu[b] if is_2d else x_cpu[b, 0]

                    # RGB Viz (Sentinel-2: B4, B3, B2)
                    rgb = np.stack(
                        [img_t0[2], img_t0[1], img_t0[0]],
                        axis=-1
                    )
                    rgb = np.clip(rgb * 3.5, 0, 1)

                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

                    axes[0].imshow(rgb)
                    axes[0].set_title("Original")
                    axes[0].axis("off")

                    axes[1].imshow(y_cpu[b], cmap="gray", vmin=0, vmax=1)
                    axes[1].set_title("Ground Truth")
                    axes[1].axis("off")

                    axes[2].imshow(preds_cpu[b], cmap="gray", vmin=0, vmax=1)
                    axes[2].set_title("Prediction")
                    axes[2].axis("off")

                    plt.tight_layout()
                    plt.savefig(save_dir / f"{img_id:05d}.png", dpi=200, bbox_inches="tight")
                    plt.close(fig)

                    img_id += 1

    test_loss /= len(dataloader)
    test_iou = total_intersection / (total_union + 1e-6)

    print(f"\n{model_name.upper()} Loss: {test_loss:.4f} | IoU: {test_iou:.4f}\n")

def main():
    args = parser.parse_args()

    is_2d = args.model in ["unet2d", "unetplusplus"]
    dataset_mode = "2d" if is_2d else "3d"

    model = get_model(args.model).to(DEVICE)
    model.load_state_dict(torch.load(args.weights, map_location=DEVICE))

    dataset = SilvicultureDataset(
        img_folder="data/patches/images/",
        mask_folder="data/patches/masks/",
        augment=False,
        allowed_cities=TEST_CITIES,
        mode=dataset_mode
    )

    test_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    criterion = DiceFocalLoss(
        alpha=0.50,
        gamma=2.0,
        dice_weight=1.0,
        focal_weight=0.5,
    )

    evaluate_and_predict(model, test_loader, criterion, args.model, is_2d, save_predictions=True)

if __name__ == "__main__":
    main()
