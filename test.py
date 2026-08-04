import argparse
from pathlib import Path

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

def visualize_predictions(model, dataloader, model_name):
    model.eval()

    save_dir = Path("preds") / model_name
    save_dir.mkdir(parents=True, exist_ok=True)

    img_id = 1

    with torch.no_grad():
        for (x, dates), y in dataloader:
            x, dates, y = x.to(DEVICE), dates.to(DEVICE), y.to(DEVICE)

            out = model(x, batch_positions=dates)
            preds = out.argmax(dim=1)

            x_cpu = x.cpu().numpy()
            y_cpu = y.cpu().numpy()
            preds_cpu = preds.cpu().numpy()

            for i in range(x.shape[0]):
                img_t0 = x_cpu[i, 0]

                # RGB visualization (Sentinel-2: B4, B3, B2)
                rgb = np.stack(
                    [img_t0[2], img_t0[1], img_t0[0]],
                    axis=-1
                )
                rgb = np.clip(rgb * 3.5, 0, 1)

                fig, axes = plt.subplots(1, 3, figsize=(15, 5))

                axes[0].imshow(rgb)
                axes[0].set_title("Original")
                axes[0].axis("off")

                axes[1].imshow(y_cpu[i], cmap="gray", vmin=0, vmax=1)
                axes[1].set_title("Ground Truth")
                axes[1].axis("off")

                axes[2].imshow(preds_cpu[i], cmap="gray", vmin=0, vmax=1)
                axes[2].set_title("Prediction")
                axes[2].axis("off")

                plt.tight_layout()
                plt.savefig(
                    save_dir / f"{img_id}.png",
                    dpi=200,
                    bbox_inches="tight"
                )
                plt.close(fig)

                img_id += 1


def main():
    args = parser.parse_args()

    model = get_model(args.model).to(DEVICE)
    model.load_state_dict(torch.load(args.weights, map_location=DEVICE))

    dataset = SilvicultureDataset(
        img_folder="data/patches/images/",
        mask_folder="data/patches/masks/",
        augment=False,
        allowed_cities=TEST_CITIES,
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

    #evaluate_model(model, test_loader, criterion)
    visualize_predictions(model, test_loader, args.model)


if __name__ == "__main__":
    main()
