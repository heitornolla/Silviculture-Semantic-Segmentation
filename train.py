import argparse

import torch
import tqdm

import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

from src.silviculture_dataset import SilvicultureDataset

from src.backbones.convgru  import ConvGRU_Seg
from src.backbones.convlstm import ConvLSTM_Seg, BConvLSTM_Seg
from src.backbones.fpn      import FPNConvLSTM
from src.backbones.unet3d   import UNet3D
from src.backbones.utae     import UTAE, RecUNet

EPOCHS = 50
BATCH_SIZE = 8
LR = 0.001

CITIES = [
    'Belo_Oriente',
    'Carbonita',
    'Ipaba',
    'Itamarandiba',
    'Josenopolis',
    'Minas_Novas',
    'Santana_do_Paraiso',
    'Sao_Joao_do_Paraiso',
    'Taiobeiras',
    'Veredinha'
]

TEST_CITIES  = ['Minas_Novas', 'Itamarandiba']
VAL_CITIES   = ['Josenopolis', 'Veredinha']
TRAIN_CITIES = [c for c in CITIES if c not in TEST_CITIES and c not in VAL_CITIES]

parser = argparse.ArgumentParser(description="Semantic Segmentation Model Training")
parser.add_argument(
    "--model", 
    type=str, 
    default="utae", 
    choices=["utae", "unet3d", "fpn", "convlstm", "convgru", "bconvlstm", "uconvlstm", "buconvlstm"], 
    help="Model to be trained"
)


def get_model(model_name, input_dim=10, num_classes=2):
    enc_widths = [64, 64, 64, 128]
    dec_widths = [32, 32, 64, 128]
    out_conv_dims = [32, num_classes]

    if model_name == "utae":
        return UTAE(
            input_dim=input_dim,
            encoder_widths=enc_widths,
            decoder_widths=dec_widths,
            out_conv=out_conv_dims,
            padding_mode="reflect"
        )
        
    elif model_name == "unet3d":
        return UNet3D(
            in_channel=input_dim,
            n_classes=num_classes,
            pad_value=0
        )
        
    elif model_name == "fpn":
        return FPNConvLSTM(
            input_dim=input_dim,
            num_classes=num_classes,
            inconv=[32, 64],
            pad_value=0
        )
        
    elif model_name == "convlstm":
        return ConvLSTM_Seg(
            num_classes=num_classes, 
            input_size=(128, 128), 
            input_dim=input_dim, 
            hidden_dim=128, 
            kernel_size=(3, 3)
        )
        
    elif model_name == "bconvlstm":
        return BConvLSTM_Seg(
            num_classes=num_classes, 
            input_size=(128, 128), 
            input_dim=input_dim, 
            hidden_dim=128, 
            kernel_size=(3, 3)
        )
        
    elif model_name == "convgru":
        return ConvGRU_Seg(
            num_classes=num_classes, 
            input_size=(128, 128), 
            input_dim=input_dim, 
            hidden_dim=128, 
            kernel_size=(3, 3)
        )

    elif model_name == "uconvlstm":
        return RecUNet(
            input_dim=input_dim,
            temporal="lstm",
            encoder_widths=enc_widths,
            decoder_widths=dec_widths,
            out_conv=out_conv_dims,
            padding_mode="reflect"
        )
        
    elif model_name == "buconvlstm":
        return RecUNet(
            input_dim=input_dim,
            temporal="blstm",
            encoder_widths=enc_widths,
            decoder_widths=dec_widths,
            out_conv=out_conv_dims,
            padding_mode="reflect"
        )
         
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def main():
    args = parser.parse_args()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Training {args.model.upper()}")

    train_ds = SilvicultureDataset(
        img_folder='data/patches/images/', 
        mask_folder='data/patches/masks/', 
        allowed_cities=TRAIN_CITIES, 
        augment=True
    )

    val_ds = SilvicultureDataset(
        img_folder='data/patches/images/', 
        mask_folder='data/patches/masks/', 
        allowed_cities=VAL_CITIES, 
        augment=False
    )

    train_loader = DataLoader(train_ds, 
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              num_workers=4, 
                              pin_memory=True, 
                              persistent_workers=True
                            )
    
    val_loader  = DataLoader(val_ds, 
                            batch_size=BATCH_SIZE, 
                            shuffle=False, 
                            num_workers=4,
                            pin_memory=True,
                            persistent_workers=True
                            )

    # input_dim=10 (Sentinel-2), out_conv=[32, 2] (2 classes)
    model = get_model(args.model).to(DEVICE)

    print(f"Params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR)

    def calculate_iou(preds, labels):
        preds = preds.argmax(dim=1)
        intersection = ((preds == 1) & (labels == 1)).float().sum()
        union = ((preds == 1) | (labels == 1)).float().sum()
        if union == 0: return 0.0
        return (intersection / union).item()

    best_iou = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss, train_iou = 0, 0
        
        for (x, dates), y in tqdm.tqdm(train_loader):
            x, dates, y = x.to(DEVICE), dates.to(DEVICE), y.to(DEVICE)
            
            optimizer.zero_grad()

            # UTAE gets images and respective dates
            out = model(x, batch_positions=dates)
            
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_iou += calculate_iou(out, y)
            
        train_loss /= len(train_loader)
        train_iou /= len(train_loader)
        
        model.eval()
        val_loss, val_iou = 0, 0
        
        with torch.no_grad():
            for (x, dates), y in val_loader:
                x, dates, y = x.to(DEVICE), dates.to(DEVICE), y.to(DEVICE)
                out = model(x, batch_positions=dates)
                loss = criterion(out, y)
                
                val_loss += loss.item()
                val_iou += calculate_iou(out, y)
                
        val_loss /= len(val_loader)
        val_iou  /= len(val_loader)
        
        print(f"Epoch {epoch}/{EPOCHS} | Train Loss: {train_loss:.4f} | Train IoU: {train_iou:.4f} | Val Loss: {val_loss:.4f} | Val IoU: {val_iou:.4f}")
        
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), f"{args.model}_best.pth")
            print("--> Saved new best model")


if __name__ == "__main__":
    main()
