import argparse
import torch
import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader

from src.loss import DiceFocalLoss
from src.silviculture_dataset import SilvicultureDataset

from src.backbones.convgru  import ConvGRU_Seg
from src.backbones.convlstm import ConvLSTM_Seg, BConvLSTM_Seg
from src.backbones.fpn      import FPNConvLSTM
from src.backbones.unet3d   import UNet3D
from src.backbones.utae     import UTAE, RecUNet


EPOCHS = 50
BATCH_SIZE = 8
LR = 5e-4

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
    help="Model backbone to be trained"
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
        return UNet3D(in_channel=input_dim, n_classes=num_classes, pad_value=0)
    elif model_name == "fpn":
        return FPNConvLSTM(input_dim=input_dim, num_classes=num_classes, inconv=[32, 64], pad_value=0)
    elif model_name == "convlstm":
        return ConvLSTM_Seg(num_classes=num_classes, input_size=(128, 128), input_dim=input_dim, hidden_dim=128, kernel_size=(3, 3))
    elif model_name == "bconvlstm":
        return BConvLSTM_Seg(num_classes=num_classes, input_size=(128, 128), input_dim=input_dim, hidden_dim=128, kernel_size=(3, 3))
    elif model_name == "convgru":
        return ConvGRU_Seg(num_classes=num_classes, input_size=(128, 128), input_dim=input_dim, hidden_dim=128, kernel_size=(3, 3))
    elif model_name == "uconvlstm":
        return RecUNet(input_dim=input_dim, temporal="lstm", encoder_widths=enc_widths, decoder_widths=dec_widths, out_conv=out_conv_dims, padding_mode="reflect")
    elif model_name == "buconvlstm":
        return RecUNet(input_dim=input_dim, temporal="blstm", encoder_widths=enc_widths, decoder_widths=dec_widths, out_conv=out_conv_dims, padding_mode="reflect")
    else:
        raise ValueError(f"Unsupported model: {model_name}")

def get_intersection_union(preds, labels):
    """
    Returns the raw intersection and union counts for the positive class
    Allows for calculating the global IoU across the epoch
    """
    preds = preds.argmax(dim=1)
    intersection = ((preds == 1) & (labels == 1)).float().sum()
    union = ((preds == 1) | (labels == 1)).float().sum()
    return intersection.item(), union.item()

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

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, 
        num_workers=4, pin_memory=True, persistent_workers=True
    )

    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, 
        num_workers=4, pin_memory=True, persistent_workers=True
    )

    # input_dim=10 (Sentinel bands) num_classes=2 (Background, Silviculture)
    model = get_model(args.model).to(DEVICE)
    #print(f"Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    criterion = DiceFocalLoss(
        alpha=0.50,         # Reduce from 0.8 to avoid false positive bias
        gamma=2.0,
        dice_weight=1.0,
        focal_weight=0.5,
    )

    optimizer = optim.AdamW(model.parameters(), lr=LR)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=4
    )

    best_iou = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        train_intersection = 0.0
        train_union = 0.0
        
        train_pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for (x, dates), y in train_pbar:
            x, dates, y = x.to(DEVICE), dates.to(DEVICE), y.to(DEVICE)
            
            optimizer.zero_grad()

            out = model(x, batch_positions=dates)
            
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            i, u = get_intersection_union(out, y)
            train_intersection += i
            train_union += u
            
        train_loss /= len(train_loader)
        train_iou = train_intersection / (train_union + 1e-6)

        model.eval()
        val_loss = 0.0
        val_intersection = 0.0
        val_union = 0.0
        
        with torch.no_grad():
            for (x, dates), y in val_loader:
                x, dates, y = x.to(DEVICE), dates.to(DEVICE), y.to(DEVICE)
                
                out = model(x, batch_positions=dates)
                loss = criterion(out, y)
                
                val_loss += loss.item()
                i, u = get_intersection_union(out, y)
                val_intersection += i
                val_union += u
                
        val_loss /= len(val_loader)
        val_iou  = val_intersection / (val_union + 1e-6)
        
        print(f"Train Loss: {train_loss:.4f} | Train IoU: {train_iou:.4f} || Val Loss:   {val_loss:.4f} | Val IoU:   {val_iou:.4f}")
        
        scheduler.step(val_iou)
        
        if val_iou > best_iou:
            best_iou = val_iou
            save_path = f"{args.model}_best.pth"
            torch.save(model.state_dict(), save_path)

if __name__ == "__main__":
    main()
