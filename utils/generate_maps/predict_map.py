import sys
sys.path.append('.')

import argparse
import os
import math

import numpy as np
import torch
import rasterio
from rasterio.windows import Window
from tqdm import tqdm

from train import get_model

parser = argparse.ArgumentParser(description="City-Wide Inference")
parser.add_argument("--model", type=str, default="utae", help="Model Architecture")
parser.add_argument("--weights", type=str, required=True, help="Weights")
parser.add_argument("--city", type=str, required=True, help="City Name (e.g.: Minas_Novas)")
parser.add_argument("--tif_dir", type=str, default="data/tifs/", help="Folder w/ tifs")
parser.add_argument("--out_dir", type=str, default="preds/qgis/", help="Output folder")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATCH_SIZE = 128

def main():
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    
    model = get_model(args.model).to(DEVICE)
    model.load_state_dict(torch.load(args.weights, map_location=DEVICE))
    model.eval()
    
    is_2d = args.model in ["unet2d", "unetplusplus"]
    
    city_tifs = sorted([os.path.join(args.tif_dir, f) for f in os.listdir(args.tif_dir) if args.city in f and f.endswith('.tif')])
    
    # Read first TIFs metadata to recreate map
    with rasterio.open(city_tifs[0]) as src:
        meta = src.meta.copy()
        h_total = src.height
        w_total = src.width
        
    meta.update({
        'count': 1,
        'dtype': 'uint8',
        'nodata': 0
    })
    
    pred_map = np.zeros((h_total, w_total), dtype=np.uint8)
    
    # For the 3D models
    dates = [0, 180, 365, 545, 730, 910, 1095, 1275, 1460, 1640]
    dates_tensor = torch.tensor(dates, dtype=torch.long).unsqueeze(0).to(DEVICE) 
    
    # Sliding Window sweep
    lines  = math.ceil(h_total / PATCH_SIZE)
    cols   = math.ceil(w_total / PATCH_SIZE)

    with torch.no_grad():
        for i in tqdm(range(lines), desc="Processed lines"):
            for j in range(cols):
                row_start = i * PATCH_SIZE
                col_start = j * PATCH_SIZE
                
                # Window may be smaller than 128x128 on the borders
                h_window = min(PATCH_SIZE, h_total - row_start)
                w_window = min(PATCH_SIZE, w_total - col_start)
                window   = Window(col_start, row_start, w_window, h_window)
                
                time_series = []
                for tif in city_tifs:
                    with rasterio.open(tif) as src:
                        patch_img = src.read(window=window)
                        patch_img = np.nan_to_num(patch_img, nan=0).astype(np.float32) / 10000.0
                        time_series.append(patch_img)
                        
                # Time Stack: (T, C, H, W)
                city_tensor = np.stack(time_series, axis=0) 
                
                # Skip empty patches
                if np.all(city_tensor == 0):
                    continue
                
                # 0 pad for the input
                if h_window < PATCH_SIZE or w_window < PATCH_SIZE:
                    pad_h = PATCH_SIZE - h_window
                    pad_w = PATCH_SIZE - w_window
                    city_tensor = np.pad(city_tensor, ((0,0), (0,0), (0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
                
                # Batch dimension: (1, T, C, H, W)
                x = torch.from_numpy(city_tensor).unsqueeze(0).to(DEVICE)
                
                if is_2d:
                    out = model(x[:, 0])
                else:
                    out = model(x, batch_positions=dates_tensor)
                    
                pred = out.argmax(dim=1).squeeze(0).cpu().numpy() # Shape: (128, 128)
                
                # Remove padding (if any)
                pred_map[row_start:row_start+h_window, col_start:col_start+w_window] = pred[:h_window, :w_window]

    out_file = os.path.join(args.out_dir, f"{args.city}_pred_{args.model}.tif")
    
    with rasterio.open(out_file, 'w', **meta) as dst:
        dst.write(pred_map, 1)

if __name__ == "__main__":
    main()
