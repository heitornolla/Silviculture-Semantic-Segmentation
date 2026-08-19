import argparse
import os
import glob
import math
import numpy as np
import rasterio

parser = argparse.ArgumentParser(description="Ensemble TIFs")
parser.add_argument("--city", type=str, required=True, help="City name (e.g.: Minas_Novas)")
parser.add_argument("--tif_dir", type=str, required=True, help="Folder with the TIF files")
parser.add_argument("--out", type=str, required=True, help="Output path")
parser.add_argument("--mode", type=str, required=True, choices=["majority", "aggregate"])
parser.add_argument("--model_type", type=str, default="all", choices=["2d", "3d", "all"], help="Filter which models to ensemble")

MODELS_2D = ["unet2d", "unetplusplus"]
MODELS_3D = ["utae", "unet3d", "fpn", "convlstm", "convgru", "bconvlstm", "uconvlstm", "buconvlstm"]

def main():
    args = parser.parse_args()
    
    # globs all tifs
    pattern = os.path.join(args.tif_dir, f"{args.city}_pred_*.tif")
    tifs = sorted(glob.glob(pattern))
    
    # removes the ensemble itself if it exists there
    tifs = [f for f in tifs if "ensemble" not in os.path.basename(f).lower()]
    
    # filters by model type (2D or 3D)
    if args.model_type != "all":
        filtered_tifs = []
        for tif in tifs:
            filename = os.path.basename(tif).lower()
            is_2d = any(m in filename for m in MODELS_2D)
            is_3d = any(m in filename for m in MODELS_3D)
            
            if args.model_type == "2d" and is_2d:
                filtered_tifs.append(tif)
            elif args.model_type == "3d" and is_3d:
                filtered_tifs.append(tif)
                
        tifs = filtered_tifs
    
    num_models = len(tifs)
    if num_models < 2:
        raise ValueError(f"Needs at least 2 models for the ensemble. Found: {num_models}")

    with rasterio.open(tifs[0]) as src0:
        meta   = src0.meta.copy()
        matrix = np.zeros((src0.height, src0.width), dtype=np.uint32)

    for tif_path in tifs:
        with rasterio.open(tif_path) as src:
            preds   = src.read(1)
            matrix += preds.astype(np.uint32)

    if args.mode == "aggregate":
        # At least one model marked as silviculture 
        thresh = 1
    elif args.mode == "majority":
        thresh = math.ceil(num_models / 2.0)
           
    end_matrix = (matrix >= thresh).astype(np.uint8)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with rasterio.open(args.out, 'w', **meta) as dst:
        dst.write(end_matrix, 1)
        
    print(f"Saved to: {args.out}\n")

if __name__ == "__main__":
    main()
