#!/bin/bash

set -e

CITIES=(
    "Belo_Oriente"
    "Carbonita"
    "Ipaba"
    "Itamarandiba"
    "Josenopolis"
    "Minas_Novas"
    "Santana_do_Paraiso"
    "Sao_Joao_do_Paraiso"
    "Taiobeiras"
    "Veredinha"
)

MODES=("majority" "aggregate")
MODEL_TYPES=("2d" "3d" "all")

TIF_DIR="qgis_tifs"
OUT_DIR="qgis_tifs/ensembles"
SCRIPT_PATH="utils/generate_maps/ensemble_tifs.py"

mkdir -p "$OUT_DIR"

for city in "${CITIES[@]}"; do    
    for mode in "${MODES[@]}"; do
        for model_type in "${MODEL_TYPES[@]}"; do
            
            output_file="${OUT_DIR}/${city}_${mode}_${model_type}.tif"
            
            python "$SCRIPT_PATH" \
                --city "$city" \
                --tif_dir "$TIF_DIR" \
                --mode "$mode" \
                --model_type "$model_type" \
                --out "$output_file"
                
        done
    done
done
