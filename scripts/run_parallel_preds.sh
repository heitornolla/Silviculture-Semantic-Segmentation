#!/bin/bash

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

MODELS=(
    "utae"
    "unet3d"
    #"fpn"
    "convlstm"
    #"bconvlstm"
    "convgru"
    "uconvlstm"
    "buconvlstm"
    "unet2d"
    "unetplusplus"
)

TIF_DIR="data/Dataset_Silvicultura/"
OUT_DIR="qgis_tifs/"

MAX_JOBS=3

run_prediction() {
    MODEL=$1
    CITY=$2

    python utils/generate_maps/predict_map.py \
        --model "$MODEL" \
        --weights "checkpoints/${MODEL}_best.pth" \
        --city "$CITY" \
        --tif_dir "$TIF_DIR" \
        --out_dir "$OUT_DIR"

    STATUS=$?

    if [ $STATUS -ne 0 ]; then
        echo "ERROR: $MODEL, $CITY"
    else
        echo "DONE: $MODEL, $CITY"
    fi
}

for MODEL in "${MODELS[@]}"; do
    for CITY in "${CITIES[@]}"; do

        # Awaits while processes are running
        # Change MAX_JOBS to allow for more
        while [ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]; do
            wait -n
        done

        run_prediction "$MODEL" "$CITY" &

    done
done

wait
