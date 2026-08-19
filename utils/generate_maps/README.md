# Generating Maps

This directory contains the tools required to generate **city-wide silviculture prediction masks** from trained models.

The workflow has two main stages:

1. **Generate a prediction map for each model and city** using `predict_map.py`.
2. **Combine predictions from multiple models** using majority voting with `ensemble_tifs.py`.

## Overview

Instead of predicting individual image patches, `predict_map.py` performs inference over an entire city's spatial extent using a **128 × 128 sliding window**. Each window is passed through a trained model, and its prediction is written back to the corresponding location in a city-wide raster.

Because different architectures may capture different spatial and temporal features, predictions from multiple models can be combined into a single consolidated mask through **ensemble learning**.

## 1. Generate City-Wide Predictions

The main inference script is:

```text
utils/generate_maps/predict_map.py
```

It loads a trained model and applies it to all `.tif` files belonging to a given city.

```bash
python utils/generate_maps/predict_map.py \
    --model utae \
    --weights checkpoints/utae_best.pth \
    --city Minas_Novas \
    --tif_dir data/Dataset_Silvicultura/ \
    --out_dir qgis_tifs/
```

Instead of running `predict_map.py` manually for every combination, you can use:

```bash
bash scripts/run_parallel_preds.sh
```

Increased `MAX_JOBS` depending on available CPU/GPU memory.

## 2. Ensemble Predictions

After generating the individual model predictions, they can be combined using:

```text
utils/generate_maps/ensemble_tifs.py
```

You can either use majority voting or aggregate all regions voted as silviculture.

Each input prediction is expected to contain binary values:

```text
0 = background
1 = silviculture
```

### Basic usage

```bash
python utils/generate_maps/ensemble_tifs.py \
    --city Minas_Novas \
    --tif_dir qgis_tifs/ \
    --mode majority \
    --out qgis_tifs/Minas_Novas_ensemble_majority.tif
```
