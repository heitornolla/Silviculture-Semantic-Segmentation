# Semantic Segmentation for Silviculture

EDA, Data Acquisition and result analysis notebooks are available in `notebooks`. Scripts for ablations can be seen in `scripts`.

Codes were tested in a Galaxy Book4 Ultra running Ubuntu and CUDA 13.1. Image acquisiton codes ran in Google Colab and Google Earth Engine. 

## Training

Train a model with:

```bash
python train.py --model utae
```

Replace `utae` with any supported architecture (currently `convlstm`, `uconvlstm`, `bconvlstm`, `buconvlstm`, `convgru`, `unet3d`, `utae`).

## Evaluation

Evaluate a trained model with:

```bash
python test.py \
    --model utae \
    --weights utae_best.pth \
    --batch_size 8
```
