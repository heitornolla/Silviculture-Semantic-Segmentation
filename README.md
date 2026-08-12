# Semantic Segmentation for Silviculture

This repository investigates **semantic segmentation of silviculture using satellite image time series**. It builds upon the **U-TAE (U-Net with Temporal Attention Encoder)** architecture originally proposed for panoptic segmentation of satellite image time series and adapts it to the binary semantic segmentation task of distinguishing **silviculture** from **anthropic land use**.

The original paper is available [here](https://arxiv.org/abs/2107.07933).

## Motivation

Although semantic segmentation has become a well-established task in remote sensing, relatively little work specifically addresses **silviculture mapping**. Most existing approaches rely exclusively on **single-date imagery**, despite the fact that plantation forests exhibit temporal patterns that can be exploited using satellite image time series.

This project investigates whether temporal deep learning models can improve silviculture segmentation by leveraging seasonal and phenological information available in multi-temporal satellite observations.

## Research Objectives

The main goals of this project are:

- Adapt the **U-TAE** architecture for semantic segmentation.
- Benchmark temporal models against conventional segmentation networks.
- Evaluate the benefits of temporal information over static imagery.
- Establish strong baselines for silviculture mapping.

---

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
