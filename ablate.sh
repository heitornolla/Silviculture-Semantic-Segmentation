for m in utae unet3d convlstm convgru uconvlstm buconvlstm; do python train.py --model $m; done
# fpn and bconvlstm intentionally left our due to compute requirements