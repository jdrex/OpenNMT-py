#!/bin/bash

# srun options
# srun --gres=gpu:1 --time=240:00:00 --partition=1080 

python train_auto.py -data data/multi30k.atok.low -save_model multi30k_auto_discrim_model -gpuid 0 > log.multi30k.auto.discrim.2
