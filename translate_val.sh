#!/bin/bash

#python translate.py -gpu 5 -model multi30k_unsup_model_acc_21.02_ppl_239.79_e2.pt -src data/multi30k/test2016.en.atok -tgt data/multi30k/test2016.de.atok -replace_unk -verbose -output multi30k.unsup.mse.noise.pred.2 > log.multi30k.unsup.mse.50.3.noise.2.translate
#perl tools/multi-bleu.perl data/multi30k/test2016.de.atok < multi30k.unsup.mse.noise.pred.2

srun --gres=gpu:1 --time=240:00:00 --partition=sm python translate.py -data_type audio -model $1 -src_dir data/speech -src scp:data/wsj/src-val-feats.scp \
    -output val.pred.$2.$1 -tgt data/wsj/tgt-val.txt.chars -gpu 0 -verbose -max_length 200 -beam_size $2 > log.val.pred.$2.$1 2>&1 &
