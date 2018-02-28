#!/bin/bash

#python translate.py -gpu 5 -model multi30k_unsup_model_acc_21.02_ppl_239.79_e2.pt -src data/multi30k/test2016.en.atok -tgt data/multi30k/test2016.de.atok -replace_unk -verbose -output multi30k.unsup.mse.noise.pred.2 > log.multi30k.unsup.mse.50.3.noise.2.translate
#perl tools/multi-bleu.perl data/multi30k/test2016.de.atok < multi30k.unsup.mse.noise.pred.2

for ((i=10;i<=$3;i+=10)); do
    srun --gres=gpu:1 --time=240:00:00 --partition=sm python translate.py -data_type audio -model $1*e$i.pt -src_dir data/speech -src scp:data/wsj/src-val-feats.scp \
	 -output val.pred.$2.$1_$i.pt -tgt data/wsj/tgt-val.txt.chars -gpu 0 -verbose -max_length 200 -beam_size $2 > log.val.pred.$2.$1_$i.pt 2>&1 &
done
