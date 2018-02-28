paste data/wsj/tgt-val-names.txt $1 > $1.ark
/usr/users/jdrexler/kaldi/src/bin/compute-wer --mode=present ark:data/wsj/tgt-val-chars.ark ark:$1.ark

python pred_to_words.py $1
paste data/wsj/tgt-val-names.txt $1.words > $1.words.ark
/usr/users/jdrexler/kaldi/src/bin/compute-wer --mode=present ark:data/wsj/tgt-val.txt.chars.words.ark ark:$1.words.ark

