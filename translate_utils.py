#!/usr/bin/env python

from __future__ import division
from builtins import bytes
import os
import argparse
import math
import codecs
import torch

import onmt
import onmt.IO
import opts
from itertools import takewhile, count
try:
    from itertools import zip_longest
except ImportError:
    from itertools import izip_longest as zip_longest

def report_score(name, score_total, words_total):
    print("%s AVG SCORE: %.4f, %s PPL: %.4f" % (
        name, score_total / words_total,
        name, math.exp(-score_total/words_total)))


def get_src_words(src_indices, index2str):
    words = []
    raw_words = (index2str[i] for i in src_indices)
    words = takewhile(lambda w: w != onmt.IO.PAD_WORD, raw_words)
    return " ".join(words)


def translate(src, model, output):
    parser = argparse.ArgumentParser(description='translate.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    opts.translate_opts(parser)

    opt = parser.parse_known_args([])[0]
    if opt.batch_size != 1:
        print("WARNING: -batch_size isn't supported currently, "
            "we set it to 1 for now!")
        opt.batch_size = 1

    opt.src = src
    opt.model = model
    opt.output = output
    
    dummy_parser = argparse.ArgumentParser(description='train.py')
    opts.model_opts(dummy_parser)
    dummy_opt = dummy_parser.parse_known_args([])[0]

    opt.cuda = opt.gpu > -1
    if opt.cuda:
        torch.cuda.set_device(opt.gpu)
        
    translator = onmt.Translator(opt, dummy_opt.__dict__)
    out_file = codecs.open(opt.output, 'w', 'utf-8')
    gold_out_file = codecs.open("gold_" + opt.output, 'w', 'utf-8')

    #print "TRANSLATOR SOURCE VOCAB"
    #for i in range(len(translator.fields["src"].vocab.itos)):
    #    print i, translator.fields["src"].vocab.itos[i]
    #print
    
    data = onmt.IO.ONMTDataset(
        opt.src, opt.tgt, translator.fields,
        use_filter_pred=False)

    test_data = onmt.IO.OrderedIterator(
        dataset=data, device=opt.gpu,
        batch_size=opt.batch_size, train=False, sort=False,
        shuffle=False)

    counter = count(1)
    for batch in test_data:
        pred_batch, gold_batch, pred_scores, gold_scores, attn, src \
            = translator.translate(batch, data)

        # z_batch: an iterator over the predictions, their scores,
        # the gold sentence, its score, and the source sentence for each
        # sentence in the batch. It has to be zip_longest instead of
        # plain-old zip because the gold_batch has length 0 if the target
        # is not included.
        z_batch = zip_longest(
                pred_batch, gold_batch,
                pred_scores, gold_scores,
                (sent.squeeze(1) for sent in src.split(1, dim=1)))

        for pred_sents, gold_sent, pred_score, gold_score, src_sent in z_batch:
            n_best_preds = [" ".join(pred) for pred in pred_sents[:opt.n_best]]
            out_file.write('\n'.join(n_best_preds))
            out_file.write('\n')
            out_file.flush()

            words = get_src_words(src_sent, translator.fields["src"].vocab.itos)
            #print words
            gold_out_file.write(words)
            gold_out_file.write('\n')
            gold_out_file.flush()
