#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import codecs
import torch

import onmt
import onmt.IO
import opts

parser = argparse.ArgumentParser(
    description='preprocess.py',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
opts.add_md_help_argument(parser)


# **Preprocess Options**
parser.add_argument('-config', help="Read options from this file")

parser.add_argument('-data_type', default="text",
                    help="Type of the source input. Options are [text|img].")
parser.add_argument('-data_img_dir', default=".",
                    help="Location of source images")

parser.add_argument('-train_src', required=True,
                    help="Path to the training source data")
parser.add_argument('-train_tgt', required=True,
                    help="Path to the training target data")
parser.add_argument('-valid_src', required=True,
                    help="Path to the validation source data")
parser.add_argument('-valid_tgt', required=True,
                    help="Path to the validation target data")

parser.add_argument('-save_data', required=True,
                    help="Output file for the prepared data")

parser.add_argument('-src_vocab',
                    help="Path to an existing source vocabulary")
parser.add_argument('-tgt_vocab',
                    help="Path to an existing target vocabulary")
parser.add_argument('-features_vocabs_prefix', type=str, default='',
                    help="Path prefix to existing features vocabularies")
parser.add_argument('-seed', type=int, default=3435,
                    help="Random seed")
parser.add_argument('-report_every', type=int, default=100000,
                    help="Report status every this many sentences")

opts.preprocess_opts(parser)

opt = parser.parse_args()
torch.manual_seed(opt.seed)


def main():
    print('Preparing training ...')
    with codecs.open(opt.train_src, "r", "utf-8") as src_file:
        src_line = src_file.readline().strip().split()
        _, _, n_src_features = onmt.IO.extract_features(src_line)
    with codecs.open(opt.train_tgt, "r", "utf-8") as tgt_file:
        tgt_line = tgt_file.readline().strip().split()
        _, _, n_tgt_features = onmt.IO.extract_features(tgt_line)

    src_fields = onmt.IO.get_fields(n_src_features, n_src_features)
    tgt_fields = onmt.IO.get_fields(n_tgt_features, n_tgt_features)
    print("Building Training...")
    train_src = onmt.IO.ONMTDataset(
        opt.train_src, opt.train_src, src_fields,
        opt.src_seq_length, opt.src_seq_length,
        src_seq_length_trunc=opt.src_seq_length_trunc,
        tgt_seq_length_trunc=opt.src_seq_length_trunc,
        dynamic_dict=opt.dynamic_dict)
    train_tgt = onmt.IO.ONMTDataset(
        opt.train_tgt, opt.train_tgt, tgt_fields,
        opt.tgt_seq_length, opt.tgt_seq_length,
        src_seq_length_trunc=opt.tgt_seq_length_trunc,
        tgt_seq_length_trunc=opt.tgt_seq_length_trunc,
        dynamic_dict=opt.dynamic_dict)
    
    print("Building Vocab...")
    onmt.IO.build_vocab(train_src, opt)
    onmt.IO.build_vocab(train_tgt, opt)

    print("Building Valid...")
    valid_src = onmt.IO.ONMTDataset(
        opt.valid_src, opt.valid_src, src_fields,
        opt.src_seq_length, opt.src_seq_length,
        src_seq_length_trunc=opt.src_seq_length_trunc,
        tgt_seq_length_trunc=opt.src_seq_length_trunc,
        dynamic_dict=opt.dynamic_dict)
    valid_tgt = onmt.IO.ONMTDataset(
        opt.valid_tgt, opt.valid_tgt, tgt_fields,
        opt.tgt_seq_length, opt.tgt_seq_length,
        src_seq_length_trunc=opt.tgt_seq_length_trunc,
        tgt_seq_length_trunc=opt.tgt_seq_length_trunc,
        dynamic_dict=opt.dynamic_dict)
    print("Saving train/valid/fields")

    # Can't save fields, so remove/reconstruct at training time.
    torch.save(onmt.IO.save_vocab(src_fields),
               open(opt.save_data + '.src.vocab.pt', 'wb'))
    torch.save(onmt.IO.save_vocab(tgt_fields),
               open(opt.save_data + '.tgt.vocab.pt', 'wb'))
    train_src.fields = []
    train_tgt.fields = []
    valid_src.fields = []
    valid_tgt.fields = []
    torch.save(train_src, open(opt.save_data + '.src.train.pt', 'wb'))
    torch.save(valid_src, open(opt.save_data + '.src.valid.pt', 'wb'))
    torch.save(train_tgt, open(opt.save_data + '.tgt.train.pt', 'wb'))
    torch.save(valid_tgt, open(opt.save_data + '.tgt.valid.pt', 'wb'))


if __name__ == "__main__":
    main()
