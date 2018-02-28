#!/usr/bin/env python

from __future__ import division

import os
import sys
import random

import torch
import torch.nn as nn
from torch import cuda

import onmt
import onmt.io
import onmt.Models
import onmt.ModelConstructor
import onmt.modules
from onmt.Utils import use_gpu
import opts

import argparse
import glob

#print torch.cuda.is_available()
#print cuda.device_count()
#print cuda.current_device()

parser = argparse.ArgumentParser(
    description='train.py',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# opts.py
opts.add_md_help_argument(parser)
opts.model_opts(parser)
opts.train_opts(parser)

opt = parser.parse_args()
if opt.word_vec_size != -1:
    opt.src_word_vec_size = opt.word_vec_size
    opt.tgt_word_vec_size = opt.word_vec_size

if opt.layers != -1:
    opt.enc_layers = opt.layers
    opt.dec_layers = opt.layers

opt.brnn = (opt.encoder_type == "brnn")
if opt.seed > 0:
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)

if opt.rnn_type == "SRU" and not opt.gpuid:
    raise AssertionError("Using SRU requires -gpuid set.")

if torch.cuda.is_available() and not opt.gpuid:
    print("WARNING: You have a CUDA device, should run with -gpuid 0")

if opt.gpuid:
    cuda.set_device(opt.gpuid[0])
    if opt.seed > 0:
        torch.cuda.manual_seed(opt.seed)

if len(opt.gpuid) > 1:
    sys.stderr.write("Sorry, multigpu isn't supported yet, coming soon!\n")
    sys.exit(1)


# Set up the Crayon logging server.
if opt.exp_host != "":
    from pycrayon import CrayonClient
    cc = CrayonClient(hostname=opt.exp_host)

    experiments = cc.get_experiment_names()
    print(experiments)
    if opt.exp in experiments:
        cc.remove_experiment(opt.exp)
    experiment = cc.create_experiment(opt.exp)

def discrim_report_func(epoch, batch, num_batches,
                start_time, lr, report_stats):
    """
    This is the user-defined batch-level traing progress
    report function.

    Args:
        epoch(int): current epoch count.
        batch(int): current batch count.
        num_batches(int): total number of batches.
        start_time(float): last report time.
        lr(float): current learning rate.
        report_stats(Statistics): old Statistics instance.
    Returns:
        report_stats(Statistics): updated Statistics instance.
    """
    if batch % opt.report_every == -1 % opt.report_every:
        report_stats.output_loss(epoch, batch+1, num_batches, start_time)
        if opt.exp_host:
            report_stats.log("progress", experiment, lr)
        report_stats = onmt.Statistics()

    return report_stats

def report_func(epoch, batch, num_batches,
                start_time, lr, report_stats):
    """
    This is the user-defined batch-level traing progress
    report function.

    Args:
        epoch(int): current epoch count.
        batch(int): current batch count.
        num_batches(int): total number of batches.
        start_time(float): last report time.
        lr(float): current learning rate.
        report_stats(Statistics): old Statistics instance.
    Returns:
        report_stats(Statistics): updated Statistics instance.
    """
    if batch % opt.report_every == -1 % opt.report_every:
        report_stats.output(epoch, batch+1, num_batches, start_time)
        if opt.exp_host:
            report_stats.log("progress", experiment, lr)
        report_stats = onmt.Statistics()

    return report_stats


def make_train_data_iter(train_dataset, opt, bs=None,s=False):
    """
    This returns user-defined train data iterator for the trainer
    to iterate over during each train epoch. We implement simple
    ordered iterator strategy here, but more sophisticated strategy
    like curriculum learning is ok too.
    """
    # Sort batch by decreasing lengths of sentence required by pytorch.
    # sort=False means "Use dataset's sortkey instead of iterator's".
    if not bs:
        bs = opt.batch_size
    return onmt.io.OrderedIterator(
                dataset=train_dataset, batch_size=bs,
                device=opt.gpuid[0] if opt.gpuid else -1,
                repeat=False, #shuffleK=0,
                sort=s, sort_within_batch=True)

def make_valid_data_iter(valid_dataset, opt):
    """
    This returns user-defined validate data iterator for the trainer
    to iterate over during each validate epoch. We implement simple
    ordered iterator strategy here, but more sophisticated strategy
    is ok too.
    """
    # Sort batch by decreasing lengths of sentence required by pytorch.
    # sort=False means "Use dataset's sortkey instead of iterator's".
    return onmt.io.OrderedIterator(
                dataset=valid_dataset, batch_size=opt.valid_batch_size,
                device=opt.gpuid[0] if opt.gpuid else -1,
                train=False, sort=False, sort_within_batch=True)


def make_loss_compute(model, tgt_vocab, dataset, opt, weight=False):
    """
    This returns user-defined LossCompute object, which is used to
    compute loss in train/validate process. You can implement your
    own *LossCompute class, by subclassing LossComputeBase.
    """
    if opt.copy_attn:
        compute = onmt.modules.CopyGeneratorLossCompute(
            model.generator, tgt_vocab, dataset, opt.copy_attn_force)
    else:
        w = 1.0
        if weight:
            w = opt.auto_lambda
            if opt.weighted:
                w = w/float(opt.mult*opt.t_mult)
                
        compute = onmt.Loss.NMTLossCompute(model.generator, tgt_vocab,
                                           opt.label_smoothing, w)

    if use_gpu(opt):
        compute.cuda()

    return compute


def train_model(model, train_dataset, valid_dataset, fields, 
                text_model, text_train_dataset, text_fields,
                speech_model, speech_train_dataset,
                discrim_models, discrim_optims,
                optim, model_opt):

    train_iter = make_train_data_iter(train_dataset, opt)
    valid_iter = make_valid_data_iter(valid_dataset, opt)

    text_train_iter = make_train_data_iter(text_train_dataset, opt)
    speech_train_iter = make_train_data_iter(speech_train_dataset, opt)

    train_loss = make_loss_compute(model, fields["tgt"].vocab,
                                   train_dataset, opt)
    text_loss = make_loss_compute(model, fields["tgt"].vocab,
                                   train_dataset, opt, True)

    valid_loss = make_loss_compute(model, fields["tgt"].vocab,
                                   valid_dataset, opt)

    trunc_size = opt.truncated_decoder  # Badly named...
    shard_size = opt.max_generator_batches
    data_type = train_dataset.data_type

    speech_lambda = opt.auto_lambda
    if model_opt.weighted:
       speech_lambda = speech_lambda/float(opt.mult)

    print "label:", model_opt.gen_label
    trainer = onmt.AudioTextSpeechTrainerAdv(model, train_iter, valid_iter,
                                             text_model, text_train_iter,
                                             speech_model, speech_train_iter,
                                             train_loss, text_loss, valid_loss, optim,
                                             discrim_models, [model_opt.gen_label, model_opt.gen_label], model_opt.gen_lambda,  speech_lambda, 
                                             trunc_size, shard_size, data_type,
                                             model_opt.mult, model_opt.t_mult)

    speech_train_iter = make_train_data_iter(speech_train_dataset, opt, 32)
    text_train_iter = make_train_data_iter(text_train_dataset, opt, 32)
    discrim_trainer = onmt.DiscrimTrainer(discrim_models, [speech_train_iter, text_train_iter], discrim_optims, [0.1, 0.9], shard_size)

    for epoch in range(opt.start_epoch, opt.epochs + 1):
        print('')

        train_stats = discrim_trainer.train(epoch, discrim_report_func)
        print('Discrim loss: %g' % train_stats.loss)

        # 1. Train for one epoch on the training set.
        train_stats, text_train_stats, speech_train_stats = trainer.train(epoch, report_func)
        print('Train perplexity: %g' % train_stats.ppl())
        print('Train accuracy: %g' % train_stats.accuracy())
        #print('Text perplexity: %g' % text_train_stats.ppl())
        #print('Text accuracy: %g' % text_train_stats.accuracy())
        print('Speech MSE: %g' % speech_train_stats.loss)

        # 2. Validate on the validation set.
        valid_stats = trainer.validate()
        print('Validation perplexity: %g' % valid_stats.ppl())
        print('Validation accuracy: %g' % valid_stats.accuracy())

        # 3. Log to remote server.
        if opt.exp_host:
            train_stats.log("train", experiment, optim.lr)
            valid_stats.log("valid", experiment, optim.lr)

        # 4. Update the learning rate
        trainer.epoch_step(valid_stats.ppl(), epoch)

        # 5. Drop a checkpoint if needed.
        if epoch >= opt.start_checkpoint_at:
            trainer.drop_checkpoint(model_opt, epoch, fields, valid_stats)
            discrim_trainer.drop_checkpoint(model_opt, epoch, fields, valid_stats)

def check_save_model_path():
    save_model_path = os.path.abspath(opt.save_model)
    model_dirname = os.path.dirname(save_model_path)
    if not os.path.exists(model_dirname):
        os.makedirs(model_dirname)


def tally_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    print('* number of parameters: %d' % n_params)
    enc = 0
    dec = 0
    for name, param in model.named_parameters():
        if 'encoder' in name:
            enc += param.nelement()
        elif 'decoder' or 'generator' in name:
            dec += param.nelement()
    print('encoder: ', enc)
    print('decoder: ', dec)


def load_dataset(data_type, mode):
    assert data_type in ["train", "valid"]

    print("Loading %s data from '%s'" % (data_type, opt.data))
    pts = glob.glob(opt.data + '.' + data_type + '.[0-9]*.pt')
    if pts:
        # Multiple onmt.io.*Dataset's, coalesce all.
        # torch.load loads them imemediately, which might eat up
        # too much memory. A lazy load would be better, but later
        # when we create data iterator, it still requires these
        # data to be loaded. So it seams we don't have a good way
        # to avoid this now.
        datasets = []
        for pt in pts:
            datasets.append(torch.load(pt))
        dataset = onmt.io.ONMTDatasetBase.coalesce_datasets(datasets)
    else:
        # Only one onmt.io.*Dataset, simple!
        if mode == "asr":
            dataset = torch.load(opt.data + '.' + data_type + '.pt')
        if mode == "speech":
            dataset = torch.load(opt.speech_data + '.' + data_type + '.pt')
        if mode == "text":
            dataset = torch.load(opt.text_data + '.' + data_type + '.pt')

    print(' * number of %s sentences: %d' % (data_type, len(dataset)))

    return dataset


def load_fields(train_dataset, valid_dataset, checkpoint, mode="audio", tgt_fields=None):
    data_type = train_dataset.data_type

    if mode == "audio":
        data_path = opt.data
    else:
        data_path = opt.text_data
    fields = onmt.io.load_fields_from_vocab(
                torch.load(data_path + '.vocab.pt'), data_type)
    fields = dict([(k, f) for (k, f) in fields.items()
                  if k in train_dataset.examples[0].__dict__])

    if tgt_fields:
        fields['tgt'] = tgt_fields
        
    # We save fields in vocab.pt, so assign them back to dataset here.
    train_dataset.fields = fields
    valid_dataset.fields = fields

    if opt.train_from and mode != "text":
        print('Loading vocab from checkpoint at %s.' % opt.train_from)
        fields = onmt.io.load_fields_from_vocab(
                    checkpoint['vocab'], data_type)

    if data_type == 'text' or mode == 'text':
        print(' * vocabulary size. source = %d; target = %d' %
              (len(fields['src'].vocab), len(fields['tgt'].vocab)))
        print(fields['tgt'].vocab.itos[0], fields['tgt'].vocab.itos[10])
    else:
        print(' * vocabulary size. target = %d' %
              (len(fields['tgt'].vocab)))
        print(fields['tgt'].vocab.itos[0], fields['tgt'].vocab.itos[10])

    return fields


def collect_report_features(fields):
    src_features = onmt.io.collect_features(fields, side='src')
    tgt_features = onmt.io.collect_features(fields, side='tgt')

    for j, feat in enumerate(src_features):
        print(' * src feature %d size = %d' % (j, len(fields[feat].vocab)))
    for j, feat in enumerate(tgt_features):
        print(' * tgt feature %d size = %d' % (j, len(fields[feat].vocab)))


def build_model(model_opt, opt, fields, text_fields, checkpoint):
    print('Building model...')
    model, text_model, speech_model = onmt.ModelConstructor.make_audio_text_model_from_text(model_opt, fields, text_fields,
                                                  use_gpu(opt), checkpoint)
    if len(opt.gpuid) > 1:
        print('Multi gpu training: ', opt.gpuid)
        model = nn.DataParallel(model, device_ids=opt.gpuid, dim=1)
        text_model = nn.DataParallel(text_model, device_ids=opt.gpuid, dim=1)
    print(model)
    print
    print(text_model)
    print
    print(speech_model)

    return model, text_model, speech_model


def build_optim(model, text_model, speech_model, checkpoint):
    if opt.train_from:
        print('Loading optimizer from checkpoint.')
        optim = checkpoint['optim']
        optim.optimizer.load_state_dict(
            checkpoint['optim'].optimizer.state_dict())
    else:
        # what members of opt does Optim need?
        optim = onmt.Optim(
            opt.optim, opt.learning_rate, opt.max_grad_norm,
            lr_decay=opt.learning_rate_decay,
            start_decay_at=opt.start_decay_at,
            beta1=opt.adam_beta1,
            beta2=opt.adam_beta2,
            adagrad_accum=opt.adagrad_accumulator_init,
            decay_method=opt.decay_method,
            warmup_steps=opt.warmup_steps,
            model_size=opt.rnn_size)

    optim.set_parameters(model.encoder.parameters())
    #optim.set_parameters(text_model.encoder.parameters())
    if speech_model:
        optim.set_parameters(speech_model.decoder.parameters())

    return optim

def build_discrim_optim(model, checkpoint, ind):
    '''
    if opt.train_from:
        if ind == 0:
            print('Loading optimizer from checkpoint.')
            optim = checkpoint['src_optim']
            optim.optimizer.load_state_dict(
                checkpoint['src_optim'].optimizer.state_dict())
        else:
            print('Loading optimizer from checkpoint.')
            optim = checkpoint['tgt_optim']
            optim.optimizer.load_state_dict(
                checkpoint['tgt_optim'].optimizer.state_dict())
    else:
    '''
    # what members of opt does Optim need?
    optim = onmt.Optim(
            opt.optim, opt.learning_rate, opt.max_grad_norm,
            lr_decay=opt.learning_rate_decay,
            start_decay_at=opt.start_decay_at,
            beta1=opt.adam_beta1,
            beta2=opt.adam_beta2,
            adagrad_accum=opt.adagrad_accumulator_init,
            decay_method=opt.decay_method,
            warmup_steps=opt.warmup_steps,
            model_size=opt.rnn_size)

    optim.set_parameters(model.parameters())

    return optim


def main():

    # Load train and validate data.
    train_dataset = load_dataset("train", "asr")
    valid_dataset = load_dataset("valid", "asr")
    text_train_dataset = load_dataset("train", "text")
    text_valid_dataset = load_dataset("valid", "text")
    speech_train_dataset = load_dataset("train", "speech")
    
    print(' * maximum batch size: %d' % opt.batch_size)

    # Load checkpoint if we resume from a previous training.
    if opt.train_from:
        print('Loading checkpoint from %s' % opt.train_from)
        checkpoint = torch.load(opt.train_from,
                                map_location=lambda storage, loc: storage)
        model_opt = opt #checkpoint['opt']
        # I don't like reassigning attributes of opt: it's not clear.
        #opt.start_epoch = checkpoint['epoch'] + 1

        discrim_checkpoint = None #torch.load(opt.train_from + ".disc",
        #                        map_location=lambda storage, loc: storage)
    else:
        checkpoint = None
        discrim_checkpoint = None
        model_opt = opt

    # Load fields generated from preprocess phase.
    text_fields = load_fields(text_train_dataset, text_valid_dataset, checkpoint, "text")
    fields = load_fields(train_dataset, valid_dataset, checkpoint) #, tgt_fields=text_fields['tgt'])
    speech_fields = load_fields(speech_train_dataset, valid_dataset, checkpoint)
    
    print(fields['tgt'].vocab.itos)
    print(text_fields['tgt'].vocab.itos)

    # Report src/tgt features.
    collect_report_features(fields)
    collect_report_features(text_fields)

    # Build model.
    model, text_model, speech_model = build_model(model_opt, opt, fields, text_fields, checkpoint)
    if opt.no_speech_decoder:
        speech_model = None
        
    discrim_models = onmt.ModelConstructor.make_discriminators(opt, [m.encoder for m in [model, text_model]], use_gpu(opt), discrim_checkpoint)
    
    tally_parameters(model)
    check_save_model_path()

    # Build optimizer.
    optim = build_optim(model, text_model, speech_model, checkpoint)
    discrim_optims = []
    for i, m in enumerate(discrim_models):
        discrim_optims.append(build_discrim_optim(m.classifier, discrim_checkpoint, i))

    # Do training.
    train_model(model, train_dataset, valid_dataset, fields,
                text_model, text_train_dataset, text_fields,
                speech_model, speech_train_dataset,
                discrim_models, discrim_optims, 
                optim, model_opt)


if __name__ == "__main__":
    main()
