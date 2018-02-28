#!/usr/bin/env python

from __future__ import division

import numpy
print numpy.__path__

import os
import sys
import argparse
import torch
import torch.nn as nn
from torch import cuda

import onmt
import onmt.Models
import onmt.ModelConstructor
import onmt.modules
from onmt.Utils import aeq, use_gpu
import opts

print torch.cuda.is_available()
print cuda.device_count()
print cuda.current_device()

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


def make_train_data_iter(train_data, opt):
    """
    This returns user-defined train data iterator for the trainer
    to iterate over during each train epoch. We implement simple
    ordered iterator strategy here, but more sophisticated strategy
    like curriculum learning is ok too.
    """
    return onmt.IO.OrderedIterator(
                dataset=train_data, batch_size=opt.batch_size,
                device=opt.gpuid[0] if opt.gpuid else -1,
                repeat=False)


def make_valid_data_iter(valid_data, opt):
    """
    This returns user-defined validate data iterator for the trainer
    to iterate over during each validate epoch. We implement simple
    ordered iterator strategy here, but more sophisticated strategy
    is ok too.
    """
    return onmt.IO.OrderedIterator(
                dataset=valid_data, batch_size=opt.batch_size,
                device=opt.gpuid[0] if opt.gpuid else -1,
                train=False, sort=True)


def make_loss_compute(model, tgt_vocab, dataset, opt):
    """
    This returns user-defined LossCompute object, which is used to
    compute loss in train/validate process. You can implement your
    own *LossCompute class, by subclassing LossComputeBase.
    """
    if opt.copy_attn:
        compute = onmt.modules.CopyGeneratorLossCompute(
            model.generator, tgt_vocab, dataset, opt.copy_attn_force)
    else:
        compute = onmt.Loss.NMTLossCompute(model.generator, tgt_vocab)

    if use_gpu(opt):
        compute.cuda()

    return compute

def train_model(auto_models, valid_model, train_data, valid_data, fields_list, valid_fields, optims,
                discrim_models, discrim_optims, labels):

    #     train_model(models, valid_model, train, valid, fields, fields_valid, optims,
    #                 discrim_models, discrim_optims, advers_optims, labels)

    trunc_size = opt.truncated_decoder  # Badly named...
    shard_size = opt.max_generator_batches

    valid_iter = make_valid_data_iter(valid_data, opt)
    valid_loss = make_loss_compute(valid_model, valid_fields["tgt"].vocab, valid_data, opt)
    trainers = []
    trainers.append(onmt.Trainer(valid_model, valid_iter, valid_iter,
                                 valid_loss, valid_loss, optims[0],
                                 trunc_size, shard_size))

    for model, discrim_model, optim, label, train, fields in zip(auto_models, discrim_models, optims, labels, train_data, fields_list):
        train_iter = make_train_data_iter(train, opt)
        train_loss = make_loss_compute(model, fields["tgt"].vocab, train, opt)
        trainers.append(onmt.AdvTrainer(model, discrim_model, train_iter, valid_iter,
                                     train_loss, valid_loss, optim, label, 
                                     trunc_size, shard_size))

    discrim_trainers = []
    for model, discrim_optim, data in zip(discrim_models, discrim_optims, train_data):
        train_iter = make_train_data_iter(data, opt)
        discrim_trainers.append(onmt.DiscrimTrainer(model, train_iter, discrim_optim, shard_size))
    #for model, optim, data in zip(discrim_models, advers_optims, advers_data):
    #    train_iter = make_train_data_iter(data, opt)
    #    discrim_trainers.append(onmt.DiscrimTrainer(model, train_iter, optim, shard_size))

    for epoch in range(opt.start_epoch, opt.epochs + 1):
        print('')

        for label, trainer in zip(labels, discrim_trainers):
            # 1. Train for one epoch on the training set.
            train_stats = trainer.train(epoch, label, discrim_report_func)
            print('Train loss: %g' % train_stats.loss)
            #print('Train accuracy: %g' % train_stats.accuracy())

            if opt.exp_host:
                train_stats.log("train", experiment, optim.lr)

        for trainer in trainers[1:]:
            # 1. Train for one epoch on the training set.
            train_stats = trainer.train(epoch, report_func)
            print('Train perplexity: %g' % train_stats.ppl())
            print('Train accuracy: %g' % train_stats.accuracy())

            if opt.exp_host:
                train_stats.log("train", experiment, optim.lr)
        
        # 2. Validate on the validation set.
        valid_stats = trainers[0].validate()
        print('Validation perplexity: %g' % valid_stats.ppl())
        print('Validation accuracy: %g' % valid_stats.accuracy())

        # 3. Log to remote server.
        if opt.exp_host:
            valid_stats.log("valid", experiment, optim.lr)

        '''
        for trainer in trainers[1:]:
            # 4. Update the learning rate
            trainer.epoch_step(valid_stats.ppl(), epoch)
            
        for trainer in discrim_trainers:
            # 4. Update the learning rate
            trainer.epoch_step(valid_stats.ppl(), epoch)
        '''
                
        # 5. Drop a checkpoint if needed.
        if epoch >= opt.start_checkpoint_at:
            trainers[0].drop_checkpoint(opt, epoch, fields, valid_stats)

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


def load_fields(dataset, checkpoint, lang=None):
    if lang is not None:
        fields = onmt.IO.load_fields(torch.load(opt.data + '.' + lang + '.vocab.pt'))
    else:
        fields = onmt.IO.load_fields(torch.load(opt.data + '.vocab.pt'))

    fields = dict([(k, f) for (k, f) in fields.items()
                  if k in dataset.examples[0].__dict__])
    dataset.fields = fields

    if opt.train_from:
        print('Loading vocab from checkpoint at %s.' % opt.train_from)
        fields = onmt.IO.load_fields(checkpoint['vocab'])

    print(' * vocabulary size. source = %d; target = %d' %
          (len(fields['src'].vocab), len(fields['tgt'].vocab)))

    return fields

def collect_features(train, fields):
    # TODO: account for target features.
    # Also, why does fields need to have the structure it does?
    src_features = onmt.IO.collect_features(fields)
    aeq(len(src_features), train.n_src_feats)

    return src_features

'''
def build_model(model_opt, opt, fields, checkpoint):
    print('Building model...')
    src_auto_model, tgt_auto_model = onmt.ModelConstructor.make_auto_model(model_opt, fields,
                                                        use_gpu(opt), checkpoint)
    if len(opt.gpuid) > 1:
        print('Multi gpu training: ', opt.gpuid)
        src_auto_model = nn.DataParallel(src_auto_model, device_ids=opt.gpuid, dim=1)
        tgt_auto_model = nn.DataParallel(tgt_auto_model, device_ids=opt.gpuid, dim=1)
    print(src_auto_model)

    return src_auto_model, tgt_auto_model
'''

def build_model(model_opt, opt, fields, checkpoint):
    print('Building model...')
    model = onmt.ModelConstructor.make_base_model(model_opt, fields,
                                                  use_gpu(opt), checkpoint)
    if len(opt.gpuid) > 1:
        print('Multi gpu training: ', opt.gpuid)
        model = nn.DataParallel(model, device_ids=opt.gpuid, dim=1)
    print(model)

    return model


def build_optim(model, checkpoint):
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
            opt=opt
        )

    optim.set_parameters(model.parameters())

    return optim


def main():

    # Load train and validate data.
    # JD: fix this
    # - need to shuffle src and tgt (or split them beforehand)
    # - need two train created on the fly for cross-domain
    print("Loading train and validate data from '%s'" % opt.data)
    train_src = torch.load(opt.data + '.en.train.pt')    
    train_tgt = torch.load(opt.data + '.de.train.pt')
    
    train = torch.load(opt.data + '.train.pt')    
    valid = torch.load(opt.data + '.valid.pt')
    
    train = [train_src, train_tgt]
    #discrim_data = [discrim_src, discrim_tgt] # src_data --> 0, tgt_data --> 1
    #advers_data = [advers_src, advers_tgt] # src_data --> 1, tgt_data --> 0

    print(' * number of src training sentences: %d' % len(train_src))
    print(' * number of tgt training sentences: %d' % len(train_tgt))
    print(' * maximum batch size: %d' % opt.batch_size)

    # Load checkpoint if we resume from a previous training.
    if opt.train_from:
        print('Loading checkpoint from %s' % opt.train_from)
        checkpoint = torch.load(opt.train_from,
                                map_location=lambda storage, loc: storage)
        model_opt = checkpoint['opt']
        # I don't like reassigning attributes of opt: it's not clear
        opt.start_epoch = checkpoint['epoch'] + 1
    else:
        checkpoint = None
        model_opt = opt

    # Load fields generated from preprocess phase.
    fields_src = load_fields(train_src, checkpoint, lang='en')
    fields_tgt = load_fields(train_tgt, checkpoint, lang='de')
    fields_valid = load_fields(valid, checkpoint)
    fields_valid["src"].vocab = fields_src["src"].vocab
    fields_valid["tgt"].vocab = fields_tgt["tgt"].vocab
    fields = [fields_src, fields_tgt]

    print "train_src, src:", train_src.fields['src'].vocab.itos[10]
    print "train_src, tgt:", train_src.fields['tgt'].vocab.itos[10]
    print "train_tgt, src:", train_tgt.fields['src'].vocab.itos[10]
    print "train_tgt, tgt:", train_tgt.fields['tgt'].vocab.itos[10]
    print "valid, src:", valid.fields['src'].vocab.itos[10]
    print "valid, tgt:", valid.fields['tgt'].vocab.itos[10]

    '''
    # Collect features.
    src_features = collect_features(train_src, fields_src)
    for j, feat in enumerate(src_features):
         print(' * src feature %d size = %d' % (j, len(fields_src[feat].vocab)))
    tgt_features = collect_features(train_tgt, fields_tgt)
    for j, feat in enumerate(tgt_features):
         print(' * tgt feature %d size = %d' % (j, len(fields_tgt[feat].vocab)))
    '''
    
    # Build model.
    # src_auto_model, tgt_auto_model
    models = []
    for field in fields:
        models.append(build_model(model_opt, opt, field, checkpoint))

    valid_model = onmt.Models.NMTModel(models[0].encoder, models[1].decoder)
    valid_model.generator = models[1].generator
    if use_gpu(opt):
        valid_model.cuda()

    discrim_models = onmt.ModelConstructor.make_discriminators(opt, [m.encoder for m in models], use_gpu(opt))
    
    for model in models:
         tally_parameters(model)
    check_save_model_path()

    # Build optimizer.
    optims = []
    for model in models:
        optims.append(build_optim(model, checkpoint))

    discrim_optims = []
    advers_optims = []
    for model in discrim_models:
        discrim_optims.append(build_optim(model.classifier, checkpoint))
        advers_optims.append(build_optim(model.encoder, checkpoint))
    
    # Do training.
    labels = [0.1, 0.9]
    train_model(models, valid_model, train, valid, fields, fields_valid, optims,
                discrim_models, discrim_optims, labels)

if __name__ == "__main__":
    main()
