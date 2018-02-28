#!/usr/bin/env python

from __future__ import division

import sys
import numpy
print numpy.__path__
print sys.path
import codecs

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

import translate_utils

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

translate_parser = argparse.ArgumentParser(
    description='translate.py',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
opts.translate_opts(translate_parser)
translate_opt = translate_parser.parse_known_args([])[0]
translate_opt.cuda = opt.gpuid[0] > -1

opt.lower = True

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
                repeat=False, dropout=0.1, shuffleK=3)


def make_cd_data_iter(src, tgt, fields, opt):

    '''
    with codecs.open(src, "r", "utf-8") as src_file:
        src_line = src_file.readline().strip().split()
        _, _, n_src_features = onmt.IO.extract_features(src_line)
    with codecs.open(tgt, "r", "utf-8") as tgt_file:
        tgt_line = tgt_file.readline().strip().split()
        _, _, n_tgt_features = onmt.IO.extract_features(tgt_line)

    fields = onmt.IO.get_fields(n_src_features, n_tgt_features)
    print("Building Training...")
    '''
    print "MAKING CD DATASET"
    print src
    print tgt
    train = onmt.IO.ONMTDataset(src, tgt, fields, 50, 50,
                                src_seq_length_trunc=0,
                                tgt_seq_length_trunc=0)
    
    '''
    opt.src_vocab_size = 50000
    opt.src_words_min_frequency = 0
    opt.tgt_vocab_size = 50000
    opt.tgt_words_min_frequency = 0
    opt.share_vocab = False
    
    print("Building Vocab...")
    onmt.IO.build_vocab(train, opt)
    '''
    
    return onmt.IO.OrderedIterator(
                dataset=train, batch_size=opt.batch_size,
                device=opt.gpuid[0] if opt.gpuid else -1,
                repeat=False, dropout=0.1, shuffleK=3)

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

def train_model(auto_models, cd_models, train_data, valid_data, fields_list, valid_fields, optims, cd_optims,
                discrim_models, discrim_optims, labels, advers_optims):

    trunc_size = opt.truncated_decoder  # Badly named...
    shard_size = opt.max_generator_batches

    dataFiles = ['data/multi30k/train.de.atok', 'data/multi30k/train.en.atok']
    #dataFiles = ['data/multi30k/val.de.atok', 'data/multi30k/val.en.atok']

    valid_trainers = []
    #translators = []
    for m, d, f in zip(cd_models, valid_data, valid_fields):
        valid_iter = make_valid_data_iter(d, opt)
        valid_loss = make_loss_compute(m, f["tgt"].vocab, d, opt)
        valid_trainers.append(onmt.Trainer(m, valid_iter, valid_iter,
                                        valid_loss, valid_loss, optims[0],
                                        trunc_size, shard_size))
        #translators.append(onmt.Translator(translate_opt, opt, f, m))

    '''
    translate_iters = []
    for d, t in zip(['data//multi30k/train.de.atok', 'data/multi30k/train.en.atok'], translators):
        #translate_iters.append(make_valid_data_iter(d, opt))
        data = onmt.IO.ONMTDataset(d, None, t.fields, use_filter_pred=False)
        test_data = onmt.IO.OrderedIterator(dataset=data, device=opt.gpuid[0] if opt.gpuid else -1,
                                            batch_size=opt.batch_size,
                                            train=False, sort=False, shuffle=False)
        translate_iters.append(test_data)
    '''

    cd_model_dict = dict()
    cd_model_dict["src-tgt"] = cd_models[0]
    cd_model_dict["tgt-src"] = cd_models[1]
    
    src_train_iter = make_train_data_iter(train_data[0], opt)
    tgt_train_iter = make_train_data_iter(train_data[1], opt)
    src_train_loss = make_loss_compute(auto_models[0], fields_list[0]["tgt"].vocab, train_data[0], opt) # src --> src vocab
    tgt_train_loss = make_loss_compute(auto_models[1], fields_list[1]["tgt"].vocab, train_data[1], opt) # tgt --> tgt vocab
    src_tgt_loss = make_loss_compute(cd_models[0], fields_list[1]["tgt"].vocab, train_data[0], opt) # tgt --> tgt vocab
    tgt_src_loss = make_loss_compute(cd_models[1], fields_list[0]["tgt"].vocab, train_data[1], opt) # src --> src vocab
    unsup_trainer = onmt.UnsupTrainer(auto_models, cd_models, discrim_models, [src_train_iter, tgt_train_iter], valid_iter,
                                     [src_train_loss, tgt_train_loss], [src_tgt_loss, tgt_src_loss], valid_loss, optims, cd_optims,
                                     [0.9, 0.9], trunc_size, shard_size)

    src_train_iter = make_train_data_iter(train_data[0], opt)
    tgt_train_iter = make_train_data_iter(train_data[1], opt)
    discrim_trainer = onmt.DiscrimTrainer(discrim_models, [src_train_iter, tgt_train_iter], discrim_optims, labels, shard_size)

    '''
    src_train_iter = make_train_data_iter(train_data[0], opt)
    tgt_train_iter = make_train_data_iter(train_data[1], opt)
    advers_trainer = onmt.DiscrimTrainer(discrim_models, [src_train_iter, tgt_train_iter], advers_optims, [0.9, 0.9], shard_size)
    '''
    
    for epoch in range(opt.start_epoch, opt.epochs + 1):
        print('')

        # 1. Train for one epoch on the training set.
        train_stats = discrim_trainer.train(epoch, discrim_report_func)
        print('Train loss: %g' % train_stats.loss)

        if opt.exp_host:
            train_stats.log("train", experiment, optim.lr)

        # 1. Train for one epoch on the training set.
        train_stats = unsup_trainer.train(epoch, report_func)
        print('Train perplexity: %g' % train_stats.ppl())
        print('Train accuracy: %g' % train_stats.accuracy())

        if opt.exp_host:
            train_stats.log("train", experiment, optim.lr)
        
        # 2. Validate on the validation set.

        for valid_trainer, v_f, srcF, name in zip(valid_trainers, valid_fields, dataFiles, ["src-tgt", "tgt-src"]):
            valid_stats = valid_trainer.validate()
            print('Validation perplexity: %g' % valid_stats.ppl())
            print('Validation accuracy: %g' % valid_stats.accuracy())

            '''
            if epoch > 1:
                um = unsup_trainer.drop_checkpoint(opt, epoch, fields_list, valid_stats)
                tgtF = "src." + str(epoch) + ".txt"
                translate_utils.translate(dataFiles[0], "src_" + um, tgtF)
                tgtF = "tgt." + str(epoch) + ".txt"
                translate_utils.translate(dataFiles[1], "tgt_" + um, tgtF)

                break
            '''
            # 5. Drop a checkpoint if needed.
            if epoch >= opt.start_checkpoint_at and epoch >= 10:
                m = valid_trainer.drop_checkpoint(opt, epoch, v_f, valid_stats)
                tgtF = name + "." + str(epoch) + "a.txt"
                translate_utils.translate(srcF, m, tgtF)
                unsup_trainer.cd_iters[name] = make_cd_data_iter("gold_" + tgtF, tgtF, v_f, opt)
                unsup_trainer.trainCD = True
        
        # 3. Log to remote server.
        if opt.exp_host:
            valid_stats.log("valid", experiment, optim.lr)

        #cd_data = []
        '''
        for translator, data, data_iter, name in zip(translators, train_data, translate_iters, ["src-tgt", "tgt-src"]):
            outF = open(name + "." + str(epoch) + ".txt", "w")
            #translations = []
            for batch in data_iter:
                pred_batch, _, _, _, _, _ = translator.translate(batch, data)
                for p in pred_batch:
                    outF.write(p[0] + "\n")
                #translations.append([p[0] for p in pred_batch])
            outF.close()
            #cd_data.append(Dataset(data, translations)
        '''
        '''
        for trainer in trainers[1:]:
            # 4. Update the learning rate
            trainer.epoch_step(valid_stats.ppl(), epoch)
            
        for trainer in discrim_trainers:
            # 4. Update the learning rate
            trainer.epoch_step(valid_stats.ppl(), epoch)
        '''
                

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


def build_optim(model, checkpoint, method):
    '''
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
    '''
    if method == "adam":
        optim = onmt.Optim(
            "adam", 3e-4, opt.max_grad_norm,
            lr_decay=opt.learning_rate_decay,
            start_decay_at=opt.start_decay_at,
            opt=opt, beta1=0.5
        )
    elif method == "rmsprop":
        optim = onmt.Optim(
            "rmsprop", 5e-4, opt.max_grad_norm,
            lr_decay=opt.learning_rate_decay,
            start_decay_at=opt.start_decay_at,
            opt=opt
        )
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

    src_lang = 'de'
    tgt_lang = 'en'
    
    # Load train and validate data.
    # JD: fix this
    # - need to shuffle src and tgt (or split them beforehand)
    # - need two train created on the fly for cross-domain
    print("Loading train and validate data from '%s'" % opt.data)
    train_src = torch.load(opt.data + '.' + src_lang + '.train.pt')    
    train_tgt = torch.load(opt.data + '.' + tgt_lang + '.train.pt')
        
    #train_src_tgt = torch.load(opt.data + '.' + src_lang + '-' + tgt_lang + '.train.pt')
    valid_src_tgt = torch.load(opt.data + '.' + src_lang + '-' + tgt_lang + '.valid.pt')
    #train_tgt_src = torch.load(opt.data + '.' + tgt_lang + '-' + src_lang + '.train.pt')
    valid_tgt_src = torch.load(opt.data + '.' + tgt_lang + '-' + src_lang + '.valid.pt')
    
    train = [train_src, train_tgt]
    valid = [valid_src_tgt, valid_tgt_src]
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
    fields_src = load_fields(train_src, checkpoint, lang=src_lang)
    fields_tgt = load_fields(train_tgt, checkpoint, lang=tgt_lang)
    #fields_src_tgt = load_fields(train_src_tgt, checkpoint, lang=src_lang + '-' + tgt_lang)
    #fields_tgt_src = load_fields(train_tgt_src, checkpoint, lang=tgt_lang + '-' + src_lang)
    fields_valid_src_tgt = load_fields(valid_src_tgt, checkpoint, lang=src_lang + '-' + tgt_lang)
    fields_valid_tgt_src = load_fields(valid_tgt_src, checkpoint, lang=tgt_lang + '-' + src_lang)
    fields = [fields_src, fields_tgt]
    fields_valid = [fields_valid_src_tgt, fields_valid_tgt_src]

    # TODO (JD): compare train_src.fields with fields_src, etc.
    print "train_src, src:", train_src.fields['src'].vocab.itos[10]
    print "train_src, tgt:", train_src.fields['tgt'].vocab.itos[10]
    print "train_tgt, src:", train_tgt.fields['src'].vocab.itos[10]
    print "train_tgt, tgt:", train_tgt.fields['tgt'].vocab.itos[10]
    print "valid_src_tgt, src:", valid_src_tgt.fields['src'].vocab.itos[10]
    print "valid_src_tgt, tgt:", valid_src_tgt.fields['tgt'].vocab.itos[10]
    print "valid_tgt_src, src:", valid_tgt_src.fields['src'].vocab.itos[10]
    print "valid_tgt_src, tgt:", valid_tgt_src.fields['tgt'].vocab.itos[10]

    '''
    src_vectors = numpy.zeros((len(fields_src['src'].vocab.itos), opt.src_word_vec_size))
    src_vector_f = open('/data/sls/scratch/jdrexler/MUSE/data/wiki.' + src_lang + '.vec', 'r')
    first = True
    j = 0
    for line in src_vector_f:
        if first:
            first = False
            continue
        j += 1
        f = line.strip().split()
        if f[0] in fields_src['src'].vocab.stoi:
            sys.stdout.flush()
            src_vectors[fields_src['src'].vocab.stoi[f[0]], :] = [float(v) for v in f[1:]]
        else:
            print j, f[0]

    tgt_vectors = numpy.zeros((len(fields_tgt['src'].vocab.itos), opt.tgt_word_vec_size))
    tgt_vector_f = open('/data/sls/scratch/jdrexler/MUSE/data/wiki.' + tgt_lang + '.vec', 'r')
    first = True
    j = 0
    for line in tgt_vector_f:
        if first:
            first = False
            continue
        j += 1
        f = line.strip().split()
        if f[0] in fields_tgt['src'].vocab.stoi:
            sys.stdout.flush()
            tgt_vectors[fields_tgt['src'].vocab.stoi[f[0]], :] = [float(v) for v in f[1:]]
        else:
            print j, f[0]
    src_tensor = torch.FloatTensor(src_vectors)
    tgt_tensor = torch.FloatTensor(tgt_vectors)
    print src_tensor.size(), tgt_tensor.size()
    torch.save(src_tensor, src_lang + '_embeddings.pt')
    torch.save(tgt_tensor, tgt_lang + '_embeddings.pt')
    '''
    
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

    #models[0].encoder.embeddings.load_pretrained_vectors(src_lang + '_embeddings.pt', False)
    #models[1].encoder.embeddings.load_pretrained_vectors(tgt_lang + '_embeddings.pt', False)

    src_tgt_model = onmt.Models.NMTModel(models[0].encoder, models[1].decoder)
    src_tgt_model.generator = models[1].generator
    if use_gpu(opt):
        src_tgt_model.cuda()

    tgt_src_model = onmt.Models.NMTModel(models[1].encoder, models[0].decoder)
    tgt_src_model.generator = models[0].generator
    if use_gpu(opt):
        tgt_src_model.cuda()

    cd_models = [src_tgt_model, tgt_src_model]
    
    discrim_models = onmt.ModelConstructor.make_discriminators(opt, [m.encoder for m in models], use_gpu(opt))
    
    for model in models:
         tally_parameters(model)
    check_save_model_path()
    
    # Build optimizer.
    optims = []
    for model in models:
        optims.append(build_optim(model, checkpoint, "adam"))

    cd_optims = []
    for model in cd_models:
        cd_optims.append(build_optim(model, checkpoint, "adam"))

    discrim_optims = []
    advers_optims = []
    for model in discrim_models:
        discrim_optims.append(build_optim(model.classifier, checkpoint, "rmsprop"))
        advers_optims.append(build_optim(model.encoder, checkpoint, "rmsprop"))
    
    # Do training.
    labels = [0.1, 0.9, 0.1, 0.9]
    train_model(models, cd_models, train, valid, fields, fields_valid, optims,
                cd_optims, discrim_models, discrim_optims, labels, advers_optims)

if __name__ == "__main__":
    main()
