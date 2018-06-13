from __future__ import division
"""
This is the loadable seq2seq trainer library that is
in charge of training details, loss compute, and statistics.
See train.py for a use case of this library.

Note!!! To make this a general library, we implement *only*
mechanism things here(i.e. what to do), and leave the strategy
things to users(i.e. how to do it). Also see train.py(one of the
users of this library) for the strategy things we do.
"""
import time
import sys
import math
import torch
import torch.nn as nn
from torch.autograd import Variable

import onmt
import onmt.io
import onmt.modules
import numpy as np

class Statistics(object):
    """
    Accumulator for loss statistics.
    Currently calculates:

    * accuracy
    * perplexity
    * elapsed time
    """
    def __init__(self, loss=0., n_words=0., n_correct=0.):
        self.loss = loss
        self.n_words = n_words
        self.n_correct = n_correct
        self.n_src_words = 0
        self.start_time = time.time()

    def update(self, stat):
        self.loss += stat.loss
        self.n_words += stat.n_words
        self.n_correct += stat.n_correct

    def update_loss(self, loss):
        self.loss += loss

    def accuracy(self):
        return 100 * (self.n_correct.type_as(self.loss) / self.n_words.type_as(self.loss))

    def ppl(self):
        return math.exp(min(self.loss / self.n_words.type_as(self.loss), 100.))

    def elapsed_time(self):
        return time.time() - self.start_time

    def output(self, epoch, batch, n_batches, start):
        """Write out statistics to stdout.

        Args:
           epoch (int): current epoch
           batch (int): current batch
           n_batch (int): total batches
           start (int): start time of epoch.
        """
        t = self.elapsed_time()
        print(("Epoch %2d, %5d/%5d; acc: %6.2f; ppl: %6.2f; " +
               "%3.0f src tok/s; %3.0f tgt tok/s; %6.0f s elapsed") %
              (epoch, batch,  n_batches,
               self.accuracy(),
               self.ppl(),
               self.n_src_words / (t + 1e-5),
               self.n_words / (t + 1e-5),
               time.time() - start))
        sys.stdout.flush()

    def output_loss(self, epoch, batch, n_batches, start):
        t = self.elapsed_time()
        print(("Epoch %2d, %5d/%5d; loss: %6.2f; " +
               "%3.0f src tok/s; %3.0f tgt tok/s; %6.0f s elapsed") %
              (epoch, batch,  n_batches,
               self.loss,
               self.n_src_words / (t + 1e-5),
               self.n_words / (t + 1e-5),
               time.time() - start))
        sys.stdout.flush()

    def log(self, prefix, experiment, lr):
        t = self.elapsed_time()
        experiment.add_scalar_value(prefix + "_ppl", self.ppl())
        experiment.add_scalar_value(prefix + "_accuracy", self.accuracy())
        experiment.add_scalar_value(prefix + "_tgtper",  self.n_words / t)
        experiment.add_scalar_value(prefix + "_lr", lr)
                                   
class Trainer(object):
    """
    Class that controls the training process.

    Args:
            model(:py:class:`onmt.Model.NMTModel`): translation model to train
            train_iter: training data iterator
            valid_iter: validate data iterator
            train_loss(:obj:`onmt.Loss.LossComputeBase`):
               training loss computation
            valid_loss(:obj:`onmt.Loss.LossComputeBase`):
               training loss computation
            optim(:obj:`onmt.Optim.Optim`):
               the optimizer responsible for update
            trunc_size(int): length of truncated back propagation through time
            shard_size(int): compute loss in shards of this size for efficiency
            data_type(string): type of the source input: [text|img|audio]
    """

    def __init__(self, model, train_iter, valid_iter,
                 train_loss, valid_loss, optim,
                 trunc_size=0, shard_size=32, data_type='text'):
        # Basic attributes.
        self.model = model
        self.train_iter = train_iter
        self.valid_iter = valid_iter
        self.train_loss = train_loss
        self.valid_loss = valid_loss
        self.optim = optim
        self.trunc_size = trunc_size
        self.shard_size = shard_size
        self.data_type = data_type

        # Set model in training mode.
        self.model.train()

    def train(self, epoch, report_func=None):
        """ Train next epoch.
        Args:
            epoch(int): the epoch number
            report_func(fn): function for logging

        Returns:
            stats (:obj:`onmt.Statistics`): epoch loss statistics
        """
        total_stats = Statistics()
        report_stats = Statistics()

        for i, batch in enumerate(self.train_iter):            
            target_size = batch.tgt.size(0)
            # Truncated BPTT
            trunc_size = self.trunc_size if self.trunc_size else target_size

            dec_state = None
            src = onmt.io.make_features(batch, 'src', self.data_type)
            if self.data_type == 'text':
                _, src_lengths = batch.src
                report_stats.n_src_words += src_lengths.sum()
            else:
                src_lengths = None

            tgt_outer = onmt.io.make_features(batch, 'tgt')

            for j in range(0, target_size-1, trunc_size):
                # 1. Create truncated target.
                tgt = tgt_outer[j: j + trunc_size]

                # 2. F-prop all but generator.
                self.model.zero_grad()
                outputs, attns, dec_state = \
                    self.model(src, tgt, src_lengths, dec_state)

                # 3. Compute loss in shards for memory efficiency.
                batch_stats = self.train_loss.sharded_compute_loss(
                        batch, outputs, attns, j,
                        trunc_size, self.shard_size)

                # 4. Update the parameters and statistics.
                self.optim.step()
                total_stats.update(batch_stats)
                report_stats.update(batch_stats)

                # If truncated, don't backprop fully.
                if dec_state is not None:
                    dec_state.detach()

            if report_func is not None:
                report_stats = report_func(
                        epoch, i, len(self.train_iter),
                        total_stats.start_time, self.optim.lr, report_stats)

        return total_stats
    
    def add_noise(self, src_lengths, src, tgt_outer):
        s_l_n = src_lengths.cpu().numpy()
        s_n = src.data.cpu().numpy()
        t_n = tgt_outer.data.cpu().numpy()
        
        for s in range(len(s_l_n)):
            # remove word with low probability
            for w in range(s_l_n[s]-2, 0, -1):
                if np.random.uniform() < 0.: #0.1: # JD make this a param
                    s_l_n[s] -= 1
                    s_n[w:-1, s] = s_n[w+1:, s]
            # shuffle
            indexes = np.arange(1, s_l_n[s]-1, 1)
            for k in range(0): # JD make this a param
                if k % 2 == 0:
                    loop = range(0, s_l_n[s]-1, 2)
                else:
                    loop = range(1, s_l_n[s]-1, 2)
                for j in loop:
                    np.random.shuffle(indexes[j:j+2])

            indexes = np.concatenate(([0], indexes, [s_l_n[s]-1]))
            s_n[:s_l_n[s], s] = s_n[indexes, s]

        sort_order = np.flip(np.argsort(s_l_n), 0)
        print sort_order
        
        sorted_lens = s_l_n[sort_order]
        sorted_src = s_n[:, sort_order]
        sorted_tgt = t_n[:, sort_order]
        
        print s_l_n.shape, src_lengths.size(), sorted_lens.shape
        print s_n.shape, src.size(), sorted_src.shape
        print t_n.shape, tgt_outer.size(), sorted_tgt.shape
        
        src_lengths = torch.cuda.LongTensor(sorted_lens)
        src = Variable(torch.cuda.LongTensor(sorted_src))
        tgt_outer = Variable(torch.cuda.LongTensor(sorted_tgt))

        src_lengths.cuda()
        src.cuda()
        tgt_outer.cuda()

        return src_lengths, src, tgt_outer

    def validate(self):
        """ Validate model.

        Returns:
            :obj:`onmt.Statistics`: validation loss statistics
        """
        # Set model in validating mode.
        self.model.eval()

        stats = Statistics()

        for batch in self.valid_iter:
            src = onmt.io.make_features(batch, 'src', self.data_type)
            if self.data_type == 'text':
                _, src_lengths = batch.src
            else:
                src_lengths = None

            tgt = onmt.io.make_features(batch, 'tgt')

            # F-prop through the model.
            outputs, attns, _ = self.model(src, tgt, src_lengths)

            # outputs = length x batch x 512
            
            # Compute loss.
            batch_stats = self.valid_loss.monolithic_compute_loss(
                    batch, outputs, attns)

            # Update statistics.
            stats.update(batch_stats)

        # Set model back to training mode.
        self.model.train()

        return stats

    def epoch_step(self, ppl, epoch):
        return self.optim.update_learning_rate(ppl, epoch)

    def drop_checkpoint(self, opt, epoch, fields, valid_stats):
        """ Save a resumable checkpoint.

        Args:
            opt (dict): option object
            epoch (int): epoch number
            fields (dict): fields and vocabulary
            valid_stats : statistics of last validation run
        """
        real_model = (self.model.module
                      if isinstance(self.model, nn.DataParallel)
                      else self.model)
        real_generator = (real_model.generator.module
                          if isinstance(real_model.generator, nn.DataParallel)
                          else real_model.generator)

        model_state_dict = real_model.state_dict()
        model_state_dict = {k: v for k, v in model_state_dict.items()
                            if 'generator' not in k}
        generator_state_dict = real_generator.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'generator': generator_state_dict,
            'vocab': onmt.io.save_fields_to_vocab(fields),
            'opt': opt,
            'epoch': epoch,
            'optim': self.optim,
        }
        torch.save(checkpoint,
                   '%s_acc_%.2f_ppl_%.2f_e%d.pt'
                   % (opt.save_model, valid_stats.accuracy(),
                      valid_stats.ppl(), epoch))

        return '%s_acc_%.2f_ppl_%.2f_e%d.pt' % (opt.save_model, valid_stats.accuracy(),
                      valid_stats.ppl(), epoch)

class AudioTextTrainer(Trainer):
    """
    Class that controls the training process.

    Args:
            model(:py:class:`onmt.Model.NMTModel`): translation model to train
            train_iter: training data iterator
            valid_iter: validate data iterator
            train_loss(:obj:`onmt.Loss.LossComputeBase`):
               training loss computation
            valid_loss(:obj:`onmt.Loss.LossComputeBase`):
               training loss computation
            optim(:obj:`onmt.Optim.Optim`):
               the optimizer responsible for update
            trunc_size(int): length of truncated back propagation through time
            shard_size(int): compute loss in shards of this size for efficiency
            data_type(string): type of the source input: [text|img|audio]
    """

    def __init__(self, model, train_iter, valid_iter,
                 text_model, text_train_iter, text_valid_iter,
                 train_loss, valid_loss, optim,
                 trunc_size=0, shard_size=32, data_type='text', mult=1):
        # Basic attributes.
        self.model = model
        self.train_iter = train_iter
        self.valid_iter = valid_iter

        self.text_model = text_model
        self.text_train_iter = text_train_iter
        self.text_valid_iter = text_valid_iter
        
        self.train_loss = train_loss
        self.valid_loss = valid_loss
        self.optim = optim
        self.trunc_size = trunc_size
        self.shard_size = shard_size
        self.data_type = data_type

        self.mult = mult
        # Set model in training mode.
        self.model.train()

    def train(self, epoch, report_func=None):
        """ Train next epoch.
        Args:
            epoch(int): the epoch number
            report_func(fn): function for logging

        Returns:
            stats (:obj:`onmt.Statistics`): epoch loss statistics
        """
        total_stats = Statistics()
        report_stats = Statistics()
        text_total_stats = Statistics()

        asr_batches = [s for s in self.train_iter]
        text_batches = [t for t in self.text_train_iter]
        
        nBatches = len(asr_batches)
        multiplier = self.mult #int(len(text_batches)/nBatches)

        #print "nBatches:", nBatches
        #print "mult:", multiplier
        
        for i in range(nBatches):
            #print i
            self.process_batch(asr_batches[i], self.model, "audio", report_stats, total_stats)
            for j in range(multiplier):
                #print j
                try:
                    self.process_batch(text_batches[i*multiplier+j], self.text_model, "text", report_stats, text_total_stats)
                except:
                    print "out of range:", len(text_batches), i*multiplier+j
                
            if report_func is not None:
                report_stats = report_func(
                    epoch, i, nBatches,
                    total_stats.start_time, self.optim.lr, report_stats)

        return total_stats, text_total_stats

    def process_batch(self, batch, model, data_type, report_stats, total_stats):
        target_size = batch.tgt.size(0)
        # Truncated BPTT
        trunc_size = self.trunc_size if self.trunc_size else target_size

        dec_state = None
        src = onmt.io.make_features(batch, 'src', data_type)
        if data_type == 'text':
            _, src_lengths = batch.src
            report_stats.n_src_words += src_lengths.sum()
        else:
            src_lengths = None

        tgt_outer = onmt.io.make_features(batch, 'tgt')

        for j in range(0, target_size-1, trunc_size):
            # 1. Create truncated target.
            tgt = tgt_outer[j: j + trunc_size]

            # 2. F-prop all but generator.
            model.zero_grad()
            outputs, attns, dec_state = model(src, tgt, src_lengths, dec_state)

            # 3. Compute loss in shards for memory efficiency.
            batch_stats = self.train_loss.sharded_compute_loss(
                batch, outputs, attns, j,
                trunc_size, self.shard_size)

            # 4. Update the parameters and statistics.
            self.optim.step()
            total_stats.update(batch_stats)
            report_stats.update(batch_stats)

            # If truncated, don't backprop fully.
            if dec_state is not None:
                dec_state.detach()

        model.zero_grad()
        return report_stats, total_stats

    def validate(self):
        """ Validate model.

        Returns:
            :obj:`onmt.Statistics`: validation loss statistics
        """
        # Set model in validating mode.
        self.model.eval()

        stats = Statistics()

        for batch in self.valid_iter:
            src = onmt.io.make_features(batch, 'src', self.data_type)
            if self.data_type == 'text':
                _, src_lengths = batch.src
            else:
                src_lengths = None

            tgt = onmt.io.make_features(batch, 'tgt')

            # F-prop through the model.
            outputs, attns, _ = self.model(src, tgt, src_lengths)

            # Compute loss.
            batch_stats = self.valid_loss.monolithic_compute_loss(
                    batch, outputs, attns)

            # Update statistics.
            stats.update(batch_stats)

        # Set model back to training mode.
        self.model.train()

        return stats

    def drop_checkpoint(self, opt, epoch, fields, valid_stats):
        """ Save a resumable checkpoint.

        Args:
            opt (dict): option object
            epoch (int): epoch number
            fields (dict): fields and vocabulary
            valid_stats : statistics of last validation run
        """
        real_model = (self.model.module
                      if isinstance(self.model, nn.DataParallel)
                      else self.model)
        real_text_model = (self.text_model.module
                      if isinstance(self.text_model, nn.DataParallel)
                      else self.text_model)
        real_generator = (real_model.generator.module
                          if isinstance(real_model.generator, nn.DataParallel)
                          else real_model.generator)

        model_state_dict = real_model.state_dict()
        model_state_dict = {k: v for k, v in model_state_dict.items()
                            if 'generator' not in k}
        text_model_state_dict = real_text_model.state_dict()
        #text_model_state_dict = {k: v for k, v in text_model_state_dict.items()
        #                    if 'generator' not in k}
        generator_state_dict = real_generator.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'generator': generator_state_dict,
            'vocab': onmt.io.save_fields_to_vocab(fields),
            'opt': opt,
            'epoch': epoch,
            'optim': self.optim,
            'text_model': text_model_state_dict,
        }
        torch.save(checkpoint,
                   '%s_acc_%.2f_ppl_%.2f_e%d.pt'
                   % (opt.save_model, valid_stats.accuracy(),
                      valid_stats.ppl(), epoch))

        return '%s_acc_%.2f_ppl_%.2f_e%d.pt' % (opt.save_model, valid_stats.accuracy(),
                      valid_stats.ppl(), epoch)

class AudioTextTrainerAdv(AudioTextTrainer):
    """
    Class that controls the training process.

    Args:
            model(:py:class:`onmt.Model.NMTModel`): translation model to train
            train_iter: training data iterator
            valid_iter: validate data iterator
            train_loss(:obj:`onmt.Loss.LossComputeBase`):
               training loss computation
            valid_loss(:obj:`onmt.Loss.LossComputeBase`):
               training loss computation
            optim(:obj:`onmt.Optim.Optim`):
               the optimizer responsible for update
            trunc_size(int): length of truncated back propagation through time
            shard_size(int): compute loss in shards of this size for efficiency
            data_type(string): type of the source input: [text|img|audio]
    """

    def __init__(self, model, train_iter, valid_iter,
                 text_model, text_train_iter, text_valid_iter,
                 train_loss, text_loss, valid_loss, text_valid_loss, optim, text_optim, 
                 discrim_models, labels, gen_optims, gen_lambda=1., 
                 trunc_size=0, shard_size=32, data_type='text', mult=1):
        # Basic attributes.
        self.model = model
        self.train_iter = train_iter
        self.valid_iter = valid_iter

        self.text_model = text_model
        self.text_train_iter = text_train_iter
        self.text_valid_iter = text_valid_iter
        
        self.train_loss = train_loss
        self.text_loss = text_loss
        self.valid_loss = valid_loss
        self.text_valid_loss = text_valid_loss
        self.optim = optim
        self.text_optim = text_optim

        self.discrim_models = discrim_models
        self.gen_optims = gen_optims
        self.labels = labels
        
        self.trunc_size = trunc_size
        self.shard_size = shard_size
        self.data_type = data_type

        self.criterion = nn.BCEWithLogitsLoss() #CrossEntropyLoss()

        # Set model in training mode.
        self.model.train()
        self.discrim_models[0].train()
        self.discrim_models[1].train()

        self.mult = mult
        self.gen_lambda = gen_lambda
        
    def train(self, epoch, report_func=None):
        """ Train next epoch.
        Args:
            epoch(int): the epoch number
            report_func(fn): function for logging

        Returns:
            stats (:obj:`onmt.Statistics`): epoch loss statistics
        """
        total_stats = Statistics()
        report_stats = Statistics()
        text_total_stats = Statistics()

        asr_batches = [s for s in self.train_iter]
        text_batches = [t for t in self.text_train_iter]
        
        nBatches = len(asr_batches)
        multiplier = self.mult #int(len(text_batches)/nBatches)

        #print "nBatches:", nBatches
        #print "mult:", multiplier
        
        for i in range(nBatches):
            #print i
            self.process_batch(asr_batches[i], "audio", report_stats, total_stats)
            for j in range(multiplier):
                #print j
                try:
                    self.process_batch(text_batches[i*multiplier+j], "text", report_stats, text_total_stats)
                except:
                    print len(text_batches), i*multiplier+j
                
            if report_func is not None:
                report_stats = report_func(
                    epoch, i, nBatches,
                    total_stats.start_time, self.optim.lr, report_stats)

        return total_stats, text_total_stats

    def process_batch(self, batch, data_type, report_stats, total_stats):
        target_size = batch.tgt.size(0)
        # Truncated BPTT
        trunc_size = self.trunc_size if self.trunc_size else target_size

        dec_state = None
        src = onmt.io.make_features(batch, 'src', data_type)
        if data_type == 'text':
            _, src_lengths = batch.src
            report_stats.n_src_words += src_lengths.sum()
            loss = self.text_loss
            optim = self.text_optim
            model = self.text_model
            discrim_model = self.discrim_models[1]
            label = self.labels[1]
            gen_optim = self.gen_optims[1]
        else:
            src_lengths = None
            loss = self.train_loss
            optim = self.optim
            model = self.model
            discrim_model = self.discrim_models[0]
            label = self.labels[0]
            gen_optim = self.gen_optims[0]
            
        tgt_outer = onmt.io.make_features(batch, 'tgt')

        for j in range(0, target_size-1, trunc_size):
            # 1. Create truncated target.
            tgt = tgt_outer[j: j + trunc_size]

            # 2. F-prop all but generator.
            model.zero_grad()
            outputs, attns, dec_state = model(src, tgt, src_lengths, dec_state)

            # 3. Compute loss in shards for memory efficiency.
            batch_stats = loss.sharded_compute_loss(
                batch, outputs, attns, j,
                trunc_size, self.shard_size)

            # 4. Update the parameters and statistics.
            optim.step()
            total_stats.update(batch_stats)
            report_stats.update(batch_stats)

            # If truncated, don't backprop fully.
            if dec_state is not None:
                dec_state.detach()

        model.zero_grad()
        discrim_model.zero_grad()
        outputs = discrim_model(src, src_lengths)
        l = [label]*outputs.size()[0]
        labels = Variable(torch.cuda.FloatTensor(l).view(-1,1))
        if src_lengths is not None:
            weights = torch.cuda.FloatTensor(src.size()[0], src.size()[1]).zero_()
            for j in range(len(src_lengths)):
                weights[:src_lengths[j], j] = self.gen_lambda
        else:
            src_labels = src.squeeze().sum(1)[:, 0:-1:8].data.cpu().numpy()
            w = np.zeros(src_labels.shape)
            w[src_labels != 0.] = self.gen_lambda
            weights = torch.cuda.FloatTensor(w)

        self.criterion.weight = weights.view(-1,1)[:outputs.size()[0], :]

        loss = self.criterion(outputs, labels)
        loss.backward()
        #total_stats.update_loss(loss.data[0])
        #report_stats.update_loss(loss.data[0])

        # 4. Update the parameters and statistics.
        gen_optim.step()

        model.zero_grad()
        discrim_model.zero_grad()
        
        return report_stats, total_stats

    def validate(self):
        """ Validate model.

        Returns:
            :obj:`onmt.Statistics`: validation loss statistics
        """
        # Set model in validating mode.
        self.model.eval()

        stats = Statistics()

        for batch in self.valid_iter:
            src = onmt.io.make_features(batch, 'src', self.data_type)
            src_lengths = None

            tgt = onmt.io.make_features(batch, 'tgt')

            # F-prop through the model.
            outputs, attns, _ = self.model(src, tgt, src_lengths)

            # Compute loss.
            batch_stats = self.valid_loss.monolithic_compute_loss(
                    batch, outputs, attns)

            # Update statistics.
            stats.update(batch_stats)

        # Set model back to training mode.
        self.model.train()

        return stats

    def validate_text(self):
        """ Validate text_model.

        Returns:
            :obj:`onmt.Statistics`: validation loss statistics
        """
        # Set text_model in validating mode.
        self.text_model.eval()

        stats = Statistics()

        for batch in self.text_valid_iter:
            src = onmt.io.make_features(batch, 'src')
            _, src_lengths = batch.src

            tgt = onmt.io.make_features(batch, 'tgt')

            # F-prop through the text_model.
            outputs, attns, _ = self.text_model(src, tgt, src_lengths)

            # Compute loss.
            batch_stats = self.text_valid_loss.monolithic_compute_loss(
                    batch, outputs, attns)

            # Update statistics.
            stats.update(batch_stats)

        # Set text_model back to training mode.
        self.text_model.train()

        return stats

    def drop_checkpoint(self, opt, epoch, fields, valid_stats):
        """ Save a resumable checkpoint.

        Args:
            opt (dict): option object
            epoch (int): epoch number
            fields (dict): fields and vocabulary
            valid_stats : statistics of last validation run
        """
        real_model = (self.model.module
                      if isinstance(self.model, nn.DataParallel)
                      else self.model)
        real_generator = (real_model.generator.module
                          if isinstance(real_model.generator, nn.DataParallel)
                          else real_model.generator)

        model_state_dict = real_model.state_dict()
        model_state_dict = {k: v for k, v in model_state_dict.items()
                            if 'generator' not in k}
        generator_state_dict = real_generator.state_dict()

        # text model
        real_text_model = (self.text_model.module
                      if isinstance(self.text_model, nn.DataParallel)
                      else self.text_model)
        text_model_state_dict = real_text_model.state_dict()
        text_model_state_dict = {k: v for k, v in text_model_state_dict.items()
                            if 'generator' not in k}

        checkpoint = {
            'model': model_state_dict,
            'text_model': text_model_state_dict,
            'generator': generator_state_dict,
            'vocab': onmt.io.save_fields_to_vocab(fields),
            'opt': opt,
            'epoch': epoch,
            'optim': self.optim,
            'text_optim': self.text_optim,
            'src_optim': self.gen_optims[0],
            'tgt_optim': self.gen_optims[1],
        }
        torch.save(checkpoint,
                   '%s_acc_%.2f_ppl_%.2f_e%d.pt'
                   % (opt.save_model, valid_stats.accuracy(),
                      valid_stats.ppl(), epoch))

        return '%s_acc_%.2f_ppl_%.2f_e%d.pt' % (opt.save_model, valid_stats.accuracy(),
                      valid_stats.ppl(), epoch)

class AudioTextSpeechTrainerAdv(AudioTextTrainer):
    """
    Class that controls the training process.

    Args:
            model(:py:class:`onmt.Model.NMTModel`): translation model to train
            train_iter: training data iterator
            valid_iter: validate data iterator
            train_loss(:obj:`onmt.Loss.LossComputeBase`):
               training loss computation
            valid_loss(:obj:`onmt.Loss.LossComputeBase`):
               training loss computation
            optim(:obj:`onmt.Optim.Optim`):
               the optimizer responsible for update
            trunc_size(int): length of truncated back propagation through time
            shard_size(int): compute loss in shards of this size for efficiency
            data_type(string): type of the source input: [text|img|audio]
    """

    def __init__(self, model, train_iter, valid_iter,
                 text_model, text_train_iter, text_valid_iter,
                 speech_model, speech_train_iter, 
                 train_loss, text_loss, valid_loss, text_valid_loss, optim,
                 discrim_models, labels, gen_lambda=1., speech_lambda=1.,  
                 trunc_size=0, shard_size=32, data_type='text',
                 mult = 1, tMult = 1, unsup=False, big_text=False):
        # Basic attributes.
        self.model = model
        self.train_iter = train_iter
        self.valid_iter = valid_iter

        self.text_model = text_model
        self.text_train_iter = text_train_iter
        self.text_valid_iter = text_valid_iter
        
        self.speech_model = speech_model
        self.speech_train_iter = speech_train_iter

        self.train_loss = train_loss
        self.text_loss = text_loss
        self.valid_loss = valid_loss
        self.text_valid_loss = text_valid_loss
        self.optim = optim

        self.discrim_models = discrim_models
        self.labels = labels
        
        self.trunc_size = trunc_size
        self.shard_size = shard_size
        self.data_type = data_type

        self.mult = mult
        self.tMult = tMult
        self.gen_lambda = gen_lambda
        self.speech_lambda = speech_lambda
        print "multipliers:", self.mult, self.tMult

        self.big_text = big_text
        self.criterion = nn.BCEWithLogitsLoss() #CrossEntropyLoss()
        self.speech_loss = nn.SmoothL1Loss()

        self.ff = False
        self.sup = not unsup
        if not self.sup:
            print "UNSUPERVISED!!"
        
        # Set model in training mode.
        self.model.train()
        if self.discrim_models[0] is not None:
            self.discrim_models[0].train()
            self.discrim_models[1].train()
        
    def train(self, epoch, report_func=None, batch_override=-1, text=None):
        """ Train next epoch.
        Args:
            epoch(int): the epoch number
            report_func(fn): function for logging

        Returns:
            stats (:obj:`onmt.Statistics`): epoch loss statistics
        """
        total_stats = Statistics()
        report_stats = Statistics()
        text_total_stats = Statistics()
        speech_total_stats = Statistics()

        asr_batches = [s for s in self.train_iter]
        
        nBatches = len(asr_batches)
        if batch_override > 0:
            nBatches = min(nBatches, batch_override)

        #supBatches = min(len(text_batches), len(speech_batches))
        multiplier = self.mult #int(supBatches/nBatches)
        tMultiplier = self.tMult
        speech_batches = [t for t in self.speech_train_iter][:nBatches*multiplier]
        if self.big_text:
            text_batches = [t for t in text]
        else:
            text_batches = [t for t in self.text_train_iter]#[:nBatches*multiplier*tMultiplier]
        
        print "nBatches:", nBatches
        #print "mult:", multiplier

        print "tot speech:", len(speech_batches)
        print "tot text:", len(text_batches)
        for i in range(nBatches):
            print "batch:", i
            if self.sup:
                self.process_batch(asr_batches[i], self.model, "audio", None, self.labels[0], report_stats, total_stats)
            for j in range(multiplier):
                print " speech:", i*multiplier + j
                try:
                    self.process_speech(speech_batches[i*multiplier + j], self.speech_model, self.discrim_models[0], self.labels[0], speech_total_stats)
                except:
                    print "speech out of range"
                for k in range(tMultiplier):
                    print " text:", i*multiplier*tMultiplier + j*tMultiplier + k
                    try:
                    #self.process_batch(text_batches[i*multiplier + j*tMultiplier + k], self.text_model, "text",  None, self.labels[1], report_stats, text_total_stats)
                        self.process_batch(text_batches[i*multiplier + j*tMultiplier + k], self.text_model, "text",  self.discrim_models[1], self.labels[1], report_stats, text_total_stats)
                    except:
                        print "text out of range"
                #try:
                #    self.process_speech(asr_batches[i], self.speech_model, self.discrim_models[0], self.labels[0], speech_total_stats)
                #except:
                #    print "WEIRD ERROR IN PROCESS SPEECH"
                
            if report_func is not None:
                report_stats = report_func(
                    epoch, i, nBatches,
                    total_stats.start_time, self.optim.lr, report_stats)

        return total_stats, text_total_stats, speech_total_stats

    def process_batch(self, batch, model, data_type, discrim_model, label, report_stats, total_stats):
        target_size = batch.tgt.size(0)
        # Truncated BPTT
        trunc_size = self.trunc_size if self.trunc_size else target_size

        dec_state = None
        src = onmt.io.make_features(batch, 'src', data_type)
        if data_type == 'text':
            _, src_lengths = batch.src
            report_stats.n_src_words += src_lengths.sum()
            loss = self.text_loss
            #print "sizes:", src.size(), batch.tgt.size()
        else:
            src_lengths = None
            loss = self.train_loss

        tgt_outer = onmt.io.make_features(batch, 'tgt')
        #print "tgt:", batch.tgt[:, 0]
        #print "features:", tgt_outer[:, 0]
        
        for j in range(0, target_size-1, trunc_size):
            # 1. Create truncated target.
            tgt = tgt_outer[j: j + trunc_size]

            # 2. F-prop all but generator.
            model.zero_grad()
            outputs, attns, dec_state = model(src, tgt, src_lengths, dec_state)

            # 3. Compute loss in shards for memory efficiency.
            batch_stats = loss.sharded_compute_loss(
                batch, outputs, attns, j,
                trunc_size, self.shard_size)

            # 4. Update the parameters and statistics.
            self.optim.step()
            total_stats.update(batch_stats)
            report_stats.update(batch_stats)

            # If truncated, don't backprop fully.
            if dec_state is not None:
                dec_state.detach()

        model.zero_grad()
        if discrim_model is None:
            return report_stats, total_stats

        discrim_model.zero_grad()
        outputs = discrim_model(src, src_lengths)
        l = [label]*outputs.size()[0]
        labels = Variable(torch.cuda.FloatTensor(l).view(-1,1))
        if src_lengths is not None:
            weights = torch.cuda.FloatTensor(src.size()[0], src.size()[1]).zero_()
            for j in range(len(src_lengths)):
                weights[:src_lengths[j], j] = self.gen_lambda
        else:
            src_labels = src.squeeze().sum(1)[:, 0:-1:8].data.cpu().numpy()
            w = np.zeros(src_labels.shape)
            w[src_labels != 0.] = self.gen_lambda
            weights = torch.cuda.FloatTensor(w)

        self.criterion.weight = weights.view(-1,1)[:outputs.size()[0], :]

        loss = self.criterion(outputs, labels)
        loss.backward()
        #total_stats.update_loss(loss.data[0])
        #report_stats.update_loss(loss.data[0])

        # 4. Update the parameters and statistics.
        self.optim.step()

        model.zero_grad()
        discrim_model.zero_grad()
        
        return report_stats, total_stats

    def process_speech(self, batch, model, discrim_model, label, total_stats):
        # Truncated BPTT

        dec_state = None
        src = onmt.io.make_features(batch, 'src', "audio")[:, :, :, :1400]
        src_lengths = None
        #print batch.src.size(), target_size

        if model is not None:
            tgt_outer = onmt.io.make_features(batch, 'src', "audio")[:, :, :, :1400]
            tgt_outer = tgt_outer.squeeze()
            try:
                tgt_outer = tgt_outer.transpose(0, 2).transpose(1, 2)
            except:
                tgt_outer = tgt_outer.view(1, tgt_outer.size(0), tgt_outer.size(1))
                tgt_outer = tgt_outer.transpose(0, 2).transpose(1, 2)

            target_size = tgt_outer.size(0)
            if self.ff:
                trunc_size = target_size
            else:
                trunc_size = 200
                
            for j in range(0, target_size-1, trunc_size):
                if j > 1400:
                    break
                #print target_size, tgt_outer.size(), j, j+trunc_size
                # 1. Create truncated target.
                tgt = tgt_outer[j: j + trunc_size]
                model.zero_grad()
                outputs, attns, dec_state = model(src, tgt, src_lengths, dec_state)

                src_labels = tgt.squeeze().sum(1).data.cpu().numpy()
                w = np.zeros(src_labels.shape)
                w[src_labels != 0.] = self.speech_lambda
                weights = torch.cuda.FloatTensor(w)
                self.speech_loss.weight = weights.view(-1,1)[:outputs.size()[0], :]

                loss = self.speech_loss(outputs, tgt[1:, :, :])
                loss.backward()
                total_stats.update_loss(loss.data[0])

                # 4. Update the parameters and statistics.
                self.optim.step()
    
                # If truncated, don't backprop fully.
                if dec_state is not None:
                    dec_state.detach()

                #print "starting adverserial"
                model.zero_grad()
                
        if discrim_model is None:
            return total_stats
        
        discrim_model.zero_grad()
        outputs = discrim_model(src, src_lengths)
        l = [label]*outputs.size()[0]
        labels = Variable(torch.cuda.FloatTensor(l).view(-1,1))
        if src_lengths is not None:
            weights = torch.cuda.FloatTensor(src.size()[0], src.size()[1]).zero_()
            for j in range(len(src_lengths)):
                weights[:src_lengths[j], j] = self.gen_lambda
        else:
            src_labels = src.squeeze().sum(1)[:, 0:-1:8].data.cpu().numpy()
            w = np.zeros(src_labels.shape)
            w[src_labels != 0.] = self.gen_lambda
            weights = torch.cuda.FloatTensor(w)

        self.criterion.weight = weights.view(-1,1)[:outputs.size()[0], :]

        loss = self.criterion(outputs, labels)
        loss.backward()

        # 4. Update the parameters and statistics.
        self.optim.step()

        if model:
            model.zero_grad()
        discrim_model.zero_grad()
        
        return total_stats

    def validate(self):
        """ Validate model.

        Returns:
            :obj:`onmt.Statistics`: validation loss statistics
        """
        # Set model in validating mode.
        self.model.eval()

        stats = Statistics()

        for batch in self.valid_iter:
            src = onmt.io.make_features(batch, 'src', self.data_type)
            if self.data_type == 'text':
                _, src_lengths = batch.src
            else:
                src_lengths = None

            tgt = onmt.io.make_features(batch, 'tgt')

            # F-prop through the model.
            outputs, attns, _ = self.model(src, tgt, src_lengths)

            # Compute loss.
            batch_stats = self.valid_loss.monolithic_compute_loss(
                    batch, outputs, attns)

            # Update statistics.
            stats.update(batch_stats)

        # Set model back to training mode.
        self.model.train()

        return stats

    def validate_text(self):
        """ Validate text_model.

        Returns:
            :obj:`onmt.Statistics`: validation loss statistics
        """
        # Set text_model in validating mode.
        self.text_model.eval()

        stats = Statistics()

        for batch in self.text_valid_iter:
            src = onmt.io.make_features(batch, 'src')
            _, src_lengths = batch.src

            tgt = onmt.io.make_features(batch, 'tgt')

            # F-prop through the text_model.
            outputs, attns, _ = self.text_model(src, tgt, src_lengths)

            # Compute loss.
            batch_stats = self.text_valid_loss.monolithic_compute_loss(
                    batch, outputs, attns)

            # Update statistics.
            stats.update(batch_stats)

        # Set text_model back to training mode.
        self.text_model.train()

        return stats

    def drop_checkpoint(self, opt, epoch, fields, valid_stats):
        """ Save a resumable checkpoint.

        Args:
            opt (dict): option object
            epoch (int): epoch number
            fields (dict): fields and vocabulary
            valid_stats : statistics of last validation run
        """
        real_model = (self.model.module
                      if isinstance(self.model, nn.DataParallel)
                      else self.model)
        real_generator = (real_model.generator.module
                          if isinstance(real_model.generator, nn.DataParallel)
                          else real_model.generator)

        model_state_dict = real_model.state_dict()
        model_state_dict = {k: v for k, v in model_state_dict.items()
                            if 'generator' not in k}
        generator_state_dict = real_generator.state_dict()

        # text model
        real_text_model = (self.text_model.module
                      if isinstance(self.text_model, nn.DataParallel)
                      else self.text_model)
        text_model_state_dict = real_text_model.state_dict()
        text_model_state_dict = {k: v for k, v in text_model_state_dict.items()
                            if 'generator' not in k}
        # speech model
        if self.speech_model:
            real_speech_model = (self.speech_model.module
                                 if isinstance(self.speech_model, nn.DataParallel)
                                 else self.speech_model)
            speech_model_state_dict = real_speech_model.state_dict()
            speech_model_state_dict = {k: v for k, v in speech_model_state_dict.items()
                                       if 'generator' not in k}
        else:
            speech_model_state_dict = None

        checkpoint = {
            'model': model_state_dict,
            'text_model': text_model_state_dict,
            'speech_model': speech_model_state_dict,
            'generator': generator_state_dict,
            'vocab': onmt.io.save_fields_to_vocab(fields),
            'opt': opt,
            'epoch': epoch,
            'optim': self.optim,
        }
        torch.save(checkpoint,
                   '%s_acc_%.2f_ppl_%.2f_e%d.pt'
                   % (opt.save_model, valid_stats.accuracy(),
                      valid_stats.ppl(), epoch))

        return '%s_acc_%.2f_ppl_%.2f_e%d.pt' % (opt.save_model, valid_stats.accuracy(),
                      valid_stats.ppl(), epoch)

class AdvTrainer(Trainer):
    def __init__(self, model, discrim_model, train_iter, valid_iter,
                 train_loss, valid_loss, optim, label, 
                 trunc_size, shard_size):
        """
        Args:
            model: the seq2seq model.
            train_iter: the train data iterator.
            valid_iter: the validate data iterator.
            train_loss: the train side LossCompute object for computing loss.
            valid_loss: the valid side LossCompute object for computing loss.
            optim: the optimizer responsible for lr update.
            trunc_size: a batch is divided by several truncs of this size.
            shard_size: compute loss in shards of this size for efficiency.
        """
        # Basic attributes.
        self.model = model
        self.discrim_model = discrim_model
        self.train_iter = train_iter
        self.valid_iter = valid_iter
        self.train_loss = train_loss
        self.valid_loss = valid_loss
        self.optim = optim
        self.label = label
        self.trunc_size = trunc_size
        self.shard_size = shard_size
        self.criterion = nn.BCEWithLogitsLoss() #CrossEntropyLoss()

        # Set model in training mode.
        self.model.train()

    def train(self, epoch, report_func=None):
        """ Called for each epoch to train. """
        total_stats = Statistics()
        report_stats = Statistics()

        for i, batch in enumerate(self.train_iter):
            target_size = batch.tgt.size(0)
            # Truncated BPTT
            trunc_size = self.trunc_size if self.trunc_size else target_size

            dec_state = None
            _, src_lengths = batch.src

            src = onmt.io.make_features(batch, 'src')
            tgt_outer = onmt.io.make_features(batch, 'tgt')
            report_stats.n_src_words += src_lengths.sum()

            for j in range(0, target_size-1, trunc_size):
                # 1. Create truncated target.
                tgt = tgt_outer[j: j + trunc_size]

                # 2. F-prop all but generator.
                self.model.zero_grad()
                outputs, attns, dec_state = \
                    self.model(src, tgt, src_lengths, dec_state)

                # 3. Compute loss in shards for memory efficiency.
                batch_stats = self.train_loss.sharded_compute_loss(
                        batch, outputs, attns, j,
                        trunc_size, self.shard_size)

                # 4. Update the parameters and statistics.
                self.optim.step()
                total_stats.update(batch_stats)
                report_stats.update(batch_stats)

                # If truncated, don't backprop fully.
                if dec_state is not None:
                    dec_state.detach()

            self.model.zero_grad()
            outputs = self.discrim_model(src, src_lengths)

            # loss re: true_label, backprop through discrim
            fake_l = [1-self.label]*outputs.size()[0]
            labels = Variable(torch.cuda.FloatTensor(fake_l).view(-1,1)) #Long for CELoss
            if src_lengths is not None:
                weights = torch.cuda.FloatTensor(src.size()[0], src.size()[1]).zero_()
                for j in range(len(src_lengths)):
                    weights[:src_lengths[j], j] = self.gen_lambda
            else:
                src_labels = src.squeeze().sum(1)[:, 0:-1:8].data.cpu().numpy()
                w = np.zeros(src_labels.shape)
                w[src_labels != 0.] = self.gen_lambda
                weights = torch.cuda.FloatTensor(w)
            
            self.criterion.weight = weights.view(-1,1)[:outputs.size()[0], :]

            loss = self.criterion(outputs, labels)
            loss.backward()

            self.optim.step()
            outputs_2 = self.discrim_model(src, src_lengths)
            
            if i % 10 == 0:
                print "adverserial:", i, self.label
                print outputs.data[0:5]
                print outputs_2.data[0:5]

            if report_func is not None:
                report_stats = report_func(
                        epoch, i, len(self.train_iter),
                        total_stats.start_time, self.optim.lr, report_stats)

        return total_stats

    def validate(self):
        """ Called for each epoch to validate. """
        # Set model in validating mode.
        self.model.eval()

        stats = Statistics()

        for batch in self.valid_iter:
            _, src_lengths = batch.src
            src = onmt.io.make_features(batch, 'src')
            tgt = onmt.io.make_features(batch, 'tgt')

            # F-prop through the model.
            outputs, attns, _ = self.model(src, tgt, src_lengths)

            # Compute loss.
            batch_stats = self.valid_loss.monolithic_compute_loss(
                    batch, outputs, attns)

            # Update statistics.
            stats.update(batch_stats)

        # Set model back to training mode.
        self.model.train()

        return stats

class UnsupTrainer(Trainer):
    def __init__(self, models, cd_models, discrim_models, train_iters, valid_iter,
                 train_losses, cd_losses, valid_loss, optims, cd_optims, labels, 
                 trunc_size, shard_size):
        """
        Args:
            model: the seq2seq model.
            train_iter: the train data iterator.
            valid_iter: the validate data iterator.
            train_loss: the train side LossCompute object for computing loss.
            valid_loss: the valid side LossCompute object for computing loss.
            optim: the optimizer responsible for lr update.
            trunc_size: a batch is divided by several truncs of this size.
            shard_size: compute loss in shards of this size for efficiency.
        """
        # Basic attributes.
        self.src_model = models[0]
        self.tgt_model = models[1]
        self.src_discrim_model = discrim_models[0]
        self.tgt_discrim_model = discrim_models[1]
        self.src_train_iter = train_iters[0]
        self.tgt_train_iter = train_iters[1]
        self.src_train_loss = train_losses[0]
        self.tgt_train_loss = train_losses[1]

        self.valid_iter = valid_iter
        self.valid_loss = valid_loss
        
        self.src_optim = optims[0]
        self.tgt_optim = optims[1]
        self.src_label = labels[0]
        self.tgt_label = labels[1]
        
        self.trunc_size = trunc_size
        self.shard_size = shard_size
        self.criterion = nn.BCEWithLogitsLoss(weight=torch.FloatTensor(1,1).zero_()) #CrossEntropyLoss()

        self.trainCD = False
        self.cd_iters = dict()
        self.src_tgt_model = cd_models[0]
        self.tgt_src_model = cd_models[1]
        self.src_tgt_train_loss = cd_losses[0]
        self.tgt_src_train_loss = cd_losses[1]
        self.src_tgt_optim = cd_optims[0]
        self.tgt_src_optim = cd_optims[1]
        
        # Set model in training mode.
        self.src_model.train()
        self.tgt_model.train()

    def train(self, epoch, report_func=None):
        """ Called for each epoch to train. """
        total_stats = Statistics()
        report_stats = Statistics()

        src_batches = [s for s in self.src_train_iter]
        tgt_batches = [t for t in self.tgt_train_iter]
        
        if self.trainCD:
            src_tgt_batches = [s for s in self.cd_iters["src-tgt"]]
            tgt_src_batches = [s for s in self.cd_iters["tgt-src"]]
            print len(src_tgt_batches), len(tgt_src_batches)
            nBatches = min(len(src_batches), len(tgt_batches), len(src_tgt_batches), len(tgt_src_batches))
        else:
            nBatches = min(len(src_batches), len(tgt_batches))

        print "nBatches:", nBatches
        for i in range(nBatches):
            print "batch", i
            print "processing src"
            self.process_batch(src_batches[i], self.src_model, self.src_optim, self.src_train_loss,
                            self.src_discrim_model, self.src_label, report_stats, total_stats)

            if self.tgt_optim is not None:
                print "processing tgt"
                self.process_batch(tgt_batches[i], self.tgt_model, self.tgt_optim, self.tgt_train_loss,
                          self.tgt_discrim_model, self.tgt_label, report_stats, total_stats)

            if self.trainCD:
                print "processing src_tgt"
                self.process_batch(src_tgt_batches[i], self.src_tgt_model, self.src_tgt_optim, self.src_tgt_train_loss,
                            self.src_discrim_model, self.src_label, report_stats, total_stats)
                print "processing tgt_src"
                self.process_batch(tgt_src_batches[i], self.tgt_src_model, self.tgt_src_optim, self.tgt_src_train_loss,
                            self.tgt_discrim_model, self.tgt_label, report_stats, total_stats)
                
            if report_func is not None:
                report_stats = report_func(
                        epoch, i, nBatches,
                        total_stats.start_time, self.src_optim.lr, report_stats)

        return total_stats

    def process_batch(self, batch, model, optim, train_loss, discrim_model, label, report_stats, total_stats):
        target_size = batch.tgt.size(0)
        
        # Truncated BPTT
        trunc_size = self.trunc_size if self.trunc_size else target_size
    
        dec_state = None
        _, src_lengths = batch.src

        src = onmt.io.make_features(batch, 'src')
        tgt_outer = onmt.io.make_features(batch, 'tgt')

        #src_lengths, src, tgt_outer = self.add_noise(src_lengths, src, tgt_outer)
        print "BATCH INPUT:", src.size()
        report_stats.n_src_words += src_lengths.sum()

        for j in range(0, target_size-1, trunc_size):
            # 1. Create truncated target.
            tgt = tgt_outer[j: j + trunc_size]
        
            # 2. F-prop all but generator.
            model.zero_grad()
            outputs, attns, dec_state = \
                model(src, tgt, src_lengths, dec_state)

            # 3. Compute loss in shards for memory efficiency.
            batch_stats = train_loss.sharded_compute_loss(
                batch, outputs, attns, j,
                trunc_size, self.shard_size)

            # 4. Update the parameters and statistics.
            optim.step()
            total_stats.update(batch_stats)
            report_stats.update(batch_stats)

            # If truncated, don't backprop fully.
            if dec_state is not None:
                dec_state.detach()

        model.zero_grad()
        outputs = discrim_model(src, src_lengths)
    
        # loss re: true_label, backprop through discrim
        fake_l = [1-label]*outputs.size()[0]
        labels = Variable(torch.cuda.FloatTensor(fake_l).view(-1,1)) #Long for CELoss
        weights = torch.FloatTensor(src.size()[0], src.size()[1]).zero_()
        for i in range(len(src_lengths)):
            weights[:src_lengths[i], i] = 1.
        self.criterion.weight = weights.view(-1,1)[:outputs.size()[0], :]
        
        loss = self.criterion(outputs, labels)
        loss.backward()

        optim.step()
        #outputs_2 = discrim_model(src, src_lengths)
            
        #print "adverserial:", label
        #print outputs.data[0:5]
        #print outputs_2.data[0:5]

    def add_noise(self, src_lengths, src, tgt_outer):
        s_l_n = src_lengths.cpu().numpy()
        s_n = src.data.cpu().numpy()
        #src_lengths = src_lengths.cpu()
        #src = src.cpu()
        #src_lengths = src_lengths.numpy()
        #src = src.numpy()
        
        for s in range(len(s_l_n)):
            # remove word with low probability
            for w in range(s_l_n[s]-2, 0, -1):
                if np.random.uniform() < 0.1: # JD make this a param
                    s_l_n[s] -= 1
                    s_n[w:-1, s] = s_n[w+1:, s]
            # shuffle
            indexes = np.arange(s_l_n[s])
            for k in range(3): # JD make this a param
                if k % 2 == 0:
                    loop = range(0, s_l_n[s]-1, 2)
                else:
                    loop = range(1, s_l_n[s]-1, 2)
                for j in loop:
                    np.random.shuffle(indexes[j:j+2])
            s_n[:s_l_n[s], s] = s_n[indexes, s]

        t_n = tgt_outer.data.cpu().numpy()
        
        sort_order = np.flip(np.argsort(s_l_n), 0)
        #print s_l_n.size(), sort_order.shape, sort_order
        s_l_n = s_l_n[sort_order]
        s_n = s_n[:, sort_order]
        t_n = t_n[:, sort_order]

        src_lengths = torch.cuda.LongTensor(s_l_n)
        src = Variable(torch.cuda.LongTensor(s_n))
        tgt_outer = Variable(torch.cuda.LongTensor(t_n))

        src_lengths.cuda()
        src.cuda()
        tgt_outer.cuda()

        return src_lengths, src, tgt_outer
            
    def validate(self):
        """ Called for each epoch to validate. """
        # Set model in validating mode.
        self.model.eval()

        stats = Statistics()

        for batch in self.valid_iter:
            _, src_lengths = batch.src
            src = onmt.io.make_features(batch, 'src')
            tgt = onmt.io.make_features(batch, 'tgt')

            # F-prop through the model.
            outputs, attns, _ = self.model(src, tgt, src_lengths)

            # Compute loss.
            batch_stats = self.valid_loss.monolithic_compute_loss(
                    batch, outputs, attns)

            # Update statistics.
            stats.update(batch_stats)

        # Set model back to training mode.
        self.model.train()

        return stats

    def drop_checkpoint(self, opt, epoch, fields, valid_stats):
        """ Called conditionally each epoch to save a snapshot. """
        real_model = (self.src_model.module
                      if isinstance(self.src_model, nn.DataParallel)
                      else self.src_model)
        real_generator = (real_model.generator.module
                          if isinstance(real_model.generator, nn.DataParallel)
                          else real_model.generator)

        model_state_dict = real_model.state_dict()
        model_state_dict = {k: v for k, v in model_state_dict.items()
                            if 'generator' not in k}
        generator_state_dict = real_generator.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'generator': generator_state_dict,
            'vocab': onmt.IO.save_vocab(fields[0]),
            'opt': opt,
            'epoch': epoch,
            'optim': self.src_optim
        }
        torch.save(checkpoint,
                   'src_%s_acc_%.2f_ppl_%.2f_e%d.pt'
                   % (opt.save_model, valid_stats.accuracy(),
                      valid_stats.ppl(), epoch))

        
        """ Called conditionally each epoch to save a snapshot. """
        real_model = (self.tgt_model.module
                      if isinstance(self.tgt_model, nn.DataParallel)
                      else self.tgt_model)
        real_generator = (real_model.generator.module
                          if isinstance(real_model.generator, nn.DataParallel)
                          else real_model.generator)

        model_state_dict = real_model.state_dict()
        model_state_dict = {k: v for k, v in model_state_dict.items()
                            if 'generator' not in k}
        generator_state_dict = real_generator.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'generator': generator_state_dict,
            'vocab': onmt.IO.save_vocab(fields[1]),
            'opt': opt,
            'epoch': epoch,
            'optim': self.tgt_optim
        }
        torch.save(checkpoint,
                   'tgt_%s_acc_%.2f_ppl_%.2f_e%d.pt'
                   % (opt.save_model, valid_stats.accuracy(),
                      valid_stats.ppl(), epoch))

        return '%s_acc_%.2f_ppl_%.2f_e%d.pt' % (opt.save_model, valid_stats.accuracy(),
                      valid_stats.ppl(), epoch)
