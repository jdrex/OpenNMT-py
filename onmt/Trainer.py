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
    def __init__(self, loss=0, n_words=0, n_correct=0):
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
        return 100 * (self.n_correct / self.n_words)

    def ppl(self):
        return math.exp(min(self.loss / self.n_words, 100))

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

            src = onmt.IO.make_features(batch, 'src')
            tgt_outer = onmt.IO.make_features(batch, 'tgt')
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
            src = onmt.IO.make_features(batch, 'src')
            tgt = onmt.IO.make_features(batch, 'tgt')

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

        src = onmt.IO.make_features(batch, 'src')
        tgt_outer = onmt.IO.make_features(batch, 'tgt')

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
        weights = torch.cuda.FloatTensor(src.size()[0], src.size()[1]).zero_()
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
            src = onmt.IO.make_features(batch, 'src')
            tgt = onmt.IO.make_features(batch, 'tgt')

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
