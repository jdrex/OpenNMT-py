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
import onmt.modules
from onmt.Trainer import Statistics
import numpy as np

class DiscrimTrainer(object):
    def __init__(self, models, train_iters, optims, labels, shard_size):
        """
        Args:
            model: the seq2seq model.
            train_iter: the train data iterator.
            optim: the optimizer responsible for lr update.
            trunc_size: a batch is divided by several truncs of this size.
            shard_size: compute loss in shards of this size for efficiency.
        """
        # Basic attributes.
        self.src_model = models[0]
        self.src_train_iter = train_iters[0]
        self.src_optim = optims[0]
        self.src_label = labels[0]

        if len(models) > 1:
            self.tgt_model = models[1]
            self.tgt_train_iter = train_iters[1]
            self.tgt_optim = optims[1]
            self.tgt_label = labels[1]
            self.tgt_model.train()

        self.shard_size = shard_size
        self.criterion = nn.BCEWithLogitsLoss() #CrossEntropyLoss()
        # Set model in training mode.
        self.src_model.train()

    def train(self, epoch, report_func=None):
        """ Called for each epoch to train. """
        total_stats = Statistics()
        report_stats = Statistics()

        src_batches = [s for s in self.src_train_iter]
        nBatches = len(src_batches)
        
        if hasattr(self, 'tgt_model'):
            tgt_batches = [t for t in self.tgt_train_iter]
    
            nBatches = min(len(src_batches), len(tgt_batches))
        
        for i in range(nBatches):
            # SRC
            batch = src_batches[i]
            
            src = onmt.io.make_features(batch, 'src', 'audio')
            src_labels = src.squeeze().sum(1)[:, 0:-1:8].data.cpu().numpy()
            #print src.size(), src_labels.shape
            
            self.src_model.zero_grad()
            outputs = self.src_model(src, None)
            l = [self.src_label]*outputs.size()[0]
            labels = Variable(torch.cuda.FloatTensor(l).view(-1,1))
            w = np.zeros(src_labels.shape)
            w[src_labels != 0.] = 1.
            weights = torch.cuda.FloatTensor(w)
            #print src_labels.shape, w.shape, weights.size()
            self.criterion.weight = weights.view(-1,1)[:outputs.size()[0], :]

            #print outputs.size(), labels.size()
            loss = self.criterion(outputs, labels)
            loss.backward()
            #if i % 10 == 0:
            #    print "discriminator", i, self.src_label
            #    print outputs.data[0:5], loss.data[0]
            
            total_stats.update_loss(loss.data[0])
            report_stats.update_loss(loss.data[0])

            # 4. Update the parameters and statistics.
            self.src_optim.step()

            if not hasattr(self, 'tgt_model'):
                continue

            # TGT
            if self.tgt_optim is None:
                continue
            
            batch = tgt_batches[i]
            _, src_lengths = batch.src

            src = onmt.io.make_features(batch, 'src')
            #src_lengths, src = self.add_noise(src_lengths, src)

            report_stats.n_src_words += src_lengths.sum()

            self.tgt_model.zero_grad()
            outputs = self.tgt_model(src, src_lengths)
            l = [self.tgt_label]*outputs.size()[0]
            labels = Variable(torch.cuda.FloatTensor(l).view(-1,1))
            weights = torch.cuda.FloatTensor(src.size()[0], src.size()[1]).zero_()
            for j in range(len(src_lengths)):
                weights[:src_lengths[j], j] = 1.
            self.criterion.weight = weights.view(-1,1)[:outputs.size()[0], :]

            #print outputs.size(), labels.size()
            loss = self.criterion(outputs, labels)
            loss.backward()
            #if i % 10 == 0:
            #    print "discriminator", i, self.tgt_label
            #    print outputs.data[0:5], loss.data[0]
            
            total_stats.update_loss(loss.data[0])
            report_stats.update_loss(loss.data[0])

            # 4. Update the parameters and statistics.
            self.tgt_optim.step()

            if report_func is not None:
                report_stats = report_func(
                        epoch, i, nBatches,
                        total_stats.start_time, self.src_optim.lr, report_stats)

        return total_stats

    def add_noise(self, src_lengths, src):
        s_l_n = src_lengths.cpu().numpy()
        s_n = src.data.cpu().numpy()
        
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

        sort_order = np.flip(np.argsort(s_l_n), 0)
        #print s_l_n.size(), sort_order.shape, sort_order
        s_l_n = s_l_n[sort_order]
        s_n = s_n[:, sort_order]

        src_lengths = torch.LongTensor(s_l_n)
        src = Variable(torch.LongTensor(s_n))

        src_lengths.cuda()
        src.cuda()

        return src_lengths, src
    
    def epoch_step(self, ppl, epoch):
        """ Called for each epoch to update learning rate. """
        return self.optim.updateLearningRate(ppl, epoch)

    def drop_checkpoint(self, opt, epoch, fields, valid_stats):
        """ Called conditionally each epoch to save a snapshot. """
        real_src_model = (self.src_model.module
                      if isinstance(self.src_model, nn.DataParallel)
                      else self.src_model)

        src_model_state_dict = real_src_model.state_dict()
        src_model_state_dict = {k: v for k, v in src_model_state_dict.items()
                            if 'generator' not in k}

        real_tgt_model = (self.tgt_model.module
                      if isinstance(self.tgt_model, nn.DataParallel)
                      else self.tgt_model)

        tgt_model_state_dict = real_tgt_model.state_dict()
        tgt_model_state_dict = {k: v for k, v in tgt_model_state_dict.items()
                            if 'generator' not in k}
        checkpoint = {
            'src_model': src_model_state_dict,
            'tgt_model': tgt_model_state_dict,
            'opt': opt,
            'epoch': epoch,
            'src_optim': self.src_optim,
            'tgt_optim': self.tgt_optim
        }
        torch.save(checkpoint,
                   '%s_acc_%.2f_ppl_%.2f_e%d.pt.disc'
                   % (opt.save_model, valid_stats.accuracy(),
                      valid_stats.ppl(), epoch))

class DoubleDiscrimTrainer(object):
    def __init__(self, model, train_iter, discrim_optim, gener_optim, true_label, shard_size):
        """
        Args:
            model: the seq2seq model.
            train_iter: the train data iterator.
            optim: the optimizer responsible for lr update.
            trunc_size: a batch is divided by several truncs of this size.
            shard_size: compute loss in shards of this size for efficiency.
        """
        # Basic attributes.
        self.model = model
        self.train_iter = train_iter
        self.discrim_optim = discrim_optim
        self.gener_optim = gener_optim
        self.true_label = true_label
        self.shard_size = shard_size
        self.criterion = nn.CrossEntropyLoss()
        # Set model in training mode.
        self.model.train()

    def train(self, epoch, report_func=None):
        """ Called for each epoch to train. """
        total_stats = Statistics()
        report_stats = Statistics()
        
        for i, batch in enumerate(self.train_iter):

            _, src_lengths = batch.src

            src = onmt.io.make_features(batch, 'src')

            #src_lengths, src = self.add_noise(src_lengths, src)

            report_stats.n_src_words += src_lengths.sum()

            # compute outputs
            self.model.zero_grad()
            outputs = self.model(src, src_lengths)

            # loss re: true_label, backprop through discrim
            true_l = [self.true_label]*outputs.size()[0]
            labels = Variable(torch.cuda.LongTensor(true_l))

            loss = self.criterion(outputs, labels)
            loss.backward(retain_graph=True)
            if i % 10 == 0:
                print "discriminator:", i, outputs.data[0:5]
            
            total_stats.update_loss(loss.data[0])
            report_stats.update_loss(loss.data[0])

            self.discrim_optim.step()

            # loss re: false_label, backprop through generator
            self.model.zero_grad()
            #outputs = self.model(src, src_lengths)
            fake_l = [1-self.true_label]*outputs.size()[0]
            labels = Variable(torch.cuda.LongTensor(fake_l))

            loss = self.criterion(outputs, labels)
            loss.backward()
            if i % 10 == 0:
                print "generator:", i, outputs.data[0:5]
            
            total_stats.update_loss(loss.data[0])
            report_stats.update_loss(loss.data[0])

            self.gener_optim.step()

            if report_func is not None:
                report_stats = report_func(
                        epoch, i, len(self.train_iter),
                        total_stats.start_time, self.discrim_optim.lr, report_stats)

        return total_stats

    def add_noise(self, src_lengths, src):
        s_l_n = src_lengths.cpu().numpy()
        s_n = src.data.cpu().numpy()
        
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

        sort_order = np.flip(np.argsort(s_l_n), 0)
        #print s_l_n.size(), sort_order.shape, sort_order
        s_l_n = s_l_n[sort_order]
        s_n = s_n[:, sort_order]

        src_lengths = torch.cuda.LongTensor(s_l_n)
        src = Variable(torch.cuda.LongTensor(s_n))

        src_lengths.cuda()
        src.cuda()

        return src_lengths, src

    def epoch_step(self, ppl, epoch):
        """ Called for each epoch to update learning rate. """
        self.discrim_optim.updateLearningRate(ppl, epoch)
        return self.gener_optim.updateLearningRate(ppl, epoch)

    def drop_checkpoint(self, opt, epoch, fields, valid_stats):
        """ Called conditionally each epoch to save a snapshot. """
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
            'vocab': onmt.io.save_vocab(fields),
            'opt': opt,
            'epoch': epoch,
            'optim': self.optim
        }
        torch.save(checkpoint,
                   '%s_acc_%.2f_ppl_%.2f_e%d.pt'
                   % (opt.save_model, valid_stats.accuracy(),
                      valid_stats.ppl(), epoch))
