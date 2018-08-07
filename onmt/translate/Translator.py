import torch
from torch.autograd import Variable

import onmt.translate.Beam
import onmt.io

import numpy

class Translator(object):
    """
    Uses a model to translate a batch of sentences.


    Args:
       model (:obj:`onmt.modules.NMTModel`):
          NMT model to use for translation
       fields (dict of Fields): data fields
       beam_size (int): size of beam to use
       n_best (int): number of translations produced
       max_length (int): maximum length output to produce
       global_scores (:obj:`GlobalScorer`):
         object to rescore final translations
       copy_attn (bool): use copy attention during translation
       cuda (bool): use cuda
       beam_trace (bool): trace beam search for debugging
    """
    def __init__(self, model, fields,
                 beam_size, lm=None, n_best=1,
                 max_length=100,
                 global_scorer=None, copy_attn=False, cuda=False,
                 beam_trace=False, min_length=0, alpha=1., beta=0.5, gamma=1.):
        self.model = model
        self.fields = fields
        self.n_best = n_best
        self.max_length = max_length
        self.global_scorer = global_scorer
        self.copy_attn = copy_attn
        self.beam_size = beam_size
        self.cuda = cuda
        self.min_length = min_length

        self.lm = lm
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.logsoftmax = torch.nn.LogSoftmax(dim=2)

        print "weights:", alpha, beta, gamma
        print "nBest:", n_best
        # for debugging
        self.beam_accum = None
        if beam_trace:
            self.beam_accum = {
                "predicted_ids": [],
                "beam_parent_ids": [],
                "scores": [],
                "log_probs": []}

    def translate_batch(self, batch, data):
        """
        Translate a batch of sentences.

        Mostly a wrapper around :obj:`Beam`.

        Args:
           batch (:obj:`Batch`): a batch from a dataset object
           data (:obj:`Dataset`): the dataset object


        Todo:
           Shouldn't need the original dataset.
        """

        # (0) Prep each of the components of the search.
        # And helper method for reducing verbosity.
        beam_size = self.beam_size
        batch_size = batch.batch_size
        data_type = data.data_type
        vocab = self.fields["tgt"].vocab
        beam = [onmt.translate.Beam(beam_size, n_best=self.n_best,
                                    cuda=self.cuda,
                                    global_scorer=self.global_scorer,
                                    pad=vocab.stoi[onmt.io.PAD_WORD],
                                    eos=vocab.stoi[onmt.io.EOS_WORD],
                                    bos=vocab.stoi[onmt.io.BOS_WORD],
                                    min_length=self.min_length)
                for __ in range(batch_size)]

        
        # Help functions for working with beams and batches
        def var(a): return Variable(a, volatile=True)

        def rvar(a): return var(a.repeat(1, beam_size, 1))

        def bottle(m):
            return m.view(batch_size * beam_size, -1)

        def unbottle(m):
            return m.view(beam_size, batch_size, -1)

        # (1) Run the encoder on the src.
        src = onmt.io.make_features(batch, 'src', data_type)
        print "tb:", src[:, 0, :]
        src_lengths = None
        if data_type == 'text':
            _, src_lengths = batch.src

        enc_states, context = self.model.encoder(src, src_lengths)
        dec_states = self.model.decoder.init_decoder_state(
                                        src, context, enc_states)
            
        if src_lengths is None:
            src_lengths = torch.Tensor(batch_size).type_as(context.data)\
                                                  .long()\
                                                  .fill_(context.size(0))

        # (2) Repeat src objects `beam_size` times.
        src_map = rvar(batch.src_map.data) \
            if data_type == 'text' and self.copy_attn else None
        context = rvar(context.data)
        context_lengths = src_lengths.repeat(beam_size)
        dec_states.repeat_beam_size_times(beam_size)
    
        if self.lm:
            print "b, b:", batch_size, beam_size, batch_size*beam_size
            lm_states, lm_weights, lm_costs = self.lm.initial_states(batch_size*beam_size)
            #print len(lm_states), len(lm_weights), len(lm_costs)
            #print lm_states[0].size(), lm_weights[0].size(), lm_costs
            #print lm_costs
            
        # (3) run the decoder to generate sentences, using beam search.
        for i in range(self.max_length):
            if all((b.done() for b in beam)):
                break

            # Construct batch x beam_size nxt words.
            # Get all the pending current beam words and arrange for forward.
            inp = var(torch.stack([b.get_current_state() for b in beam])
                      .t().contiguous().view(1, -1))

            # inp = 1 x (batch*beam)
            
            # Turn any copied words to UNKs
            # 0 is unk
            if self.copy_attn:
                inp = inp.masked_fill(
                    inp.gt(len(self.fields["tgt"].vocab) - 1), 0)

            # Temporary kludge solution to handle changed dim expectation
            # in the decoder
            inp = inp.unsqueeze(2)

            # Run one step.
            dec_out, dec_states, attn = self.model.decoder(
                inp, context, dec_states, context_lengths=context_lengths, step=i)
            dec_out = dec_out.squeeze(0)
            # dec_out: beam x rnn_size

            # (b) Compute a vector of batch*beam word scores.
            if not self.copy_attn:
                out = self.model.generator.forward(dec_out).data
                out = unbottle(out)
                # beam x tgt_vocab
            else:
                out = self.model.generator.forward(dec_out,
                                                   attn["copy"].squeeze(0),
                                                   src_map)
                # beam x (tgt_vocab + extra_vocab)
                out = data.collapse_copy_scores(
                    unbottle(out.data),
                    batch, self.fields["tgt"].vocab)
                # beam x tgt_vocab
                out = out.log()

            if self.lm:
                # out: beam x batch x outputs
                #print "OUT:", i, out.size()
                #print out[0, :, 10]
                #lmscores = torch.cuda.FloatTensor(out.size()).zero_()
                #lmscores[:, :, 10] = -10
                print i
                #print out.size(), out[0, 0, :]
                #print out.size(), out[1, 0, :]
                input_ = inp.squeeze().cpu().data.numpy()
                if i == 0:
                    input_[:] = 2
                input_ = input_.tolist()
                #print input_[:beam_size]
                
                lm_states, lm_weights = self.lm.transition(input_, lm_states, lm_weights)
                #print len(lm_states), len(lm_weights)
                lm_out = self.lm.probability_computer(lm_states, lm_weights)
                print "LM BEFORE:", len(lm_out), lm_out[0]
                print "b, b:", beam_size, batch_size
                lm_out = - torch.cuda.FloatTensor(numpy.asarray(lm_out)).view(beam_size, batch_size, -1)
                shifted = lm_out - lm_out.max(dim=2, keepdim=True)[0]
                lm_out = shifted - shifted.exp().sum(dim=2, keepdim=True).log()
                #print "LM AFTER:", lm_out[0, 0, :]
                #print lm_out[1, 0, :]
                #print "AM BEFORE:", out[0, 0, :]
                #shifted = out - out.max(dim=2, keepdim=True)[0]
                #out = shifted - shifted.exp().sum(dim=2, keepdim=True).log()
                #print "AM AFTER:", out[0, 0, :]
                #print out[1, 0, :]
                #print "LM:", lm_out[:, 0, :].max(dim=1)[1]
                #print "AM:", out[:, 0, :].max(dim=1)[1]
                out = self.alpha*out + self.beta*lm_out
                #shifted = out - out.max(dim=2, keepdim=True)[0]
                #out = shifted - shifted.exp().sum(dim=2, keepdim=True).log()
                #print "AM+LM:", out[:, 0, :].max(dim=1)[1]
                
                out = out + self.gamma*i
                #print "AM+LM+len:", out[:, 0, :].max(dim=1)[1]
                #print "COMBINED:", out[0, 0, :]
                #print 
            
            # (c) Advance each beam.
            updated_lm_states = []
            updated_lm_weights = []
            for j, b in enumerate(beam):
                attn_in = unbottle(attn["std"]).data[:, j, :context_lengths[j]]
                b.advance(out[:, j], attn_in)
                dec_states.beam_update(j, b.get_current_origin(), beam_size)
                if self.lm:
                    for p in b.get_current_origin():
                        #print j, p, j*beam_size+p
                        updated_lm_states.append(lm_states[j*beam_size+p])
                        updated_lm_weights.append(lm_weights[j*beam_size+p])
            if self.lm:
                lm_states = updated_lm_states
                lm_weights = updated_lm_weights
            
        # (4) Extract sentences from beam.
        ret = self._from_beam(beam)
        ret["gold_score"] = [0] * batch_size
        if "tgt" in batch.__dict__:
            ret["gold_score"] = self._run_target(batch, data)
        ret["batch"] = batch
        return ret

    def _from_beam(self, beam):
        ret = {"predictions": [],
               "scores": [],
               "attention": []}
        for b in beam:
            n_best = self.n_best
            scores, ks = b.sort_finished(minimum=n_best)
            hyps, attn = [], []
            for i, (times, k) in enumerate(ks[:n_best]):
                hyp, att = b.get_hyp(times, k)
                hyps.append(hyp)
                attn.append(att)
            ret["predictions"].append(hyps)
            ret["scores"].append(scores)
            ret["attention"].append(attn)
        return ret

    def _run_target(self, batch, data):
        data_type = data.data_type
        if data_type == 'text':
            _, src_lengths = batch.src
        else:
            src_lengths = None
        src = onmt.io.make_features(batch, 'src', data_type)
        tgt_in = onmt.io.make_features(batch, 'tgt')[:-1]

        #  (1) run the encoder on the src
        enc_states, context = self.model.encoder(src, src_lengths)
        dec_states = self.model.decoder.init_decoder_state(src,
                                                           context, enc_states)

        #  (2) if a target is specified, compute the 'goldScore'
        #  (i.e. log likelihood) of the target under the model
        tt = torch.cuda if self.cuda else torch
        gold_scores = tt.FloatTensor(batch.batch_size).fill_(0)
        dec_out, dec_states, attn = self.model.decoder(
            tgt_in, context, dec_states, context_lengths=src_lengths)

        tgt_pad = self.fields["tgt"].vocab.stoi[onmt.io.PAD_WORD]
        for dec, tgt in zip(dec_out, batch.tgt[1:].data):
            # Log prob of each word.
            out = self.model.generator.forward(dec)
            tgt = tgt.unsqueeze(1)
            scores = out.data.gather(1, tgt)
            scores.masked_fill_(tgt.eq(tgt_pad), 0)
            print gold_scores
            print scores.squeeze()
            gold_scores += scores.squeeze()
        return gold_scores
