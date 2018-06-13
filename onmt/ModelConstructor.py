"""
This file is for models creation, which consults options
and creates each encoder and decoder accordingly.
"""
import torch
import torch.nn as nn

import onmt
import onmt.io
import onmt.Models
import onmt.modules
from onmt.Models import NMTModel, MeanEncoder, RNNEncoder, \
                        StdRNNDecoder, InputFeedRNNDecoder, InputFeedRNNDecoderNoAttention, \
                        DiscrimClassifier, DiscrimModel, AudioDecoder, FFAudioDecoder, SpeechModel
from onmt.modules import Embeddings, ImageEncoder, CopyGenerator, \
                         TransformerEncoder, TransformerDecoder, \
                         CNNEncoder, CNNDecoder, AudioEncoder, GlobalAudioEncoder, ConvGlobalAudioEncoder
from onmt.Utils import use_gpu


def make_embeddings(opt, word_dict, feature_dicts, for_encoder=True):
    """
    Make an Embeddings instance.
    Args:
        opt: the option in current environment.
        word_dict(Vocab): words dictionary.
        feature_dicts([Vocab], optional): a list of feature dictionary.
        for_encoder(bool): make Embeddings for encoder or decoder?
    """
    if for_encoder:
        embedding_dim = opt.src_word_vec_size
    else:
        embedding_dim = opt.tgt_word_vec_size

    word_padding_idx = word_dict.stoi[onmt.io.PAD_WORD]
    num_word_embeddings = len(word_dict)

    feats_padding_idx = [feat_dict.stoi[onmt.io.PAD_WORD]
                         for feat_dict in feature_dicts]
    num_feat_embeddings = [len(feat_dict) for feat_dict in
                           feature_dicts]

    return Embeddings(word_vec_size=embedding_dim,
                      position_encoding=opt.position_encoding,
                      feat_merge=opt.feat_merge,
                      feat_vec_exponent=opt.feat_vec_exponent,
                      feat_vec_size=opt.feat_vec_size,
                      dropout=opt.dropout,
                      word_padding_idx=word_padding_idx,
                      feat_padding_idx=feats_padding_idx,
                      word_vocab_size=num_word_embeddings,
                      feat_vocab_sizes=num_feat_embeddings)


def make_encoder(opt, embeddings):
    """
    Various encoder dispatcher function.
    Args:
        opt: the option in current environment.
        embeddings (Embeddings): vocab embeddings for this encoder.
    """
    if opt.encoder_type == "transformer":
        return TransformerEncoder(opt.enc_layers, opt.rnn_size,
                                  opt.dropout, embeddings)
    elif opt.encoder_type == "cnn":
        return CNNEncoder(opt.enc_layers, opt.rnn_size,
                          opt.cnn_kernel_width,
                          opt.dropout, embeddings)
    elif opt.encoder_type == "mean":
        return MeanEncoder(opt.enc_layers, embeddings)
    else:
        # "rnn" or "brnn"
        return RNNEncoder(opt.rnn_type, opt.brnn, opt.enc_layers,
                          opt.rnn_size, opt.dropout, embeddings)


def make_decoder(opt, embeddings):
    """
    Various decoder dispatcher function.
    Args:
        opt: the option in current environment.
        embeddings (Embeddings): vocab embeddings for this decoder.
    """
    if opt.decoder_type == "transformer":
        return TransformerDecoder(opt.dec_layers, opt.rnn_size,
                                  opt.global_attention, opt.copy_attn,
                                  opt.dropout, embeddings)
    elif opt.decoder_type == "cnn":
        return CNNDecoder(opt.dec_layers, opt.rnn_size,
                          opt.global_attention, opt.copy_attn,
                          opt.cnn_kernel_width, opt.dropout,
                          embeddings)
    elif opt.input_feed:
        if opt.global_attention=='none':
            return InputFeedRNNDecoderNoAttention(opt.rnn_type, opt.brnn,
                                    opt.dec_layers, opt.rnn_size,
                                    'general',
                                    opt.coverage_attn,
                                    opt.context_gate,
                                    opt.copy_attn,
                                    opt.dropout,
                                    embeddings)
        else:
            return InputFeedRNNDecoder(opt.rnn_type, opt.brnn,
                                    opt.dec_layers, opt.rnn_size,
                                    opt.global_attention,
                                    opt.coverage_attn,
                                    opt.context_gate,
                                    opt.copy_attn,
                                    opt.dropout,
                                    embeddings)
    else:
        return StdRNNDecoder(opt.rnn_type, opt.brnn,
                             opt.dec_layers, opt.rnn_size,
                             opt.global_attention,
                             opt.coverage_attn,
                             opt.context_gate,
                             opt.copy_attn,
                             opt.dropout,
                             embeddings)

def make_discriminators(opt, encoders, gpu, checkpoint):
    classifier = DiscrimClassifier(opt.rnn_size*2, opt.discrim_size, opt.discrim_layers)
    models = [DiscrimModel(encoder, classifier) for encoder in encoders]

    if checkpoint is not None:
        print('Loading model parameters.')
        models[0].load_state_dict(checkpoint['src_model'])
        models[1].load_state_dict(checkpoint['tgt_model'])

    # Make the whole model leverage GPU if indicated to do so.
    if gpu:
        for model in models:
            model.cuda()
    else:
        for model in models:
            model.cpu()

    return models
    
def load_test_model(opt, dummy_opt):
    checkpoint = torch.load(opt.model,
                            map_location=lambda storage, loc: storage)
    fields = onmt.io.load_fields_from_vocab(
        checkpoint['vocab'], data_type=opt.data_type)

    model_opt = checkpoint['opt']
    for arg in dummy_opt:
        if arg not in model_opt:
            model_opt.__dict__[arg] = dummy_opt[arg]

    model = make_base_model(model_opt, fields,
                            use_gpu(opt), checkpoint)
    model.eval()
    model.generator.eval()
    return fields, model, model_opt

def load_semi_sup_test_model(opt, dummy_opt):
    checkpoint = torch.load(opt.model,
                            map_location=lambda storage, loc: storage)
    fields = onmt.io.load_fields_from_vocab(
        checkpoint['vocab'], data_type=opt.data_type)

    text_fields = onmt.io.load_fields_from_vocab(
                torch.load(opt.vocab), "text")
    text_fields['tgt'] = fields['tgt']

    print(' * vocabulary size. source = %d; target = %d' %
          (len(text_fields['src'].vocab), len(text_fields['tgt'].vocab)))

    model_opt = checkpoint['opt']
    for arg in dummy_opt:
        if arg not in model_opt:
            model_opt.__dict__[arg] = dummy_opt[arg]

    model, text_model, _ = make_audio_text_model(model_opt, fields, text_fields, use_gpu(opt), checkpoint)
    
    model.eval()
    text_model.eval()
    model.generator.eval()
    text_model.generator.eval()

    return fields, text_fields, model, text_model, model_opt

def make_audio_text_model(model_opt, fields, text_fields, gpu, checkpoint=None):
    model = make_base_model(model_opt, fields, gpu, checkpoint)
    
    src_dict = text_fields["src"].vocab
    feature_dicts = onmt.io.collect_feature_vocabs(text_fields, 'src')
    src_embeddings = make_embeddings(model_opt, src_dict, feature_dicts)
    text_encoder = make_encoder(model_opt, src_embeddings)

    generator = model.generator
    
    text_model = NMTModel(text_encoder, model.decoder)
    text_model.model_type = 'text'
    text_model.decoder.set_generator(None)

    try:
        if model_opt.conv_global_encoder:
            global_speech_encoder = ConvGlobalAudioEncoder(model_opt.enc_layers,
                               model_opt.brnn,
                               model_opt.rnn_size,
                               model_opt.dropout,
                               model_opt.sample_rate,
                               model_opt.window_size)
        else:
            global_speech_encoder = GlobalAudioEncoder(model_opt.enc_layers,
                               model_opt.brnn,
                               model_opt.rnn_size,
                               model_opt.dropout,
                               model_opt.sample_rate,
                               model_opt.window_size)
    except:
        global_speech_encoder = GlobalAudioEncoder(model_opt.enc_layers,
                               model_opt.brnn,
                               model_opt.rnn_size,
                               model_opt.dropout,
                               model_opt.sample_rate,
                               model_opt.window_size)

    try:
        print "ff:", model_opt.ff_speech_decoder
    
        if model_opt.ff_speech_decoder:
            speech_decoder = FFAudioDecoder(model_opt.rnn_size*3, model_opt.rnn_size, model_opt.dec_layers)
        else:
            speech_decoder = AudioDecoder(model_opt.rnn_type, model_opt.brnn,
                             model_opt.dec_layers, model_opt.rnn_size,
                             model_opt.global_attention,
                             model_opt.coverage_attn,
                             model_opt.context_gate,
                             model_opt.copy_attn,
                             model_opt.dropout)
    except:
        speech_decoder = AudioDecoder(model_opt.rnn_type, model_opt.brnn,
                             model_opt.dec_layers, model_opt.rnn_size,
                             model_opt.global_attention,
                             model_opt.coverage_attn,
                             model_opt.context_gate,
                             model_opt.copy_attn,
                             model_opt.dropout)
        
    speech_model = SpeechModel(model.encoder, global_speech_encoder, speech_decoder)
    
    # Load the model states from checkpoint or initialize them.
    if checkpoint is not None:
        print('Loading model parameters.')
        text_model.load_state_dict(checkpoint['text_model'])
        if checkpoint.has_key('speech_model') and checkpoint['speech_model'] is not None:
            print('  Loading speech model parameters')
            speech_model.load_state_dict(checkpoint['speech_model'])
    else:
        if model_opt.param_init != 0.0:
            print('Intializing model parameters.')
            for p in text_model.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)
            for p in speech_model.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)
        if hasattr(text_model.encoder, 'embeddings'):
            text_model.encoder.embeddings.load_pretrained_vectors(
                    model_opt.pre_word_vecs_enc, model_opt.fix_word_vecs_enc)

    # Add generator to model (this registers it as parameter of model).
    text_model.decoder.set_generator(generator)
    text_model.generator = generator

    # Make the whole model leverage GPU if indicated to do so.
    if gpu:
        text_model.cuda()
        speech_model.cuda()
    else:
        text_model.cpu()
        speech_model.cpu()

    return model, text_model, speech_model

def make_audio_text_model_from_text(model_opt, fields, text_fields, gpu, checkpoint=None):
    model = make_base_model(model_opt, fields, gpu, None)
    
    src_dict = text_fields["src"].vocab
    feature_dicts = onmt.io.collect_feature_vocabs(text_fields, 'src')
    src_embeddings = make_embeddings(model_opt, src_dict, feature_dicts)
    text_encoder = make_encoder(model_opt, src_embeddings)

    generator = model.generator
    generator.load_state_dict(checkpoint['generator'])

    text_model = NMTModel(text_encoder, model.decoder)
    text_model.model_type = 'text'
    text_model.decoder.set_generator(None)

    try:
        if model_opt.conv_global_encoder:
            global_speech_encoder = ConvGlobalAudioEncoder(model_opt.enc_layers,
                               model_opt.brnn,
                               model_opt.rnn_size,
                               model_opt.dropout,
                               model_opt.sample_rate,
                               model_opt.window_size)
        else:
            global_speech_encoder = GlobalAudioEncoder(model_opt.enc_layers,
                               model_opt.brnn,
                               model_opt.rnn_size,
                               model_opt.dropout,
                               model_opt.sample_rate,
                               model_opt.window_size)
    except:
        global_speech_encoder = GlobalAudioEncoder(model_opt.enc_layers,
                               model_opt.brnn,
                               model_opt.rnn_size,
                               model_opt.dropout,
                               model_opt.sample_rate,
                               model_opt.window_size)

        
    print "ff:", model_opt.ff_speech_decoder
    
    if model_opt.ff_speech_decoder:
        speech_decoder = FFAudioDecoder(model_opt.rnn_size*3, model_opt.rnn_size, model_opt.dec_layers)
    else:
        speech_decoder = AudioDecoder(model_opt.rnn_type, model_opt.brnn,
                             model_opt.dec_layers, model_opt.rnn_size,
                             model_opt.global_attention,
                             model_opt.coverage_attn,
                             model_opt.context_gate,
                             model_opt.copy_attn,
                             model_opt.dropout)

    speech_model = SpeechModel(model.encoder, global_speech_encoder, speech_decoder)

    if model_opt.param_init != 0.0:
        print('Intializing model parameters.')
        for p in text_model.parameters():
            p.data.uniform_(-model_opt.param_init, model_opt.param_init)
        for p in speech_model.parameters():
            p.data.uniform_(-model_opt.param_init, model_opt.param_init)
    if hasattr(text_model.encoder, 'embeddings'):
        text_model.encoder.embeddings.load_pretrained_vectors(
            model_opt.pre_word_vecs_enc, model_opt.fix_word_vecs_enc)

    # Load the model states from checkpoint or initialize them.
    if checkpoint is not None:
        print('Loading model parameters.')
        text_model.load_state_dict(checkpoint['model'])
        if checkpoint.has_key('speech_model') and checkpoint['speech_model'] is not None:
            print('  Loading speech model parameters')
            speech_model.load_state_dict(checkpoint['speech_model'])

    # Add generator to model (this registers it as parameter of model).
    text_model.decoder.set_generator(generator)
    text_model.generator = generator

    # Make the whole model leverage GPU if indicated to do so.
    if gpu:
        text_model.cuda()
        speech_model.cuda()
    else:
        text_model.cpu()
        speech_model.cpu()

    return model, text_model, speech_model


def make_base_model(model_opt, fields, gpu, checkpoint=None):
    """
    Args:
        model_opt: the option loaded from checkpoint.
        fields: `Field` objects for the model.
        gpu(bool): whether to use gpu.
        checkpoint: the model gnerated by train phase, or a resumed snapshot
                    model from a stopped training.
    Returns:
        the NMTModel.
    """
    assert model_opt.model_type in ["text", "img", "audio"], \
        ("Unsupported model type %s" % (model_opt.model_type))

    # Make encoder.
    if model_opt.model_type == "text":
        src_dict = fields["src"].vocab
        feature_dicts = onmt.io.collect_feature_vocabs(fields, 'src')
        src_embeddings = make_embeddings(model_opt, src_dict,
                                         feature_dicts)
        encoder = make_encoder(model_opt, src_embeddings)
    elif model_opt.model_type == "img":
        encoder = ImageEncoder(model_opt.enc_layers,
                               model_opt.brnn,
                               model_opt.rnn_size,
                               model_opt.dropout)
    elif model_opt.model_type == "audio":
        encoder = AudioEncoder(model_opt.enc_layers,
                               model_opt.brnn,
                               model_opt.rnn_size,
                               model_opt.dropout,
                               model_opt.sample_rate,
                               model_opt.window_size)

    # Make decoder.
    tgt_dict = fields["tgt"].vocab
    feature_dicts = onmt.io.collect_feature_vocabs(fields, 'tgt')
    tgt_embeddings = make_embeddings(model_opt, tgt_dict,
                                     feature_dicts, for_encoder=False)

    # Share the embedding matrix - preprocess with share_vocab required.
    if model_opt.share_embeddings:
        # src/tgt vocab should be the same if `-share_vocab` is specified.
        if src_dict != tgt_dict:
            raise AssertionError('The `-share_vocab` should be set during '
                                 'preprocess if you use share_embeddings!')

        tgt_embeddings.word_lut.weight = src_embeddings.word_lut.weight

    decoder = make_decoder(model_opt, tgt_embeddings)
    if hasattr(model_opt, "attn_window"):
        decoder.attn_window = model_opt.attn_window
    
    # Make NMTModel(= encoder + decoder).
    model = NMTModel(encoder, decoder)
    model.model_type = model_opt.model_type

    gen_in_size = model_opt.rnn_size
    if model_opt.brnn:
        gen_in_size = gen_in_size*2
        
    # Make Generator.
    if not model_opt.copy_attn:
        generator = nn.Sequential(
            nn.Linear(gen_in_size, len(fields["tgt"].vocab)),
            nn.LogSoftmax())
        if model_opt.share_decoder_embeddings:
            generator[0].weight = decoder.embeddings.word_lut.weight
    else:
        generator = CopyGenerator(gen_in_size,
                                  fields["tgt"].vocab)

    # Load the model states from checkpoint or initialize them.
    if checkpoint is not None:
        print('Loading model parameters.')
        model.load_state_dict(checkpoint['model'])
        generator.load_state_dict(checkpoint['generator'])
    else:
        if model_opt.param_init != 0.0:
            print('Intializing model parameters.')
            for p in model.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)
            for p in generator.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)
        if hasattr(model.encoder, 'embeddings'):
            model.encoder.embeddings.load_pretrained_vectors(
                    model_opt.pre_word_vecs_enc, model_opt.fix_word_vecs_enc)
        if hasattr(model.decoder, 'embeddings'):
            model.decoder.embeddings.load_pretrained_vectors(
                    model_opt.pre_word_vecs_dec, model_opt.fix_word_vecs_dec)

    decoder.set_generator(generator)
    
    # Add generator to model (this registers it as parameter of model).
    model.generator = generator

    # Make the whole model leverage GPU if indicated to do so.
    if gpu:
        model.cuda()
    else:
        model.cpu()

    return model

def make_unsup_model(model_opt, fields, gpu, checkpoint=None):
    """
    Args:
        model_opt: the option loaded from checkpoint.
        fields: `Field` objects for the model.
        gpu(bool): whether to use gpu.
        checkpoint: the model gnerated by train phase, or a resumed snapshot
                    model from a stopped training.
    Returns:
        the NMTModel.
    """
    assert model_opt.model_type in ["text", "img"], \
        ("Unsupported model type %s" % (model_opt.model_type))

    # JD! FIX FIELDS - SEE MAKE_AUTO_MODEL!!
    # Make encoder.
    src_dict = fields["src"].vocab
    feature_dicts = onmt.IO.collect_feature_dicts(fields, 'src')
    src_embeddings = make_embeddings(model_opt, src_dict, feature_dicts)
    src_encoder = make_encoder(model_opt, src_embeddings)

    # Make decoder.
    tgt_dict = fields["tgt"].vocab
    feature_dicts = onmt.IO.collect_feature_dicts(fields, 'tgt')
    tgt_embeddings = make_embeddings(model_opt, tgt_dict,
                                     feature_dicts, for_encoder=False)
    tgt_encoder = make_encoder(model_opt, tgt_embeddings)

    # Share the embedding matrix - preprocess with share_vocab required
    if model_opt.share_embeddings:
        tgt_embeddings.word_lut.weight = src_embeddings.word_lut.weight

    src_decoder = make_decoder(model_opt, src_embeddings)
    tgt_decoder = make_decoder(model_opt, tgt_embeddings)

    # Make NMTModel(= encoder + decoder).
    src_auto_model = NMTModel(src_encoder, src_decoder)
    tgt_auto_model = NMTModel(tgt_encoder, tgt_decoder)
    src_tgt_model = NMTModel(src_encoder, tgt_decoder)
    tgt_src_model = NMTModel(tgt_encoder, src_decoder)

    # Make Generator.
    src_generator = nn.Sequential(
        nn.Linear(model_opt.rnn_size, len(fields["src"].vocab)),
        nn.LogSoftmax())
    tgt_generator = nn.Sequential(
        nn.Linear(model_opt.rnn_size, len(fields["tgt"].vocab)),
        nn.LogSoftmax())

    # Load the model states from checkpoint or initialize them.
    # JD TODO: FIX THIS!
    if checkpoint is not None:
        print('Loading model parameters.')
        src_auto_model.load_state_dict(checkpoint['src_auto_model'])
        tgt_auto_model.load_state_dict(checkpoint['tgt_auto_model'])
        src_tgt_model.load_state_dict(checkpoint['src_tgt_model'])
        tgt_src_model.load_state_dict(checkpoint['tgt_src_model'])
        src_generator.load_state_dict(checkpoint['src_generator'])
        tgt_generator.load_state_dict(checkpoint['tgt_generator'])
    else:
        if model_opt.param_init != 0.0:
            print('Intializing model parameters.')
            for p in src_encoder.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)
            for p in src_decoder.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)
            for p in tgt_encoder.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)
            for p in tgt_decoder.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)
                
            src_encoder.embeddings.load_pretrained_vectors(
                model_opt.pre_word_vecs_enc, model_opt.fix_word_vecs_enc)
            src_decoder.embeddings.load_pretrained_vectors(
                model_opt.pre_word_vecs_enc, model_opt.fix_word_vecs_enc)
            tgt_encoder.embeddings.load_pretrained_vectors(
                model_opt.pre_word_vecs_dec, model_opt.fix_word_vecs_dec)
            tgt_decoder.embeddings.load_pretrained_vectors(
                model_opt.pre_word_vecs_dec, model_opt.fix_word_vecs_dec)

            for p in src_generator.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)
            for p in tgt_generator.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)

    # Add generator to model (this registers it as parameter of model).
    src_auto_model.generator = src_generator
    tgt_auto_model.generator = tgt_generator
    tgt_src_model.generator = src_generator
    src_tgt_model.generator = tgt_generator
    
    # Make the whole model leverage GPU if indicated to do so.
    if gpu:
        src_auto_model.cuda()
        tgt_auto_model.cuda()
        tgt_src_model.cuda()
        src_tgt_model.cuda()
    else:
        src_auto_model.cpu()
        tgt_auto_model.cpu()
        tgt_src_model.cpu()
        src_tgt_model.cpu()

    return src_auto_model, tgt_auto_model, tgt_src_model, src_tgt_model

def make_auto_model(model_opt, fields, gpu, checkpoint=None):
    """
    Args:
        model_opt: the option loaded from checkpoint.
        fields: `Field` objects for the model.
        gpu(bool): whether to use gpu.
        checkpoint: the model gnerated by train phase, or a resumed snapshot
                    model from a stopped training.
    Returns:
        the NMTModel.
    """
    assert model_opt.model_type in ["text", "img"], \
        ("Unsupported model type %s" % (model_opt.model_type))

    src_fields = fields[0]
    tgt_fields = fields[1]
    
    # Make encoder.
    src_dict = src_fields["src"].vocab
    feature_dicts = onmt.IO.collect_feature_dicts(src_fields, 'src')
    src_embeddings = make_embeddings(model_opt, src_dict, feature_dicts)
    src_encoder = make_encoder(model_opt, src_embeddings)
    src_embeddings = make_embeddings(model_opt, src_dict, feature_dicts, for_encoder=False)
    src_decoder = make_decoder(model_opt, src_embeddings)

    # Make decoder.
    tgt_dict = tgt_fields["src"].vocab
    feature_dicts = onmt.IO.collect_feature_dicts(tgt_fields, 'src')
    tgt_embeddings = make_embeddings(model_opt, tgt_dict, feature_dicts)
    tgt_encoder = make_encoder(model_opt, tgt_embeddings)
    tgt_embeddings = make_embeddings(model_opt, tgt_dict, feature_dicts, for_encoder=False)
    tgt_decoder = make_decoder(model_opt, tgt_embeddings)

    # Make NMTModel(= encoder + decoder).
    src_auto_model = NMTModel(src_encoder, src_decoder)
    tgt_auto_model = NMTModel(tgt_encoder, tgt_decoder)

    # Make Generator.
    src_generator = nn.Sequential(
        nn.Linear(model_opt.rnn_size, len(src_fields["tgt"].vocab)),
        nn.LogSoftmax())
    tgt_generator = nn.Sequential(
        nn.Linear(model_opt.rnn_size, len(tgt_fields["tgt"].vocab)),
        nn.LogSoftmax())

    # Load the model states from checkpoint or initialize them.
    if checkpoint is not None:
        # JD TODO: FIX THIS!
        print('Loading model parameters.')
        src_auto_model.load_state_dict(checkpoint['src_auto_model'])
        tgt_auto_model.load_state_dict(checkpoint['tgt_auto_model'])
        src_generator.load_state_dict(checkpoint['src_generator'])
        tgt_generator.load_state_dict(checkpoint['tgt_generator'])
    else:
        if model_opt.param_init != 0.0:
            print('Intializing model parameters.')
            for p in src_encoder.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)
            for p in src_decoder.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)
            for p in tgt_encoder.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)
            for p in tgt_decoder.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)
                
            src_encoder.embeddings.load_pretrained_vectors(
                model_opt.pre_word_vecs_enc, model_opt.fix_word_vecs_enc)
            src_decoder.embeddings.load_pretrained_vectors(
                model_opt.pre_word_vecs_enc, model_opt.fix_word_vecs_enc)
            tgt_encoder.embeddings.load_pretrained_vectors(
                model_opt.pre_word_vecs_dec, model_opt.fix_word_vecs_dec)
            tgt_decoder.embeddings.load_pretrained_vectors(
                model_opt.pre_word_vecs_dec, model_opt.fix_word_vecs_dec)

            for p in src_generator.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)
            for p in tgt_generator.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)

    # Add generator to model (this registers it as parameter of model).
    src_auto_model.generator = src_generator
    tgt_auto_model.generator = tgt_generator
    
    # Make the whole model leverage GPU if indicated to do so.
    if gpu:
        src_auto_model.cuda()
        tgt_auto_model.cpu()
    else:
        src_auto_model.cpu()
        tgt_auto_model.cpu()

    return src_auto_model, tgt_auto_model

