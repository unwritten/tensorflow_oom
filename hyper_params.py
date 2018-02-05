# coding=utf-8

from __future__ import division

class HyperParams:
    """Hyper parameters"""
    
    # signal processing
    sr = 16000 # Sampling rate. Paper => 24000
    n_fft = 2048 # fft points (samples)
    frame_shift = 0.0125 # seconds
    frame_length = 0.05 # seconds
    hop_length = int(sr*frame_shift) # samples  This is dependent on the frame_shift.
    win_length = int(sr*frame_length) # samples This is dependent on the frame_length.
    mel_spec_size = 80 # Number of Mel banks to generate
    linear_spec_size = (n_fft//2 + 1)
    power = 1.2 # Exponent for amplifying the predicted magnitude
    n_iter = 15 # Number of inversion iterations

    char_embed_size = 512
    n_decoder_rnn_layers = 2
    min_encoder_length = 5
    min_decoder_time = 0.5 # in seconds
    max_gradient_norm = 5.0

    chars = " !&',-.0123:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz" # padding index is zero
    char2id = [("padding_char", 0)]
    char2id += [(char, i+1) for i, char in enumerate(chars)]
    char2id = dict(char2id)
    char_vocab_size = len(char2id)

    norm_type = "bn"  # a normalizer function. value: bn, ln, or ins
    loss_type = "l1" # Or you can test "l2"

    preemphasize = 0.97
    ref_level_db = 20
    min_level_db = -100
