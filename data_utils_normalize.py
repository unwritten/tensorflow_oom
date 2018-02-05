# coding=utf-8

from __future__ import division
from __future__ import print_function

import numpy as np
import librosa
import librosa.filters
import soundfile
import tensorflow as tfO
import tensorflow as tf
import time

import sys

from scipy import signal
from hyper_params import HyperParams as hp


np.random.seed(20171013)

def get_mel_and_linear_spec(audio):
    """Compute mel and linear targets from audio"""
    audio_pre = preemphasize(audio)
    D = np.abs(stft(audio_pre))

    S = amp_to_db(linear_to_mel(D))
    mel_spec = normalize(S)

    S = amp_to_db(D) - hp.ref_level_db
    linear_spec = normalize(S)

    return mel_spec.T, linear_spec.T

def get_mel_spec(audio):
    """Compute mel and linear targets from audio"""
    audio_pre = preemphasize(audio)
    D = np.abs(stft(audio_pre))

    S = amp_to_db(linear_to_mel(D))
    mel_spec = normalize(S)

    return mel_spec.T

def get_mel_spectrogram_from_wave(file):
    audio, sr = soundfile.read(file)
    mel = get_mel_spec(audio)
    return mel

def get_mel_spectrogram(audio):
    """Compute mel and linear targets from audio"""

    wave, _ = librosa.effects.trim(audio)

    pre_emphasis = 0.97
    wave = preemphasize(wave)
    wave = np.append(wave[0], wave[1:] - pre_emphasis * wave[:-1])

    # a short-time Fourier transform (STFT)
    # using a 50 ms frame size, 12.5 ms frame
    # hop, and a Hann window function
    stft = librosa.stft(wave, n_fft=hp.n_fft, win_length=hp.win_length, hop_length=hp.hop_length, window='hann')

    # transform the STFT magnitude
    # to the mel-scale using an 80 channel mel filterbank spanning 125 Hz
    # to 7.6 kHz
    params = {
        "fmin" : 125,
        "fmax" : 7600}

    mel = librosa.feature.melspectrogram(S=stft, n_mels=80, sr=hp.sr, **params)

    # use log dynamic range compression. Prior to log
    # compression, the filterbank output magnitudes are stabilized to a floor
    # of 0.01 in order to limit dynamic range in the logarithmic domain
    mel = np.abs(mel)
    mel = np.clip(mel, 1e-8, sys.float_info.max)
    mel = np.log(mel)

    return mel.T, mel.T


def post_process_mel_spec(predicted_mel_spec, wave_filename, raise_power=1.0, trim_tailing_silence=True):
    """Convert predicted linear to audio and save it"""
    converted_linear_spec = mel_to_linear(predicted_mel_spec)
    S = db_to_amp(denormalize(converted_linear_spec) + hp.ref_level_db)
    start = time.time()
    audio = griffin_lim(S ** raise_power)
    print("griffin lim time:{:.1f}".format(time.time() - start))
    audio = inv_preemphasize(audio)

    if trim_tailing_silence:
        audio_trimed, index = librosa.effects.trim(audio)
        soundfile.write(wave_filename, audio[:index[1]+hp.sr//4], hp.sr)
    else:
        soundfile.write(wave_filename, audio, hp.sr)

# Based on https://github.com/librosa/librosa/issues/434
def griffin_lim(S, alpha=0.99):
   angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
   S_complex = np.abs(S).astype(np.complex)
   for i in range(hp.n_iter):
        if i > 0:
            angles = np.exp(1j * np.angle(stft(y)))
        y_new = istft(S_complex * angles)
        if i > 0:
            y = y_new + alpha * (y_new - y_old)
        else:
            y = y_new
        y_old = y_new
   return y

def inv_spectrogram_tensorflow(spectrogram, raise_power=1.5):
    '''Builds computational graph to convert spectrogram to waveform using TensorFlow.
    Unlike inv_spectrogram, this does NOT invert the preemphasis. The caller should call
    inv_preemphasis on the output after running the graph.
    '''
    S = _db_to_amp_tensorflow(_denormalize_tensorflow(spectrogram) + hp.ref_level_db)
    return _griffin_lim_tensorflow(tf.pow(S, raise_power))


def stft(y):
    return librosa.stft(y=y, n_fft=hp.n_fft, hop_length=hp.hop_length, win_length=hp.win_length)


def istft(y):
    return librosa.istft(y, hop_length=hp.hop_length, win_length=hp.win_length)

# Conversions:

mel_basis = None
inv_mel_basis = None

def linear_to_mel(spectrogram):
    global mel_basis
    if mel_basis is None:
        mel_basis = build_mel_basis()
    return np.dot(mel_basis, spectrogram)

def mel_to_linear(mel_spec):
    global inv_mel_basis
    if inv_mel_basis is None:
        inv_mel_basis = np.linalg.pinv(build_mel_basis())
    #mel_transform = np.transpose(inv_mel_basis)
    mel_transform = np.transpose(mel_spec)
    #return np.maximum(1e-10, np.dot(inv_mel_basis, mel_spec))
    return np.maximum(1e-10, np.dot(inv_mel_basis, mel_transform))

# def build_mel_basis():
    # params = {
         # "fmin" : 125,
         # "fmax" : 7600}
    # return librosa.filters.mel(hp.sr, hp.n_fft, n_mels=hp.mel_spec_size, **params)

def build_mel_basis():
    return librosa.filters.mel(hp.sr, hp.n_fft, n_mels=hp.mel_spec_size)

def amp_to_db(x):
    return 20 * np.log10(np.maximum(1e-5, x))

def db_to_amp(x):
    return np.power(10.0, x * 0.05)

def preemphasize(x):
    return signal.lfilter([1, -hp.preemphasize], [1], x)

def inv_preemphasize(x):
    return signal.lfilter([1], [1, -hp.preemphasize], x)

def normalize(S):
    return np.clip((S - hp.min_level_db) / -hp.min_level_db, 0, 1)

def denormalize(S):
    return (np.clip(S, 0, 1) * -hp.min_level_db) + hp.min_level_db

def find_endpoint(wav, threshold_db=-40, min_silence_sec=0.8):
    window_length = int(hp.sr * min_silence_sec)
    hop_length = int(window_length / 4)
    threshold = db_to_amp(threshold_db)
    for x in range(hop_length, len(wav) - window_length, hop_length):
        if np.max(wav[x:x+window_length]) < threshold:
            return x + hop_length
    return len(wav)

def save_wav(wav, path):
    '''
    This code is converting the floating-point output of the model to 16-bit values to be saved as a
    wav file. 32767 is the maximum value for a 16-bit signed integer (2^15 - 1). The 0.01 prevents
    division by zero in case the output is silence.
    '''
    wav *= 32767 / max(0.01, np.max(np.abs(wav)))
    librosa.output.write_wav(path, wav.astype(np.int16), hp.sr)

# tensorflow ops
def _griffin_lim_tensorflow(S):
    '''TensorFlow implementation of Griffin-Lim
    Based on https://github.com/Kyubyong/tensorflow-exercises/blob/master/Audio_Processing.ipynb
    '''
    with tf.variable_scope('griffinlim'):
        S_complex = tf.identity(tf.cast(S, dtype=tf.complex64))
        y = _istft_tensorflow(S_complex)
        for i in range(hp.n_iter):
            est = _stft_tensorflow(y)
            angles = est / tf.cast(tf.maximum(1e-8, tf.abs(est)), tf.complex64)
            y = _istft_tensorflow(S_complex * angles)
    return y

def _stft_tensorflow(signals):
    return tf.contrib.signal.stft(signals, hp.win_length, hp.hop_length, hp.n_fft, pad_end=False)

def _istft_tensorflow(stfts):
    return tf.contrib.signal.inverse_stft(stfts, hp.win_length, hp.hop_length, hp.n_fft)

def _db_to_amp_tensorflow(x):
    return tf.pow(tf.ones(tf.shape(x)) * 10.0, x * 0.05)

def _denormalize_tensorflow(S):
    return (tf.clip_by_value(S, 0, 1) * -hp.min_level_db) + hp.min_level_db

def post_process_linear_spec(predicted_linear_spec, wave_filename, raise_power=1.0, trim_tailing_silence=True):
    """Convert predicted linear to audio and save it"""
    S = db_to_amp(denormalize(predicted_linear_spec.T) + hp.ref_level_db)
    start = time.time()
    audio = griffin_lim(S ** raise_power)
    print("griffin lim time:{:.1f}".format(time.time() - start))
    audio = inv_preemphasize(audio)

    if trim_tailing_silence:
        audio_trimed, index = librosa.effects.trim(audio)
        soundfile.write(wave_filename, audio[:index[1]+hp.sr//4], hp.sr)
    else:
        soundfile.write(wave_filename, audio, hp.sr)
