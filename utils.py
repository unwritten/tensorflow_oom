# coding=utf-8

from __future__ import print_function

import argparse
import codecs
import data_utils
import data_utils_normalize
import io
import json
import numpy as np
import os
from os.path import basename
import pickle
import soundfile
import sys
import threading
import time
import zipfile

import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt # drawing heat map of attention weights
import matplotlib.mlab as mlab
plt.rcParams['font.sans-serif']=['SimSun'] # set font family

from collections import Counter
from pprint import pprint
from hyper_params import HyperParams as hp

def is_diverging(loss_sum):
    n = len(loss_sum)
    # if n == 5:
    #    return True
    if n < 10010:
        return False
    n_long = min(n-1, 3000)
    n_short = min(n-1, 500)
    long_average = (loss_sum[-n_short] - loss_sum[-n_long]) / (n_long-n_short)
    short_average = (loss_sum[-1] - loss_sum[-n_short-1]) / n_short
    if short_average > 1.1*long_average:
        return True

    return False

def restore_model(session, saver, path):
    print("Restoring model from {}".format(path))
    # checkpoint_path = tf.train.latest_checkpoint(path)
    checkpoint_path = path
    print("checkpoint path", checkpoint_path)
    saver.restore(session, checkpoint_path)
    cur_global_step = int(checkpoint_path.split("-")[-1])
    print("Model restored from {} successfully, current global step is {}".format(checkpoint_path, cur_global_step))
    return cur_global_step

def ensure_dir(dir):
    if not os.path.isdir(dir):
        print("Making directory", dir)
        os.mkdir(dir)

def generate_wave_for_test_data(test_batch, dir, post_process_func, use_stop_token):
    ensure_dir(dir)
    #test_char_ids, test_mel_spec, test_linear_spec, test_bucket_id, test_wave_ids = test_batch
    if use_stop_token is True:
        test_char_ids, test_mel_spec, test_bucket_id, test_wave_ids, test_stop_tokens= test_batch
    else:
        test_char_ids, test_mel_spec, test_bucket_id, test_wave_ids = test_batch

    # for i, wave_id in enumerate(test_wave_ids):
        # wave_filename = os.path.join(dir, wave_id + ".wav")
        # post_process_func(test_linear_spec[i], wave_filename, raise_power=1.0)
        # mel_wave_filename = os.path.join(dir, wave_id + "_mel.wav")
        # data_utils_normalize.post_process_mel_spec(test_mel_spec[i], mel_wave_filename, raise_power=1.0)

def plot_histogram(values, jpg_filename):
    n, bins, patches = plt.hist(values, bins=50)
    plt.xlabel("Length")
    plt.ylabel("Count")
    plt.title(jpg_filename)
    plt.grid(False)
    plt.savefig(jpg_filename)
    plt.close()

def plot_scatter(X, Y, jpg_filename):
    plt.scatter(X, Y)
    plt.xlabel("Encoder length")
    plt.ylabel("Decoder length")
    plt.title(jpg_filename)
    plt.grid(True)
    plt.savefig(jpg_filename)
    plt.close()

def plot_attention(data, jpg_filename, X_label=None, Y_label=None):
    '''
        Plot the attention model heatmap
        Args:
            data: attn_matrix with shape [ty, tx], cutted before 'PAD'
            X_label: list of size tx, encoder tags, unicode str
            Y_label: list of size ty, decoder tags, unicode str
    '''
    fig, ax = plt.subplots(figsize=(20, 8)) # set figure size
    heatmap = ax.pcolor(data, cmap=plt.cm.Blues, alpha=0.9)
    
    # Set axis labels
    if X_label != None and Y_label != None:
        X_label = [x_label for x_label in X_label]
        Y_label = [y_label for y_label in Y_label]
        
        xticks = range(0,len(X_label))
        ax.set_xticks(xticks, minor=False) # major ticks
        ax.set_xticklabels(X_label, minor=False, rotation=45)   # labels should be 'unicode'
        
        yticks = range(0,len(Y_label))
        ax.set_yticks(yticks, minor=False)
        ax.set_yticklabels(Y_label, minor=False)   # labels should be 'unicode'
        
        ax.grid(True)
    
    # Save Figure
    plt.title(u'Attention Heatmap')
    timestamp = int(time.time())
    print("Saving figures {}, data shape {}".format(jpg_filename, data.shape))
    fig.savefig(jpg_filename)   # save the figure to file
    plt.close(fig)    # close the figure


def get_train_arguments():
    def _str_to_bool(s):
        """Convert string to bool (in argparse context)."""
        if s.lower() not in ['true', 'false']:
            raise ValueError('Argument needs to be a '
                             'boolean, got {}'.format(s))
        return {'true': True, 'false': False}[s.lower()]

    parser = argparse.ArgumentParser(description='Tacotron training configuration')
    # args used for Philly
    parser.add_argument('--input-training-data-path', type=str, default="./data/", dest='datadir',
                        help='The input data directory, which should contain the config, info, and zip files. '
                        'if used in philly, this value will be replaced by [Philly][Official]Philly Storage Path Creator input.')
    parser.add_argument('-ngpu', type=str, default=None,
                        help='The minimum number of gpus will be used, e.g "0,1,2,3"'
                        'This will be set automatically when running on Philly')
    parser.add_argument('-gpus', type=str, default='0',
                        help='[Philly] the gpus will be used, e.g "0,1,2,3" (this is added for Philly, not used.')
    parser.add_argument('--log-dir', type=str, default=None, dest='logdir',
                        help='[Philly] This is required by Philly, not used')
    parser.add_argument('--output-model-path', type=str, default=None, dest='outputdir',
                        help='[Philly] Output model dir, not used.')
    parser.add_argument('-loadsnapshotdir', type=str, default=None,
                        help='[Philly] load model dir, not used')

    parser.add_argument('-configfile', type=str, required=True,
                        help='Json configuration file path, example: ./sample_data/config_small.json')
    parser.add_argument('-restore_from', type=str, default=None,
                        help='Checkpoint path from which the model is restored, example ./log/model.ckpt-32000.')
    parser.add_argument('-isphilly', type=_str_to_bool, default=False,
                        help='Indicate if this is run on Philly.')
    parser.add_argument('-learning_rates', type=str, default=None,
                        help='If you want to set learning rate manually, use this. Example: [[0,0.001],[10,0.0005]], '
                        'which means to set the initial learning rate to 0.001, and change it to 0.0005 when global step = 10k')
    parser.add_argument('-training_info_filename', type=str, default=None,
                        help='Training infomation filename, example: ./sample_data/info.eva.3s.utf8')
    parser.add_argument('-wave_zip_filename', type=str, default=None,
                        help='Wave zip filename.')
    parser.add_argument('-batch_size', type=int, default=None,
                        help='Batch size (32 recommended), this is the batch size for each GPUs, the actual batch size would be batch_size * ngpu.')
    parser.add_argument("-case_sensitive", type=_str_to_bool, default=None,
                        help='Indicate if encoder input is case sensitive.')
    parser.add_argument("-use_phone", type=_str_to_bool, default=None,
                        help='Indicate if encoder input uses phone sequence or character sequence')
    parser.add_argument("-use_stress", type=_str_to_bool, default=True,
                        help='Indicate if encoder input uses word stress or not')
    parser.add_argument("-use_f0", type=_str_to_bool, default=True,
                        help='For acoustic feature training, whether to use f0 and uv')
    parser.add_argument("-attention_type", type=str, default="monotonic",
                        help='Can be either soft or monotonic (default)')
    parser.add_argument("-stop_gradient", type=_str_to_bool, default=False,
                        help='Whether stop gradients flowing from postnet into seq2seq (default False).')
    parser.add_argument('-reduce_factor', type=int, default=None,
                        help='Reduce factor, if attention_type=soft, values from 2 to 5 are recommended.'
                        'if attention_type=monotonic, set to 2')
    parser.add_argument('-mel_spec_size', type=int, default=None,
                        help='The dimensionality of seq2seq output (will be multiplied by reduce_factor).')
    parser.add_argument('-linear_spec_size', type=int, default=None,
                        help='The dimensionality of postnet output (will be multiplied by reduce_factor).')
    parser.add_argument("-buckets", type=str, default=None,
                        help='buckets value to overwrite config file, example: -buckets=[[60,160]], '
                        'which means the maximum length of encoder/decoder is 60/160')
    parser.add_argument('-normalize', type=_str_to_bool, default=True,
                        help='Whether to normalize mel and linear spec. Please always set it to True (default)')
    parser.add_argument('-n_thread', type=int, default=10,
                        help='Number of threads used to load data and generate wave. (default 10)')
    parser.add_argument("-scheme", type=str, default="manual",
                        help='Learning rate updating scheme, options: manual (default), noam, renoam. '
                        'renorm is recommended, and should be used together with -warmup_steps')
    parser.add_argument('-warmup_steps', type=float, default=10000.0,
                        help='Number of warmup steps for noam|renoam scheme. 10000 should work for most cases.')
    parser.add_argument('-n_max_wave', type=int, default=None,
                        help='Maximum number of waves used to train model, if provided, a subset will be randomly sampled.')
    parser.add_argument("-other", type=str, default=None,
                        help='Other options, -other=type1:key1:value1:type2:key2:value2')
    return parser.parse_args()

def pad_specs(mel_spec, linear_spec, decoder_size, reduce_factor, pad_value=0.0):
    spec_pad_width = ((0, reduce_factor*decoder_size-len(mel_spec)), (0, 0)) # both for linear and mel spec
    mel_spec_padded = np.pad(mel_spec, pad_width=spec_pad_width, mode="constant", constant_values=pad_value) # [T', n_mels]
    linear_spec_padded = np.pad(linear_spec, pad_width=spec_pad_width, mode="constant", constant_values=pad_value) # [T', n_linears]
    return [mel_spec_padded, linear_spec_padded]

def load_data_from_zfile_wave(zfile, char2id, lines, buckets, reduce_factor, storage, get_spec=True):
    encoder_size, decoder_size = buckets[-1]
    for line in lines:
        wave_id, _, _, encoder_inputs, _ = line
        assert wave_id not in storage

        wave_filename = "{}.wav".format(wave_id)
        fin = zfile.open(wave_filename, 'r')
        tmp = io.BytesIO(fin.read())
        audio, sr = soundfile.read(tmp)

        char_ids = [char2id[char] for char in encoder_inputs] + [0]*(encoder_size-len(encoder_inputs))
        if get_spec:
            mel_spec, linear_spec = data_utils_normalize.get_mel_and_linear_spec(audio)
            mel_spec_padded, linear_spec_padded = pad_specs(mel_spec, linear_spec, decoder_size, reduce_factor)
            storage[wave_id] = [char_ids, mel_spec_padded, linear_spec_padded]
        else:
            storage[wave_id] = [char_ids, audio]

def load_data_from_zfile(zfile, char2id, lines, buckets, reduce_factor, storage):
    for line in lines:
        wave_id, encoder_size, decoder_size, encoder_inputs, bucket_id = line

        pickle_filename = "{}.pkl".format(wave_id)
        fin = zfile.open(pickle_filename, 'r')
        if sys.version_info >= (3, 0):
            wave_raw_data = pickle.load(fin, encoding="latin1") # This solves UnicodeDecodeError and works with python 3.x, but not with python 2.x
        else:
            wave_raw_data = pickle.load(fin)

        assert len(wave_raw_data) == 4
        wave_id, text, mel_spec, linear_spec = wave_raw_data
        mel_spec, linear_spec = data_utils.pre_process_mel_and_linear_spec(mel_spec, linear_spec)

        encoder_size, decoder_size = buckets[bucket_id]
        char_ids = [char2id[char] for char in encoder_inputs] + [0]*(encoder_size-len(encoder_inputs))

        spec_pad_width = ((0, reduce_factor*decoder_size-len(mel_spec)), (0, 0)) # both for linear and mel spec
        mel_spec_padded = np.pad(mel_spec, pad_width=spec_pad_width, mode="constant", constant_values=np.log(1e-10)) # [T', n_mels]
        linear_spec_padded = np.pad(linear_spec, pad_width=spec_pad_width, mode="constant", constant_values=np.log(1e-10)) # [T', n_linears]

        assert wave_id not in storage
        storage[wave_id] = [char_ids, mel_spec_padded, linear_spec_padded]


def load_data_for_queue_reader(config):
    print("Loading data from Philly HDFS...")
    start_time = time.time()
    batch_size = config["batch_size"]
    buckets = config["buckets"]
    assert len(buckets) == 1
    case_sensitive = config["case_sensitive"]
    use_phone = config["use_phone"]
    use_stress = config["use_stress"]
    reduce_factor = config["reduce_factor"]
    train_info_filename = config["training_info_filename"]
    data_directory = config["data_directory"]

    n_thread = config["n_thread"]
    normalize = config["normalize"]

    use_stop_token = config["use_stop_token"]

    char_counter = Counter() # char|phone counter
    lines = []

    bucket_wave_ids = [[] for _ in buckets] # store wave ids for each bucket
    n_sentence = 0
    storage = {}
    waveid2nframes = {}
    for line in codecs.open(train_info_filename, "r", "utf-8"):
        if case_sensitive == False:
            line = line.lower()
        items = line.strip().split("\t")

        wave_id, n_frames = items[0], int(items[2])
        # decoder_size = (n_frames+1)//reduce_factor
        decoder_size = n_frames
        if n_frames * hp.frame_shift < hp.min_decoder_time: # minimum decoder time
            continue

        if use_phone == False: # input phone sequence
            text = items[3].strip()
            encoder_inputs = list(text)
        else: # input character sequence
            phones = items[5]
            phones = data_utils.clean_phones(phones, use_stress=use_stress)
            encoder_inputs = phones
        encoder_size = len(encoder_inputs)
        char_counter.update(encoder_inputs)

        bucket_id = get_bucket_id(encoder_size, decoder_size, buckets)
        if bucket_id < 0 or bucket_id >= len(buckets):
            continue
        waveid2nframes[wave_id] = n_frames
        n_sentence += 1
        bucket_wave_ids[bucket_id].append(wave_id)
        lines.append((wave_id, encoder_size, decoder_size, encoder_inputs, bucket_id))

    encoder_size, decoder_size = buckets[-1]
    char2id = {"padding": 0}
    chars = sorted(char_counter.keys())
    for i, c in enumerate(chars):
        char2id[c] = i + 1

    assert n_thread > 0
    threads = []
    zip_filename = os.path.join(data_directory, config["wave_zip_filename"])
    zfile = zipfile.ZipFile(zip_filename)

    if normalize:
        #target = load_data_from_zfile_wave
        thread_lines = lines[0::n_thread]
        load_data_from_zfile_wave(zfile, char2id, thread_lines, buckets, reduce_factor, storage, False)
    else:
        #target = load_data_from_zfile
        thread_lines = lines[i::n_thread]
        load_data_from_zfile(zfile, char2id, thread_lines, buckets, reduce_factor, storage, False)
    # for i in range(n_thread):
    #    thread_lines = lines[i::n_thread]
    #    threads.append(threading.Thread(target=target, args=(zfile, char2id, thread_lines, buckets, reduce_factor, storage, False)))
    #for t in threads:
    #    t.start()
    #    t.join()
    zfile.close()
    
    # choose a batch for testing, B/2 from training set, B/2 from testing set
    if config["test_wave_ids"]: # use test batch of previous run
        test_wave_ids = config["test_wave_ids"]
    else: # randomly select test batch
        wave_ids = list(storage.keys())
        # test_wave_ids = list(np.random.choice(wave_ids, batch_size, replace=False))
        test_wave_ids = list(np.random.choice(wave_ids, batch_size, replace=False))
        np.random.shuffle(test_wave_ids)

    batch_char_ids = []
    batch_mel_specs = []
    #batch_linear_specs = []
    batch_stop_tokens = []

    for i, wave_id in enumerate(test_wave_ids):
        char_ids, audio = storage[wave_id]
        mel_spec, linear_spec = data_utils_normalize.get_mel_and_linear_spec(audio)

        n_mel, _ = mel_spec.shape

        # save extracted mels
        #np.savetxt('E:/projects/tacotron/GoldenBridge/private/tacotron2/testtraining/miaohong_recording/mel_base.txt', mel_spec)

        #mel_spec, linear_spec = data_utils_normalize.get_mel_spectrogram(audio)
        mel_spec_padded, linear_spec_padded = pad_specs(mel_spec, linear_spec, decoder_size, reduce_factor)
        batch_char_ids.append(char_ids)
        batch_mel_specs.append(mel_spec_padded)
        #batch_linear_specs.append(linear_spec_padded)


        if use_stop_token is True:
            stop_token = []
            for i in range(decoder_size):
                if i >= n_mel - 1:
                    stop_token.append([1])
                else:
                    stop_token.append([0])

            batch_stop_tokens.append(stop_token)

        if i >= batch_size/2:
            del storage[wave_id] # delete from training set

    # random sample n_max_wave waves
    n_max_wave = config["n_max_wave"]
    if n_max_wave is not None and n_max_wave < len(storage):
        training_wave_ids = np.random.choice(list(storage.keys()), n_max_wave, replace=False)
        for wave_id in list(storage.keys()):
            if wave_id not in training_wave_ids:
                del storage[wave_id]

    bucket_id = 0
    if use_stop_token is True:
        test_batch = [np.stack(batch_char_ids, axis=0),
                  np.stack(batch_mel_specs, axis=0),
                  #np.stack(batch_linear_specs, axis=0),
                  bucket_id,
                  #test_wave_ids]
                  test_wave_ids,
                  batch_stop_tokens]
    else:
        test_batch = [np.stack(batch_char_ids, axis=0),
                  np.stack(batch_mel_specs, axis=0),
                  #np.stack(batch_linear_specs, axis=0),
                  bucket_id,
                  #test_wave_ids]
                  test_wave_ids]

    n_total_frames = sum([waveid2nframes[wave_id] for wave_id in list(storage.keys())])
    print("wave count = {}".format(len(storage)))
    print("data total hours: {}".format(n_total_frames*hp.frame_shift/3600.0))
    print("char_vocab_size = {}".format(len(char2id)))
    print("char2id:", char2id)
    print("Loading data done, time used: {} minutes".format((time.time()-start_time)/60.0))

    config["char2id"] = char2id
    config["test_wave_ids"] = test_wave_ids
    config["char_vocab_size"] = len(char2id)
    return storage, [list(storage.keys())], test_batch


def load_data(config):
    print("Loading data from Philly HDFS...")
    start_time = time.time()
    batch_size = config["batch_size"]
    buckets = config["buckets"]
    assert len(buckets) == 1
    case_sensitive = config["case_sensitive"]
    use_phone = config["use_phone"]
    use_stress = config["use_stress"]
    reduce_factor = config["reduce_factor"]
    train_info_filename = config["training_info_filename"]
    data_directory = config["data_directory"]

    n_thread = config["n_thread"]
    normalize = config["normalize"]


    # char2id = hp.char2id
    char_counter = Counter() # char|phone counter
    lines = []

    bucket_wave_ids = [[] for _ in buckets] # store wave ids for each bucket
    n_sentence = 0
    storage = {}
    waveid2nframes = {}
    for line in codecs.open(train_info_filename, "r", "utf-8"):
        if case_sensitive == False:
            line = line.lower()
        items = line.strip().split("\t")

        wave_id, n_frames = items[0], int(items[2])
        decoder_size = (n_frames+1)//reduce_factor
        if n_frames * hp.frame_shift < hp.min_decoder_time: # minimum decoder time
            continue

        if use_phone == False: # input phone sequence
            text = items[3].strip()
            encoder_inputs = list(text)
        else: # input character sequence
            phones = items[5]
            phones = data_utils.clean_phones(phones, use_stress=use_stress)
            encoder_inputs = phones
        encoder_size = len(encoder_inputs)
        char_counter.update(encoder_inputs)

        bucket_id = get_bucket_id(encoder_size, decoder_size, buckets)
        if bucket_id < 0 or bucket_id >= len(buckets):
            continue
        waveid2nframes[wave_id] = n_frames
        n_sentence += 1
        bucket_wave_ids[bucket_id].append(wave_id)
        lines.append((wave_id, encoder_size, decoder_size, encoder_inputs, bucket_id))

    char2id = {"padding": 0}
    chars = sorted(char_counter.keys())
    for i, c in enumerate(chars):
        char2id[c] = i + 1

    assert n_thread > 0
    threads = []
    zip_filename = os.path.join(data_directory, config["wave_zip_filename"])
    zfile = zipfile.ZipFile(zip_filename)

    if normalize:
        target = load_data_from_zfile_wave
    else:
        target = load_data_from_zfile
    for i in range(n_thread):
        thread_lines = lines[i::n_thread]
        threads.append(threading.Thread(target=target, args=(zfile, char2id, thread_lines, buckets, reduce_factor, storage)))
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    zfile.close()
    
    # choose a batch for testing, B/2 from training set, B/2 from testing set
    if config["test_wave_ids"]: # use test batch of previous run
        test_wave_ids = config["test_wave_ids"]
    else: # randomly select test batch
        wave_ids = list(storage.keys())
        test_wave_ids = list(np.random.choice(wave_ids, batch_size, replace=False))
        np.random.shuffle(test_wave_ids)

    batch_char_ids = []
    batch_mel_specs = []
    batch_linear_specs = []
    for i, wave_id in enumerate(test_wave_ids):
        char_ids, mel_spec, linear_spec = storage[wave_id]
        batch_char_ids.append(char_ids)
        batch_mel_specs.append(mel_spec)
        batch_linear_specs.append(linear_spec)
        if i >= batch_size/2:
            del storage[wave_id] # delete from training set

    # random sample n_max_wave waves
    n_max_wave = config["n_max_wave"]
    if n_max_wave is not None and n_max_wave < len(storage):
        training_wave_ids = np.random.choice(list(storage.keys()), n_max_wave, replace=False)
        for wave_id in list(storage.keys()):
            if wave_id not in training_wave_ids:
                del storage[wave_id]

    bucket_id = 0
    test_batch = [np.stack(batch_char_ids, axis=0),
                  np.stack(batch_mel_specs, axis=0),
                  np.stack(batch_linear_specs, axis=0),
                  bucket_id,
                  test_wave_ids]

    n_total_frames = sum([waveid2nframes[wave_id] for wave_id in list(storage.keys())])
    print("wave count = {}".format(len(storage)))
    print("data total hours: {}".format(n_total_frames*hp.frame_shift/3600.0))
    print("char_vocab_size = {}".format(len(char2id)))
    print("char2id:", char2id)
    print("Loading data done, time used: {} minutes".format((time.time()-start_time)/60.0))

    config["char2id"] = char2id
    config["test_wave_ids"] = test_wave_ids
    config["char_vocab_size"] = len(char2id)
    return storage, [list(storage.keys())], test_batch

def validate_config(config):
    # buckets
    return True

def load_config(filename):
    with codecs.open(filename, 'r', 'utf-8') as fin_json:
        config = json.load(fin_json)
    assert validate_config(config)
    print("config loaded from json")
    # pprint(config)
    return config

def get_bucket_id(input_length, output_length, buckets):
    for i in range(len(buckets)):
        if input_length < buckets[i][0] and output_length < buckets[i][1]:
            return i
    return -1
