# coding=utf-8

from __future__ import print_function

import argparse
import codecs
import data_reader
import data_utils
import data_utils_normalize
import json
import librosa
import numpy as np
import os
import pickle
import shutil
import tacotron2
import sys
import tensorflow as tf
import threading
import time
import utils

from datetime import datetime
from hyper_params import HyperParams as hp
from pprint import pprint

def get_arguments():
    def _str_to_bool(s):
        return s.lower() == "true"

    parser = argparse.ArgumentParser(description="Tacotron evaluation script configuration.")
    
    parser.add_argument('-restore_from', type=str, default="./log/models/1501012364013_2271/model.ckpt-32000", # eva character
    # parser.add_argument('-restore_from', type=str, default="./log/models/1501012364013_11184/model.ckpt-59000", # rinna, phone without stress
                        help='Model path (./model.ckpt-12000), config.json should be found in the same directory.')
    parser.add_argument('-text_file', type=str, default="./data/eva60.utf8",
                        help='Text file for generating wave files, utf8 encoded.')
    parser.add_argument("-buckets", type=str, default='[[200,400]]',
                        help='buckets value to overwrite config file, example: -buckets=[[200,400]], '
                        'which means the maximum length of encoder/decoder is 200/400')
    parser.add_argument('-output_dir', type=str, default=None,
                        help='Directory to save generated waves, if not set, waves will be saved in the same directory as restore_from')
    parser.add_argument('-output_attention', type=_str_to_bool, default=False,
                        help='Whether output attention heatmap.')
    parser.add_argument('-trim_tailing_silence', type=_str_to_bool, default=True,
                        help='Whether delete tailing silence of generated wave (default True).')
    parser.add_argument('-n_thread', type=int, default=32,
                        help='Number of threads used to run Griffin-Lim in a batch (default 10).')

    return parser.parse_args()

def test_batch_generator(text_file, config):
    batch_size = config["batch_size"]
    encoder_size, decoder_size = config["buckets"][-1]
    case_sensitive = config["case_sensitive"]
    char2id = config["char2id"]
    use_phone = config["use_phone"]
    use_stress = config.get("use_stress", False)
    mel_spec_size = config["mel_spec_size"]

    if use_phone:
        oov_id = char2id["-"] # syllable separator
    else:
        oov_id = char2id[' ']

    wave_ids = []
    char_ids = []
    with codecs.open(text_file, 'r', 'utf-8') as fin:
        for line in fin:
            items = line.strip().split("\t")
            assert len(items) == 2
            wave_id = items[0]
            text = items[1].strip()
            if not case_sensitive or use_phone:
                text = text.lower()

            if use_phone:
                text = data_utils.clean_phones(text, use_stress=use_stress)
            cids = [char2id[c] if c in char2id else oov_id for c in text]
            cids += [0] * (encoder_size - len(cids))

            wave_ids.append(wave_id)
            char_ids.append(cids)

    if len(char_ids) % batch_size != 0:
        wave_id = wave_ids[-1]
        cids = char_ids[-1]
        n = batch_size - len(char_ids) % batch_size
        wave_ids += [wave_id] * n
        char_ids += [cids] * n

    n_frames = decoder_size
    for i in range(0, len(char_ids), batch_size):
        end = min(i+batch_size, len(char_ids))
        B = end - i

        yield [wave_ids[i:end],
               np.stack(char_ids[i:end], axis=0),
               np.zeros((B, n_frames, mel_spec_size), dtype=float)]

def generate():
    args = get_arguments()

    restore_from = args.restore_from
    log_dir = os.path.dirname(restore_from)
    if args.output_dir is None:
        args.output_dir = restore_from + "." +  os.path.basename(args.text_file) + ".wave"
    utils.ensure_dir(args.output_dir)
    shutil.copy2(args.text_file, args.output_dir)

    config_filename = os.path.join(log_dir, "config.json")
    with codecs.open(config_filename, 'r', 'utf-8') as fin:
        config = json.load(fin)
        print(config)

    config["buckets"] = eval(args.buckets)
    if "mel_spec_size" not in config:
        config["mel_spec_size"] = hp.mel_spec_size
    
    batches = test_batch_generator(args.text_file, config)
    for i, batch in enumerate(batches):
        print("batch", i, len(batch[0]))

    # to-to:
    # change the post process spec
    batch_size = config["batch_size"]
    if config["normalize"]:
        post_process_func = data_utils_normalize.post_process_mel_spec
    else:
        post_process_func = data_utils.post_process_linear_spec

    with tf.Graph().as_default():
        model = tacotron2.Tacotron2(1, config, is_training=False, reuse=False)
        print("model built")

        session = tf.Session(config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True))
        session.run(tf.global_variables_initializer())
        print("global variables initialized")

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=20, keep_checkpoint_every_n_hours=6)
        utils.restore_model(session, saver, args.restore_from)
        print("model restored")

        start_time = time.time()
        batch_cnt = 0.0
        batches = test_batch_generator(args.text_file, config)
        for batch in batches:
            batch_start_time = time.time()
            wave_ids, char_ids, mel_spec = batch

            predicted_mel_spec, attentions = model.step(
                session, char_ids, mel_spec, 0)

            print("batch_cnt={} time={:.1f}".format(int(batch_cnt), time.time()-batch_start_time))
            batch_cnt += 1
            
            def generate_and_save(indexes):    
                for i in indexes:
                    wave_id = wave_ids[i]
                    wave_filename = os.path.join(args.output_dir, wave_id + ".wav")
                    # post_process_func(predicted_linear_spec[i], wave_filename, raise_power=1.0, trim_tailing_silence=args.trim_tailing_silence)
                    wave_filename = os.path.join(args.output_dir, wave_id + ".raise1.3.wav")
                    post_process_func(session[i], wave_filename, raise_power=1.3, trim_tailing_silence=args.trim_tailing_silence)
                    wave_filename = os.path.join(args.output_dir, wave_id + ".raise1.5.wav")
                    post_process_func(session[i], wave_filename, raise_power=1.5, trim_tailing_silence=args.trim_tailing_silence)
                    
                    if False:
                        mel_filename = os.path.join(args.output_dir, wave_id + ".mel.predicted.npy")
                        np.save(mel_filename, predicted_mel_spec[i])

                    attn_sum = attentions[i][:, -1].sum()
                    if attn_sum != 0:
                        print("WARNING: bucket seems to be not long enough for wave_id= ", wave_id, "attn_sum =", attn_sum)
                    args.output_attention = True
                    if args.output_attention:
                        jpg_filename = os.path.join(args.output_dir, wave_id + ".jpg")
                        utils.plot_attention(attentions[i], jpg_filename, X_label="decoder", Y_label="encoder")

            threads = []
            for i in range(args.n_thread):
                indexes = range(i, len(wave_ids), args.n_thread)
                t = threading.Thread(target=generate_and_save, args=(indexes,)) # don't miss ,
                threads.append(t)
                t.start()
            for t in threads:
                t.join()
        duration = time.time() - start_time
        print("{} batches {} generated in {} seconds, speed: {:.2f} seconds/batch".format(batch_cnt, batch_size, duration, duration/batch_cnt))


if __name__ == "__main__":
    generate()
