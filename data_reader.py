# coding=utf-8

from __future__ import print_function

import codecs
import copy
import data_utils_normalize
import numpy as np
import pickle
import os
import random
import sys
import threading
import time
import utils
import zipfile

from collections import Counter
from hyper_params import HyperParams as hp

if sys.version_info >= (3, 0):
    import queue as Queue
else:
    import Queue

class QueueDataReader(object):
    def __init__(self, bucket_wave_ids, cached_storage, config, coord):
        self.bucket_wave_ids = bucket_wave_ids
        self.cached_storage = cached_storage # dictionary: wave_id --> [char_ids_padded, audio]
        self.config = config
        self.batch_size = config["batch_size"]
        self.n_gpu = config["n_gpu"]
        self.reduce_factor = config["reduce_factor"]
        self.buckets = config["buckets"]
        self.encoder_size, self.decoder_size = self.buckets[-1]

        self.n_epoch = config["n_epoch"]
        self.cur_epoch = 0
        self.coord = coord

        self.wave_ids = []
        for bucket_id, ids in enumerate(self.bucket_wave_ids):
            self.wave_ids.extend(ids)

        self.id_queue = Queue.Queue(maxsize=len(self.wave_ids)*3) # thread safe
        self.data_queue = Queue.Queue(maxsize=self.batch_size*self.n_gpu*3) # TODO: maxsize

        self._lock = threading.Lock()
        self.n_running_id_enqueue = 0
        self.n_running_data_enqueue = 0

        self.use_stop_token = config["use_stop_token"]

    def get_wave_data(self, wave_id):
        char_ids, audio = self.cached_storage[wave_id]
        mel_spec, linear_spec = data_utils_normalize.get_mel_and_linear_spec(audio)

        n_mel, _ = mel_spec.shape
        if  self.use_stop_token is True:
            stop_token_indexes = []
            for i in range(self.decoder_size):
                if i >= n_mel - 1:
                    stop_token_indexes.append([1])
                else:
                    stop_token_indexes.append([0])

        #np.save(
        #    'E:/projects/tacotron/GoldenBridge/private/tacotron2/testtraining/miaohong_recording/mel_base', mel_spec)

        #np.save(
        #    'E:/projects/tacotron/GoldenBridge/private/tacotron2/testtraining/miaohong_recording/stop', stop_token_indexes)

        mel_spec_padded, linear_spec_padded = utils.pad_specs(mel_spec, linear_spec, self.decoder_size, self.reduce_factor)
        #return [wave_id, char_ids, mel_spec_padded, linear_spec_padded]
        if self.use_stop_token is True:
            return [wave_id, char_ids, mel_spec_padded, stop_token_indexes]
        else:
            return [wave_id, char_ids, mel_spec_padded]

    def start(self):
        def id_enqueue_func(i):
            print("Starting id enqueue thread:", i)
            try:
                while self.cur_epoch < self.n_epoch:
                    if self.coord.should_stop():
                        break
                    wave_ids = copy.deepcopy(self.wave_ids)
                    random.shuffle(wave_ids)
                    for wave_id in wave_ids:
                        self.id_queue.put(wave_id)
                    with self._lock:
                        self.cur_epoch += 1
                        print("QueueDataReader cur_epoch:", self.cur_epoch)
            except Exception as e:
                self.coord.request_stop()
                print("Exception happened in id enqueue thread:", i, "message:", e)

            with self._lock:
                self.n_running_id_enqueue -= 1
            print("Ending id enqueue thread:", i)

        def data_enqueue_func(i):
            print("Starting data enqueue thread:", i)
            try:
                while True:
                    if self.coord.should_stop():
                        break
                    with self._lock:
                        if not self.id_queue.empty():
                            wave_id = self.id_queue.get()
                        elif self.n_running_id_enqueue > 0:
                            continue
                        else:
                            break
                    wave_data = self.get_wave_data(wave_id)
                    self.data_queue.put(wave_data)

                    self.id_queue.task_done()

            except Exception as e:
                self.coord.request_stop()
                print("Exception happened in data enqueue thread:", i, "message:", e)
            
            with self._lock:
                self.n_running_data_enqueue -= 1
            print("Ending data enqueue thread:", i)

        self.n_running_id_enqueue = 1
        t = threading.Thread(target=id_enqueue_func, args=(0,))
        self.coord.register_thread(t)
        t.setDaemon(True)
        t.start()

        self.n_running_data_enqueue = self.config["n_thread"]
        for i in range(self.config["n_thread"]):
            t = threading.Thread(target=data_enqueue_func, args=(i,))
            self.coord.register_thread(t)
            t.setDaemon(True)
            t.start()

    def next_batch(self):
        wave_ids = []
        batch_char_ids = []
        batch_mel_specs = []
        #batch_linear_specs = []
        batch_stop_tokens = []
        while True:
            if len(wave_ids) == self.batch_size * self.n_gpu:
                break
            if self.coord.should_stop():
                break
            if not self.data_queue.empty(): # Assumption: only one consumer of self.data_queue
                #wave_id, char_ids, mel_spec, linear_spec = self.data_queue.get() # won't be blocked
                if  self.use_stop_token is True:
                    wave_id, char_ids, mel_spec, stop_tokens = self.data_queue.get()  # won't be blocked
                else:
                    wave_id, char_ids, mel_spec = self.data_queue.get()  # won't be blocked
                wave_ids.append(wave_ids)
                batch_char_ids.append(char_ids)
                batch_mel_specs.append(mel_spec)
                #batch_linear_specs.append(linear_spec)
                if self.use_stop_token is True:
                    batch_stop_tokens.append(stop_tokens)
                self.data_queue.task_done()
            else:
                with self._lock:
                    if self.n_running_data_enqueue == 0:
                        break

        if len(wave_ids) < self.batch_size * self.n_gpu:
            return None
        else:
            if self.use_stop_token is True:
                return [np.stack(batch_char_ids, axis=0),
                    np.stack(batch_mel_specs, axis=0),
                    #np.stack(batch_linear_specs, axis=0),
                    0,
                    #wave_ids]
                    wave_ids,
                    np.stack(batch_stop_tokens, axis=0)]
            else:
                return [np.stack(batch_char_ids, axis=0),
                    np.stack(batch_mel_specs, axis=0),
                    #np.stack(batch_linear_specs, axis=0),
                    0,
                    wave_ids]

class CachedSharedDataReader(object):
    def __init__(self, bucket_wave_ids, cached_storage, config, n_epoch):
        self.bucket_wave_ids = bucket_wave_ids
        self.cached_storage = cached_storage # dictionary: wave_id --> [char_ids, mel_spec_padded, linear_spec_padded]
        self.config = config
        self.batch_size = config["batch_size"]
        self.reduce_factor = config["reduce_factor"]
        self.buckets = config["buckets"]

        self.n_epoch = n_epoch
        self.cur_epoch = 0
        self.cur_batch_index = 0
        self.lock = threading.Lock()
        self._reset_and_shuffle()

    def _reset_and_shuffle(self):
        """Callers to ensure no synchronized access to this function"""
        self.cur_batch_index = 0
        self.batches = [] # only wave ids are saved in batches

        # shuffle wave ids in buckets, get new batches
        for bucket_id, wave_ids in enumerate(self.bucket_wave_ids):
            random.shuffle(wave_ids)
            bucket_batches = [(wave_ids[i:i+self.batch_size], bucket_id) for i in range(0, len(wave_ids), self.batch_size)]
            self.batches.extend(bucket_batches)

        # shuffle batches
        random.shuffle(self.batches)

    # def has_next(self):
        # return self.cur_batch_index < len(self.batches)

    def next_batch(self):
        """Returns next batch if has next, returns None otherwise"""
        self.lock.acquire()
        done = False
        try:
            if self.cur_batch_index >= len(self.batches):
                if self.cur_epoch + 1 < self.n_epoch:
                    self._reset_and_shuffle()
                    self.cur_epoch += 1
                else:
                    done = True

            if not done:                
                wave_ids, bucket_id = self.batches[self.cur_batch_index]
                self.cur_batch_index += 1
        except Exception as e:
            print("Exception in shared data reader's next_batch()", e)
        finally:
            self.lock.release()

        if done:
            return None

        batch_char_ids = []
        batch_mel_specs = []
        batch_linear_specs = []
        for wave_id in wave_ids:
            char_ids, mel_spec, linear_spec = self.cached_storage[wave_id]
            batch_char_ids.append(char_ids)
            batch_mel_specs.append(mel_spec)
            batch_linear_specs.append(linear_spec)

        return [np.stack(batch_char_ids, axis=0),
                np.stack(batch_mel_specs, axis=0),
                np.stack(batch_linear_specs, axis=0),
                bucket_id,
                wave_ids]

class CachedDataReader(object):
    def __init__(self, bucket_wave_ids, cached_storage, config, batch_size):
        self.bucket_wave_ids = bucket_wave_ids
        self.cached_storage = cached_storage # dictionary: wave_id --> [char_ids, mel_spec_padded, linear_spec_padded]
        self.config = config
        self.batch_size = batch_size
        self.reduce_factor = config["reduce_factor"]
        self.buckets = config["buckets"]

    def reset_and_shuffle(self):
        self.cur_batch_index = 0
        self.batches = [] # only wave ids are saved in batches

        # shuffle wave ids in buckets, get new batches
        for bucket_id, wave_ids in enumerate(self.bucket_wave_ids):
            random.shuffle(wave_ids)
            for i in range(0, len(wave_ids), self.batch_size):
                if i + self.batch_size <= len(wave_ids): # insure batch size
                    self.batches.append((wave_ids[i:i+self.batch_size], bucket_id))

        # shuffle batches
        random.shuffle(self.batches)

    def has_next(self):
        return self.cur_batch_index < len(self.batches)

    def next_batch(self):
        assert self.has_next()
        wave_ids, bucket_id = self.batches[self.cur_batch_index]
        self.cur_batch_index += 1

        batch_char_ids = []
        batch_mel_specs = []
        batch_linear_specs = []
        for wave_id in wave_ids:
            char_ids, mel_spec, linear_spec = self.cached_storage[wave_id]
            batch_char_ids.append(char_ids)
            batch_mel_specs.append(mel_spec)
            batch_linear_specs.append(linear_spec)

        return [np.stack(batch_char_ids, axis=0),
                np.stack(batch_mel_specs, axis=0),
                np.stack(batch_linear_specs, axis=0),
                bucket_id,
                wave_ids]

class DataReader(object):
    def __init__(self, config):
        self.config = config
        self.batch_size = config["batch_size"]
        self.buckets = config["buckets"]
        self.reduce_factor = config["reduce_factor"]

        self.bucket_wave_ids = [[] for _ in self.buckets] # store wave ids for each bucket
        self.char2id = {"padding_char": 0} # padding char's id is always fixed to zero
        train_info_filename = config["training_info_filename"]
        self.data_directory = config["data_directory"]
        wave_count = 0
        char_counter = Counter()
        n_total_frames = 0
        for line in codecs.open(train_info_filename, "r", "utf-8"):
            items = line.split("\t")
            assert len(items) == 4
            wave_id, n_chars, n_frames, text = items[0], int(items[1]), int(items[2]), items[3]
            bucket_id = utils.get_bucket_id(n_chars, n_frames//self.reduce_factor, self.buckets)
            if bucket_id < 0 or bucket_id >= len(self.buckets):
                continue
            wave_count += 1
            n_total_frames += n_frames
            char_counter.update(text)
            self.bucket_wave_ids[bucket_id].append(wave_id)
        print("bucket_wave_ids", self.bucket_wave_ids)
        print("wave_count = {}".format(wave_count))
        print("len(char_count) = {}".format(len(char_counter)))
        print("total hours: {}".format(n_total_frames/80.0/3600.0))

        sorted_chars = sorted(char_counter.keys())
        for i, char in enumerate(sorted_chars):
            self.char2id[char] = i + 1


    def reset_and_shuffle(self):
        self.cur_batch_index = 0
        self.batches = [] # only wave ids are saved in batches

        # shuffle wave ids in buckets, get new batches
        for bucket_id, wave_ids in enumerate(self.bucket_wave_ids):
            random.shuffle(wave_ids)
            bucket_batches = [(wave_ids[i:i+self.batch_size], bucket_id) for i in range(0, len(wave_ids), self.batch_size)]
            self.batches.extend(bucket_batches)

        # shuffle batches
        random.shuffle(self.batches)

    def has_next(self):
        return self.cur_batch_index < len(self.batches)

    def next_batch(self):
        assert self.has_next()
        wave_ids, bucket_id = self.batches[self.cur_batch_index]
        encoder_size, decoder_size = self.buckets[bucket_id]
        self.cur_batch_index += 1

        batch_char_ids = []
        batch_mel_specs = []
        batch_linear_specs = []
        for wave_id in wave_ids:
            pickle_filename = self.data_directory + wave_id + ".pkl"
            data = pickle.load(open(pickle_filename, 'rb'))
            assert len(data) == 4
            wave_id, text, linear_spec, mel_spec = data

            char_ids = [self.char2id[char] for char in text] + [0]*(encoder_size-len(text))
            batch_char_ids.append(char_ids)

            pad_width = ((0, self.reduce_factor*decoder_size-len(mel_spec)), (0, 0))
            batch_mel_specs.append(np.pad(mel_spec, pad_width=pad_width, mode="constant", constant_values=0.0))
            batch_linear_specs.append(np.pad(linear_spec, pad_width=pad_width, mode="constant", constant_values=0.0))

        return [np.stack(batch_char_ids, axis=0),
                np.stack(batch_mel_specs, axis=0),
                np.stack(batch_linear_specs, axis=0),
                bucket_id,
                wave_ids]

class FakeDataReader(object):
    def __init__(self, config):
        self.config = config

        batch_size = config["batch_size"]
        self.reduce_factor = config["reduce_factor"]
        self.mel_spec_size = config["mel_spec_size"]
        self.linear_spec_size = config["linear_spec_size"]

        self.cur_batch_index = 0
        self.batches = []
        np.random.seed(0)
        for i in range(1):
            for bucket_id, (encoder_size, decoder_size) in enumerate(config["buckets"]):
                self.batches.append(self._generate_batch(batch_size, encoder_size, decoder_size) + [bucket_id, []])

    def reset_and_shuffle(self):
        self.cur_batch_index = 0

    def has_next(self):
        return self.cur_batch_index < len(self.batches)

    def next_batch(self):
        assert self.has_next()
        batch_data = self.batches[self.cur_batch_index]
        self.cur_batch_index += 1
        return batch_data

    def _generate_batch(self, batch_size, encoder_size, decoder_size):
        char_ids = np.random.randint(0, self.config["char_vocab_size"], (batch_size, encoder_size))
        mel_spec = np.random.random([batch_size, self.reduce_factor*decoder_size, self.mel_spec_size]) # random values between [0, 1]
        linear_spec = np.random.random([batch_size, self.reduce_factor*decoder_size, self.linear_spec_size])
        return [char_ids, mel_spec, linear_spec]
