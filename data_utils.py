# coding=utf-8

from __future__ import division
from __future__ import print_function

import codecs
import copy
import io
import numpy as np
import librosa
import os
import pickle
import re
import soundfile
import time
import utils
import zipfile
import argparse
import shutil
import json

from collections import Counter
from pprint import pprint
from scipy.io.wavfile import write
import xml.etree.ElementTree as ET
from hyper_params import HyperParams as hp

def post_process_linear_spec(linear_spec, wave_filename, raise_power=1.0, trim_tailing_silence=True):
    '''
    Args:
        linear_spec: predicted linear spec by model, shape [T, linear_spec_size]
    '''
    linear_spec = np.power(np.e, linear_spec) - 1e-10
    linear_spec = np.clip(linear_spec, 0.0, None)
    linear_spec = linear_spec ** raise_power
    spectrogram2wav(linear_spec, wave_filename=wave_filename, trim_tailing_silence=trim_tailing_silence)
    return linear_spec

def pre_process_mel_and_linear_spec(mel_spec, linear_spec):
    '''
    Args:
        mel_spec: mel-scale spectrogram, shape [T, mel_spec_size]
        linear_spec: magnitude spectrogram, shape [T, linear_spec_size]
    Returns:
        processed mel_spec, linear_spec
    '''
    mel_spec = np.log(mel_spec + 1e-10)
    linear_spec = np.log(linear_spec + 1e-10)
    return mel_spec, linear_spec

def get_mel_and_linear_spec(wave_filename, trim_silence=False, pre_emphasis=None): 
    '''Extracts melspectrogram and log magnitude from given `sound_file`.
    Args:
        wave_filename: A string. Full path of a sound file.
        trim_silence: trim leading silence
        pre_emphasis: pre emphasis audio

    Returns:
        mel: shape [T, mel_spec_size]
        Transposed magnitude: A 2d array.Has shape of (T, 1+hp.n_fft//2)
    '''
    # Loading sound file
    y, sr = librosa.load(wave_filename, sr=None) # or set sr to hp.sr.

    if trim_silence:
        yt, index = librosa.effects.trim(y)
        y = y[index[0]:]

    if pre_emphasis:
        y = np.append(y[0], y[1:] - pre_emphasis*y[:-1])
    
    # stft. D: (1+n_fft//2, T)
    D = librosa.stft(y=y,
                     n_fft=hp.n_fft, 
                     hop_length=hp.hop_length, 
                     win_length=hp.win_length)
    
    # magnitude spectrogram
    magnitude = np.abs(D) #(1+n_fft//2, T)
    
    # power spectrogram
    power = magnitude**2 #(1+n_fft//2, T) 
    
    # mel spectrogram
    S = librosa.feature.melspectrogram(S=power, n_mels=hp.mel_spec_size) #(n_mels, T)

    mel = np.transpose(S.astype(np.float32)) # (T, n_mels)
    linear = np.transpose(magnitude.astype(np.float32)) # (T, 1+n_fft//2)
    return mel, linear

def ispectrogram(spectrogram):
    '''
    spectrogram: [f, t]
    '''
    return librosa.istft(spectrogram, hp.hop_length, win_length=hp.win_length, window="hann")

def spectrogram2wav(spectrogram, wave_filename=None, trim_tailing_silence=True):
    '''
    spectrogram: [t, f], i.e. [t, nfft // 2 + 1]
    '''
    spectrogram = spectrogram.T  # [f, t]
    X_best = copy.deepcopy(spectrogram)  # [f, t]
    for i in range(hp.n_iter):
        X_t = ispectrogram(X_best)
        est = librosa.stft(X_t, hp.n_fft, hp.hop_length, win_length=hp.win_length)  # [f, t]
        phase = est / np.maximum(1e-8, np.abs(est))  # [f, t]
        X_best = spectrogram * phase  # [f, t]
    X_t = ispectrogram(X_best)

    audio = np.real(X_t)
    if wave_filename is not None:
        if trim_tailing_silence:
            audio_trimed, index = librosa.effects.trim(audio)
            write(wave_filename, hp.sr, audio[:index[1]+hp.sr//4])
        else:
            write(wave_filename, hp.sr, audio)

    return audio

def test():
    wave_filename = "D:/Work/Tacotron/data/librosa.test/32-raw.wav"
    mel, linear = get_mel_and_linear_spec(wave_filename, pre_emphasis=0.97)

    spectrogram2wav(np.power(np.e, linear) ** hp.power, wave_filename.replace("raw", "reconstructed-pre-{}".format(hp.power)))
    spectrogram2wav(np.power(np.e, linear), wave_filename.replace("raw", "reconstructed-pre"))

def clean_phones(phones, use_stress=True):
    # delete digits
    if use_stress:
        phones = re.sub("([a-z]) ([0-9])", "\\1\\2", phones)
    else:
        phones = re.sub("([a-z]) ([0-9])", "\\1", phones)

    phones = [p for p in phones.split() if p != "." and p != ""]
    return phones

def analyze_info_file(train_info_filename, reduce_factor=2):
    print("Analyzing info file")
    
    path = os.path.dirname(train_info_filename)

    encoder_lens_chars = []
    encoder_lens_phones = []
    decoder_lens = []
    char_counter = Counter()
    phone_counter = Counter()
    for line in codecs.open(train_info_filename, "r", "utf-8"):
        items = line.split("\t")
        assert len(items) == 6
        wave_id, n_chars, n_frames, text, n_phones, phones = items

        n_chars = len(text.strip())
        n_frames = int(n_frames)
        char_counter.update(text.strip())

        phones = clean_phones(phones, use_stress=False)
        phone_counter.update(phones)

        encoder_length_chars = n_chars
        encoder_length_phones = len(phones)
        decoder_length = (n_frames+1) // reduce_factor

        encoder_lens_chars.append(n_chars)
        encoder_lens_phones.append(encoder_length_phones)
        decoder_lens.append(decoder_length)

    chars = sorted(char_counter.keys())
    phones = sorted(phone_counter.keys())
    print("len(char_counter) =", len(char_counter), "char_counter :", char_counter)
    # print("chars:", unicode(''.join(chars)))
    print("len(phone_counter) =", len(phone_counter), "phone_counter :", phone_counter)
    # print("phones:", ' '.join(phones))
    utils.plot_histogram(encoder_lens_chars, os.path.join(path, "encoder_lens_chars.png"))
    utils.plot_histogram(encoder_lens_phones, os.path.join(path, "encoder_lens_phones.png"))
    utils.plot_histogram(decoder_lens, os.path.join(path, "decoder_lens.png"))
    utils.plot_scatter(encoder_lens_chars, decoder_lens, os.path.join(path, "encoder_decoder_chars.png"))
    utils.plot_scatter(encoder_lens_phones, decoder_lens, os.path.join(path, "encoder_decoder_phones.png"))

def analyze_for_buckets(train_info_filename, reduce_factor=2):
    print("Analyzing data lengths...")
    path = os.path.dirname(train_info_filename)
    points = []
    encoder_lens = []
    decoder_lens = []
    char_counter = Counter()
    for line in codecs.open(train_info_filename, "r", "utf-8"):
        items = line.split("\t")
        assert len(items) >= 4
        wave_id, n_chars, n_frames, text = items[0], int(items[1]), int(items[2]), items[3]
        char_counter.update(text.strip())
        encoder_length = n_chars
        decoder_length = (n_frames+1) // reduce_factor

        encoder_lens.append(n_chars)
        decoder_lens.append(decoder_length)
        points.append([n_chars, decoder_length])

    chars = sorted(char_counter.keys())
    print("char counter", char_counter)
    # print("chars:", ''.join(chars))
    utils.plot_histogram(encoder_lens, os.path.join(path, "encoder_lens.png"))
    utils.plot_histogram(decoder_lens, os.path.join(path, "decoder_lens.png"))
    utils.plot_scatter(encoder_lens, decoder_lens, os.path.join(path, "encoder_decoder.png"))

def fix_output_bug(step_dir):
    filenames = os.listdir(step_dir)
    for filename in filenames:
        if filename.endswith(".attn.pkl"):
            print("processing", filename)
            filename = os.path.join(step_dir, filename)
            linear = pickle.load(open(filename, 'rb'))[-1]
            linear = post_process_linear_spec(linear, raise_power=1.2)
            spectrogram2wav(linear, filename.replace(".attn.pkl", ".raise.wav"))

def process_raw_data():
    start_time = time.time()
    scripts_dir = "D:\Work\Tacotron\data\eva\scripts\\"
    wave_dir = "D:\Work\Tacotron\data\eva\wave\\"
    save_dir = "D:\Work\Tacotron\data\eva\processed.v4\\"
    info_filename = "D:\Work\Tacotron\data\eva\info.processed.v4.all.utf8"

    info_template = "{wave_id}\t{n_chars}\t{n_frames}\t{text}\n" # text doesn't contain tabs
    fout_info = codecs.open(info_filename, "w", "utf-8")

    script_filenames = os.listdir(scripts_dir)
    for script_filename in script_filenames:
        id_range = script_filename[:-4]
        for line in codecs.open(scripts_dir+script_filename, 'r', 'utf-16'):
            items = line.strip().split("\t")
            if len(items) != 2:
                print("ERROR: wave_id={}, len(items)={}".format(items[0], len(items)))
                continue
            wave_id, text = items
            text = text.strip()            
            wave_filename = wave_dir + id_range + "\\" + wave_id + ".wav"

            if os.path.isfile(wave_filename) == False:
                print("File: {} doesn't exists".format(wave_filename))
                continue

            mel_spec, linear_spec = get_mel_and_linear_spec(wave_filename, trim_silence=True)

            pickle_filename = save_dir + wave_id + ".pkl"
            pickle.dump([wave_id, text, mel_spec, linear_spec], open(pickle_filename, 'wb'))
            assert len(mel_spec) == len(linear_spec)

            info_line = info_template.format(wave_id=wave_id, n_chars=len(text), n_frames=len(mel_spec), text=text)
            fout_info.write(info_line)
            print("Done processing file {}".format(wave_filename))

    fout_info.close()
    print("Time used {} minutes\n".format((time.time()-start_time)/60.0))


def read_script(xml_filename, out_dir):
    """
    Returns: [(sid, [(w, typ, phones)])]
    """
    print("paring script file", xml_filename)

    temp_filename = os.path.join(out_dir, "temp.xml")
    xml_string = codecs.open(xml_filename, 'r', 'utf-16').read().replace(u" xmlns=\"http://schemas.microsoft.com/tts\"", "")
    with codecs.open(temp_filename, 'w', 'utf-16') as fout:
        fout.write(xml_string)

    tree = ET.parse(temp_filename)
    node_script = tree.getroot()
    sents = []
    for node_si in node_script:
        sid = node_si.get("id")
        words = []
        for node_w in node_si.findall(u"./sent/words/w"):
            w = node_w.get("v")
            t = node_w.get("type")
            p = node_w.get("p")
            if p is None:
                p = "punc"+w
            words.append((w, t, p))
        sents.append((sid, words))
    return sents

def read_alignment(lab_filename, regex_algn):
    if regex_algn == None:
        raise Exception("The input regex_algn can't be none!")
    
    # can be found in voicefont\Intermediate\cmpModel2\trees_logF0.inf
    # regex to read alignment     
    phone_regex = re.compile(regex_algn["phone_regex"])
    fw_regex = re.compile(regex_algn["fw_regex"])
    bw_regex = re.compile(regex_algn["bw_regex"])
    bw_phrase_regex = re.compile(regex_algn["bw_phrase_regex"])

    pre_phone = None
    pre_fw, pre_bw = None, None
    line_info = [] # [(start, end, phone, fw, bw, bw_phrase)]
    for i, line in enumerate(codecs.open(lab_filename, "r", "utf-8")):
        if line.strip() == "":
            continue
        if line.strip() == ".":         
            continue
        fields = line.strip().lower().split(' ')
        assert len(fields) >= 3
        phone = phone_regex.findall(fields[2])[0]
        fw = fw_regex.findall(fields[2])
        bw = bw_regex.findall(fields[2])
        bw_phrase = bw_phrase_regex.findall(fields[2])

        if phone == "sil":
            fw = bw = bw_phrase = -1
        else:
            fw = int(fw[0])
            bw = int(bw[0])
            bw_phrase = int(bw_phrase[0])

        assert i % 5 == 0 or (phone==pre_phone and fw==pre_fw and bw==pre_bw)
        line_info.append((int(fields[0]), int(fields[1]), phone, fw, bw, bw_phrase))
        pre_phone, pre_fw, pre_bw = phone, fw, bw
    if len(line_info) % 5 != 0 or len(line_info) == 0:
        raise Exception("ERROR: lab file error")
    # pprint("line_info")
    # pprint(line_info)

    phone_info = [] # [(start, end, phone, word_index)]
    pre_fw, pre_bw = None, None
    word_index = -1
    for i in range(0,len(line_info), 5):
        phone = line_info[i][2]
        start = line_info[i][0]
        end = line_info[i+4][1]
        fw, bw, bw_phrase = line_info[i][3:6]
        if fw != pre_fw or bw != pre_bw:
            word_index += 1
        elif fw == 1 and bw == 1 and bw_phrase != pre_bw_phrase:
            word_index += 1
        phone_info.append((start, end, phone, word_index))
        pre_fw, pre_bw, pre_bw_phrase = fw, bw, bw_phrase
    # pprint("phone_info")
    # pprint(phone_info)

    pre_word_index = None
    word_info = [] # [(start, end, "sil"|"word")]
    start = 0
    for i, info in enumerate(phone_info):
        if info[3] != phone_info[start][3]:
            word = "sil" if phone_info[start][2] == "sil" else "word"
            word_info.append((phone_info[start][0], phone_info[i-1][1], word))
            start = i
    word = "sil" if info[2] == "sil" else "word"
    word_info.append((phone_info[start][0], info[1], word))
    # pprint("word_info")
    # pprint(word_info)
    return word_info

def get_splits(words, alignments, max_time):
    """
    Args:
        words: [(word, type=punc|normal)]
        alignments: [(begin, end, type=sil|word)], begin/end are given in 100ns
        max_time: given in seconds
        sr: sampling rate
    Returns:
        [(sample_begin, sample_end, word_begin, word_end)],
    """
    pprint("words")
    pprint(words)
    pprint("alignments")
    pprint(alignments)
    # ids of words in words (not punctuation)
    wids1 = [i for i, w in enumerate(words) if w[1] == "normal"]
    # ids of words in alignments (not silence)
    wids2 = [i for i, w in enumerate(alignments) if w[2]=="word"]
    if len(wids1) != len(wids2):
        raise Exception("ERROR: len(wids1)!=len(wids2): {}!={}".format(len(wids1), len(wids2)))

    # get candidates
    candidates = []
    for i in range(len(wids2)-1):
        # if words[wids1[i]+1][1]=="punc" and alignments[wids2[i]+1][2]=="sil":
        if alignments[wids2[i]+1][2]=="sil":
            candidates.append(i)
    candidates.append(len(wids2)-1)
    pprint("candidates")
    pprint(candidates)

    # split
    splits = []
    m = pow(10., 7)
    wstart = wids1[0]
    sstart = wids2[0]
    for i in range(len(candidates)):
        if i == len(candidates) - 1 or alignments[wids2[candidates[i+1]]][1]-alignments[sstart][0] > max_time * m:
            sbegin = alignments[sstart][0]
            k = wids2[candidates[i]]
            send = alignments[k][1]
            if candidates[i] == len(wids2) - 1: # end of sentence, keep all silence
                if k < len(alignments):
                    send += (alignments[-1][1] - alignments[k+1][0])
            else: # half of next silence
                send += (alignments[wids2[candidates[i]+1] - 1][1] - alignments[k+1][0]) / 2
            
            wbegin = wstart
            wend = wids1[candidates[i]]
            if wend+1 < len(words) and words[wend+1][1] == "punc": # add punc
                wend += 1
            splits.append((sbegin, send, wbegin, wend))
            if i < len(candidates) - 1:
                wstart = wids1[candidates[i]+1] # next "normal" word
                sstart = wids2[candidates[i]+1] # 
    print("splits", splits)
    return splits

def split_waves(wave_dir, script_dir, alignment_dir, out_dir, max_time, regex_algn):
    print("Generate the info file for Tacotron and split the long wave")
    print("The max wave time: {}".format(max_time))
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    # outputs
    saved_wave_dir = os.path.join(out_dir, "wave")
    info_filename = os.path.join(out_dir, "info.all.utf8")

    if not os.path.isdir(saved_wave_dir):
        os.makedirs(saved_wave_dir)
        
    # get all lab files
    lab_files = {} # {wave_id: filename}
    for root, dirs, files in os.walk(alignment_dir):
        for file in files:
            wave_id = file[:-4]
            lab_files[wave_id] = os.path.join(root, file)
    print("len(lab_files)", len(lab_files))

    # get all wave files {wave_id: filename}
    wave_files = {}
    for root, dirs, files in os.walk(wave_dir):
        for file in files:
            wave_id = file[:-4]
            wave_files[wave_id] = os.path.join(root, file)
    print("len(wave_files)", len(wave_files))

    # get all scripts
    sents = [] # [(sid, [(word, type, phones)])]
    for file in os.listdir(script_dir):
        xml_filename = os.path.join(script_dir, file)
        file_sents = read_script(xml_filename, out_dir)
        sents.extend(file_sents)
    sid2words = dict(sents) # {wave_id: [(w, type， phones)]}
    print("len(sid2words)", len(sid2words))

    # split wave
    wave_ids = sorted(sid2words.keys())
    text = []
    info = []
    m = pow(10., 7)
    
    overflow_cnt = 0
    exception_cnt = 0
    for wave_id in wave_ids[:]:
        try:
            print("processing", wave_id)
            if wave_id not in lab_files:
                print("Exception: lab file not found for", wave_id)
                continue
            alignment = read_alignment(lab_files[wave_id], regex_algn)

            words = sid2words[wave_id]
            splits = get_splits(words, alignment, max_time=max_time) # [(begin_sample, end_sample, begin_word, end_word)]
    
            wave_filename = wave_files[wave_id]
            print("loading wave_filename", wave_filename)
            y, sr = librosa.load(wave_filename, sr=None)
            assert sr == hp.sr

            # save splits
            for i, (sbegin, send, wbegin, wend) in enumerate(splits):
                split_wave_id = "{}_{}".format(wave_id, i)
                saved_wave_filename = os.path.join(saved_wave_dir, split_wave_id + ".wav")
                if (send - sbegin) > m * max_time:
                    overflow_cnt += 1
                audio = y[int(sbegin/m*hp.sr): int(send/m*hp.sr)]
                print("saving wave file", saved_wave_filename)
                # write(saved_wave_filename, hp.sr, audio)
                librosa.output.write_wav(saved_wave_filename, audio, hp.sr)
    
                split_words = ' '.join([w[0] for w in words[wbegin:wend+1]])
                split_phones = ' / '.join([w[2] for w in words[wbegin:wend+1]])
                n_frames = int((send-sbegin) / m / hp.frame_shift)
                temp = u"{}\t{}\t{}\t{}\t{}\t{}".format(split_wave_id, len(split_words), n_frames, split_words, len(split_phones.split()), split_phones)
                text.append(temp)
        except Exception as e:
            print("Exception when processing {}, message: {}".format(wave_id, e.message))
            exception_cnt += 1
    print("overflow_cnt =", overflow_cnt, "exception_cnt ", exception_cnt)
    with codecs.open(info_filename, "w", "utf-8") as fout:
        fout.write('\n'.join(text))

def process_split_data():
    start_time = time.time()
    split_dir = r"E:\bilian-share\Cortana\Ja-JP\jajp.splitted.5s"
    info_filename = os.path.join(split_dir, "info.all.utf8")
    wave_dir = os.path.join(split_dir, "wave")
    save_dir = os.path.join(split_dir, "pkl")

    exception_cnt = 0
    for line in codecs.open(info_filename, "r", 'utf-8'):
        items = line.strip().split("\t")
        if len(items) != 6:
            print("ERROR: wave_id={}, len(items)={}".format(items[0], len(items)))
            continue
        wave_id = items[0]
        text = items[3].strip()

        wave_filename = os.path.join(wave_dir, wave_id + ".wav")
        if os.path.isfile(wave_filename) == False:
            print("ERROR: wave file: {} doesn't exists".format(wave_filename))
            continue
        try:
            mel_spec, linear_spec = get_mel_and_linear_spec(wave_filename, trim_silence=False)
        except Exception as e:
            print("ERROR: exception when load wave file: {}, message: {}".format(wave_filename, e.message))
            exception_cnt += 1
            continue
        pickle_filename = os.path.join(save_dir, wave_id + ".pkl")
        pickle.dump([wave_id, text, mel_spec, linear_spec], open(pickle_filename, 'wb'))
        assert len(mel_spec) == len(linear_spec)
        print("Done processing file {}".format(wave_filename))

    print("Time used {} minutes, exception_cnt={}\n".format((time.time()-start_time)/60.0, exception_cnt))

def get_total_time_for_buckets(train_info_filename, buckets, use_phone=False, reduce_factor=2):
    total_time = 0.0
    bucket_time = [0.0] * len(buckets)

    for line in codecs.open(train_info_filename, "r", "utf-8"):
        items = line.split("\t")
        assert len(items) == 6
        wave_id, n_chars, n_frames, text, n_phones, phones = items

        n_chars = len(text.strip())
        n_frames = int(n_frames)        
        phones = clean_phones(phones, use_stress=True)

        encoder_size = len(phones) if use_phone else n_chars
        decoder_size = (n_frames + 1) // reduce_factor

        total_time += n_frames * hp.frame_shift
        bucket_id = utils.get_bucket_id(encoder_size, decoder_size, buckets)
        if bucket_id >= 0 and bucket_id < len(buckets):
            bucket_time[bucket_id] += n_frames * hp.frame_shift

    print("total_time:", total_time/3600.0)
    bucket_time = [t / 3600.0 for t in bucket_time]
    pprint(zip(buckets, bucket_time))
    return bucket_time

def spec_value_analysis():
    path = "D:\\Work\\Tacotron\\data\\eva\\eva.splitted.2s\\pkl"
    files = os.listdir(path)[:100]

    t = 1000000.0
    min_mel, min_linear = t, t
    max_mel, max_linear = -t, -t

    mels = []
    linears = []
    for file in files:
        filename = os.path.join(path, file)
        wave_id, text, mel_spec, linear_spec = pickle.load(open(filename, "rb"))
        mel_spec, linear_spec = pre_process_mel_and_linear_spec(mel_spec, linear_spec)

        min_mel = min(min_mel, mel_spec.min())
        min_linear = min(min_linear, linear_spec.min())
        max_mel = max(max_mel, mel_spec.max())
        max_linear = max(max_linear, linear_spec.max())

        mels.append(mel_spec)
        linears.append(linear_spec)

    print("mel: {} {}, linear: {} {}".format(min_mel, max_mel, min_linear, max_linear))

    n_bin = 1000
    mel_points = [] 
    linear_points = []
    mel_bin_size = (max_mel - min_mel) / n_bin
    linear_bin_size = (max_linear - min_linear) / n_bin
    for i, (mel, linear) in enumerate(zip(mels, linears)):
        mel = np.clip(mel, min_mel, max_mel)
        linear = np.clip(linear, min_linear, max_linear)

        mel = ((mel - min_mel) / mel_bin_size).astype(np.int32).reshape((-1))
        linear = ((linear - min_linear) / linear_bin_size).astype(np.int32).reshape((-1))

        mel_points.extend(list(mel))
        linear_points.extend(list(linear))

    utils.plot_histogram(mel_points, "D:\\Work\\Tacotron\\data\\eva\\eva.splitted.2s\\mel_values.jpg")
    utils.plot_histogram(linear_points, "D:\\Work\\Tacotron\\data\\eva\\eva.splitted.2s\\linear_values.jpg")

    print(Counter(mel_points))
    print(Counter(linear_points))

def test_read_zip_wave(zip_filename, out_dir):
    print("starting")
    wave_filename = "0000000001_0.wav"

    zfile = zipfile.ZipFile(zip_filename)
    fin = zfile.open(wave_filename, 'r')
    
    print("reading wave file")
    tmp = io.BytesIO(fin.read())
    audio, sr = soundfile.read(tmp)
    print(audio)
    print(sr)
    print(audio.shape, audio.min(), audio.max())
    wave_filename = os.path.join(out_dir, wave_filename)
    soundfile.write(wave_filename, audio, sr)

    audio, sr = soundfile.read(wave_filename)
    print(audio)
    print(sr)
    print(audio.shape, audio.min(), audio.max())

def get_linear_spec(path):
    print("Get linear spec!")
    # path = "\\\\MININT-A1PHHTQ\\Wave16kNormalized\\"
    dirs = os.listdir(path)
    for dirname in dirs:
        if len(dirname) != len("0000000001-0000000500"):
            continue
        print("processing", dirname)
        for filename in os.listdir(os.path.join(path, dirname)):
            wave_filename = os.path.join(path, dirname, filename)
            print("processing", wave_filename)
            mel, linear = get_mel_and_linear_spec(wave_filename)

            out_dir = os.path.join(path, "linear_spec")
            if(not os.path.isdir(out_dir)):
                os.makedirs(out_dir)

            linear_filename = os.path.join(out_dir, filename[:-3] + "linear.npy")
            np.save(linear_filename, linear)

def parse_args():
    parser = argparse.ArgumentParser(description='details',
                        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-mode', type=str, default="",
                        help='''-mode split|analyze|bucket|analyzebucket|testzip|splitmlf, 
    
    split: split wave with 16k wave, forced align wihch tunes in VoiceModelTrainer, xmlscript:
    usage: -mode split -wave wave_dir -lab align_dir -script script_dir -out out_dir [-maxtime 5]

    analyze: analyze the Tacotron data from Tacotron input info file:
    usage: -mode analyze -infofile infofile_path

    bucket: get totle time for buckets:
    usage: -mode bucket -infofile infofile_path [-bucket [[60, 150], [70, 150], [80, 160], [100, 200], [120, 240],[150, 300],[170, 340], [200, 400]] []] 
    usage: -mode bucket -infofile infofile_path [-bucket [[60, 150], [70, 150], [80, 160], [100, 200], [120, 240],[150, 300],[170, 340], [200, 400]] []] 
    
    analyzeBucket: analyze the Tacotron data from Tacotron input info file:
    usage: -mode analyzebucket -infofile infofile_path

    testzip: test if the wave zip works in Tacotron training:
    usage: -mode testzip wave_zip_path -out out_dir
    
    testzip: TBD:
    usage: -mode linearspec -wave wave_dir

    splitmlf: generate lab file form mlf file:
    usage: -mode splitmlf mlf_path -out out_dir''')

    parser.add_argument('-wave', type=str, default="",
                        help='input 16k wave folder')
    parser.add_argument('-lab', type=str, default="",
                        help='forced alignment lab folder')
    parser.add_argument('-script', type=str, default="",
                        help='the xml script with pronunciation and dot')
    parser.add_argument('-infofile', type=str, default="",
                        help='info file for Tacotron training')
    parser.add_argument('-bucket', type=list, default=[[60, 150], [70, 150], [80, 160], [100, 200], [120, 240],[150, 300],[170, 340], [200, 400]],
                        help='bucket setting')
    parser.add_argument('-maxtime', type=float, default=5.0,
                        help='output folder')
    parser.add_argument('-reg_type', type=str, default="rinna",
                        help='the regex type to get align from lab file')
    parser.add_argument('-reg_file', type=str, default=r".\data_utils_align_reg.json",
                        help='the regex file, if you want to generate a new type lab file, please update it!')
    parser.add_argument('-mlf', type=str, default="",
                        help='mlf file, generate forced alignment lab file from mlf')
    parser.add_argument('-out', type=str, default="",
                        help='output folder')
                        
    return parser

def split_mlf(mlf_path, out_dir):
    print("Genrate lab file from mlf!")
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    number = 0
    lines = []
    with codecs.open(mlf_path, "r", "utf-8") as fr:
        for line in fr:
            if ("MLF" in line) or line == "":                
                continue
            if ".lab" in line:
                if len(lines) > 0:
                    with codecs.open(os.path.join(out_dir, number), "wb", "utf-8") as fw:
                        fw.writelines(lines)
                        print("Save success:", number)
                        lines = []
                number = line[3:-3]
                continue
            lines.append(line)
        # last file
        with codecs.open(os.path.join(out_dir, number), "wb", "utf-8") as fw:
            fw.writelines(lines)
            print("Save success:", number)
    
    print("Complete!")
    
def main():
    parser = parse_args()
    args = parser.parse_args()
    if args.mode == '':
        parser.print_help()
    elif args.mode.lower() == 'split':
        if not os.path.isdir(args.wave):
            parser.print_help()
            print("The wave folder is not exist: {}".format(args.wave))
            return 
        if not os.path.isdir(args.lab):
            parser.print_help()
            print("The align lab folder is not exist: {}".format(args.lab))
            return 
        if not os.path.isdir(args.script):
            parser.print_help()
            print("The xml script folder is not exist: {}".format(args.script))
            return

        with open(args.reg_file) as fw:
            json_data = json.load(fw)
        if args.reg_type not in json_data:
            parser.print_help()
            print("The reg type \"{}\" is not in json file".format(args.reg_type))
            print("Json: {}".format(json_data))
            return
        split_waves(args.wave, args.script, args.lab, args.out, args.maxtime, json_data[args.reg_type])
    elif args.mode.lower() == 'analyze':
        if not os.path.isfile(args.infofile):
            parser.print_help()
            print("The Tacotron training file is not exist: {}".format(args.infofile))
            return
        analyze_info_file(args.infofile)
    elif args.mode.lower() == 'analyzebucket':
        if not os.path.isfile(args.infofile):
            parser.print_help()
            print("The Tacotron training file is not exist: {}".format(args.infofile))
            return
        analyze_for_buckets(args.infofile)
    elif args.mode.lower() == 'bucket':
        if not os.path.isfile(args.infofile):
            parser.print_help()
            print("The Tacotron training file is not exist: {}".format(args.infofile))
            return     
        get_total_time_for_buckets(args.infofile, args.bucket, use_phone=False)
    elif args.mode.lower() == 'testzip':
        if not os.path.isfile(args.wavezip):
            parser.print_help()
            print("The wave zip file is not exist: {}".format(args.wavezip))
            return 
        test_read_zip_wave(args.wavezip, args.out)
    elif args.mode.lower() == 'splitmlf':
        if not os.path.isfile(args.mlf):
            parser.print_help()
            print("The mlf file is not exist: {}".format(args.mlf))
            return 
        split_mlf(args.mlf, args.out)
    elif args.mode.lower() == 'linearspec':
        if not os.path.isdir(args.wave):
            parser.print_help()
            print("The wave folder is not exist: {}".format(args.wave))
            return 
        get_linear_spec(args.wave)
    else:
        parser.print_help()
        print("Wrong mode input: {}".format(args.mode))
    # spec_value_analysis()
    # process_split_data()
    # get_total_time_for_buckets([[60, 150], [70, 150], [80, 160]], use_phone=False) # about 20 hours
    # get_total_time_for_buckets([[60, 150], [70, 150], [80, 160]], use_phone=True) # about 20 hours
    # test()
    # fix_output_bug("/mnt/miach/log/2017-07-07-12-07-34/generated/0000063500/")
    #get_linear_spec()
    pass

if __name__ == "__main__":
    main()
