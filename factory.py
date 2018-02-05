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

def get_mel_for_wavenet(n_thread=32):
    start_time = time.time()
    threads = []
    # zip_filename = "./data/rinna.wave.all.zip"
    # output_dir = "./data/rinna.wave.all.mel"
    zip_filename = "./data/eva.wave.all.zip"
    output_dir = "./data/eva.wave.all.mel"
    zfile = zipfile.ZipFile(zip_filename)
    wave_filenames = zfile.namelist()

    def extract_and_save_mel(zfile, wave_filenames, output_dir):
        for wave_filename in wave_filenames:
            fin = zfile.open(wave_filename, 'r')
            tmp = io.BytesIO(fin.read())
            audio, sr = soundfile.read(tmp)

            mel_spec, linear_spec = data_utils_normalize.get_mel_and_linear_spec(audio)
            npy_filename = os.path.join(output_dir, wave_filename.replace("wav", "npy"))
            np.save(npy_filename, mel_spec)            

    for i in range(n_thread):
        filenames = wave_filenames[i::n_thread]
        threads.append(threading.Thread(target=extract_and_save_mel, args=(zfile, filenames, output_dir)))
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    zfile.close()
    
    print("mel features generation done, start to zip")
    os.system("zip -r ./data/eva.wave.all.mel.zip ./data/eva.wave.all.mel")

def main():
    get_mel_for_wavenet()
    pass

if __name__ == "__main__":
    main()
