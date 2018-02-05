# coding=utf-8

from __future__ import print_function

import os
import pickle
import sys
from datetime import datetime

from hyper_params import HyperParams as hp

# global variables
CURRENT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
START_DATE_STRING = "{0:%Y-%m-%d-%H-%M-%S}".format(datetime.now())

# Install packages
# packages_directory = os.path.join(CURRENT_DIRECTORY, "packages")
# if not os.path.isdir(packages_directory):
    # os.mkdir(packages_directory)
# os.system("pip install --target=%s" % (packages_directory, ) + " librosa soundfile matplotlib scipy")
# sys.path.insert(0, packages_directory)

import codecs
import data_reader
import data_utils
import data_utils_normalize
import json
import librosa
import numpy as np
import tacotron2
import tensorflow as tf
import threading
import time
import utils

from pprint import pprint

def train(config):
    print("config in train function", config)

    use_stop_token = config["use_stop_token"]
    use_linear_spec = config["use_linear_spec"]

    train_start_time = time.time()
    cached_storage, bucket_wave_ids, test_batch = utils.load_data_for_queue_reader(config) # this also changes config, should be called before building model

    post_process_func = data_utils_normalize.post_process_linear_spec
    #post_process_func = data_utils_normalize.post_process_mel_spec

    utils.generate_wave_for_test_data(
        test_batch,
        os.path.join(config["generated_dir"], "test"),
        post_process_func,
        use_stop_token)

    batch_size = config["batch_size"]
    n_thread = config["n_thread"]
    n_gpu = config["n_gpu"]

    with tf.Graph().as_default():
        coord = tf.train.Coordinator()
        reader = data_reader.QueueDataReader(bucket_wave_ids, cached_storage, config, coord)

        train_model = tacotron2.Tacotron2(n_gpu, config, is_training=True, reuse=False)
        test_model = tacotron2.Tacotron2(1, config, is_training=False, reuse=True)

        checkpoint_path = os.path.join(config["summary_dir"], "model.ckpt")
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=20, keep_checkpoint_every_n_hours=6)

        custom_config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
        custom_config.gpu_options.allocator_type = 'BFC'
        custom_config.gpu_options.per_process_gpu_memory_fraction = 0.95
        #custom_config.gpu_options.allow_growth = True
        session = tf.Session(config=custom_config)
        session.run(tf.global_variables_initializer())

        summary_writer = tf.summary.FileWriter(config["summary_dir"], session.graph)

        def _recover_model(path):
            # restore model parameters
            cur_global_step = utils.restore_model(session, saver, path)
            train_model.update_global_step(session, cur_global_step+1)

            info_filename = os.path.join(os.path.dirname(path), "train.info.pkl")
            loss_sum, learning_rates = pickle.load(open(info_filename, 'rb'))

            return loss_sum[:cur_global_step+1], learning_rates, cur_global_step
        # recover model
        path = config["restore_from"] # something like /home/miach/log/model.ckpt-12000
        init_lr = config["learning_rates"][0][1]
        if path is not None and path != "":
            loss_sum, learning_rates, last_saved_global_step = _recover_model(path)
            lr = learning_rates[last_saved_global_step]
            train_model.update_learning_rate(session, lr=lr)
        else:
            loss_sum = [0.0]
            learning_rates = {0: init_lr}
            last_saved_global_step = 0
            train_model.update_learning_rate(session, lr=init_lr)

        json_filename = os.path.join(config["summary_dir"], "config.json")
        with codecs.open(json_filename, 'w', 'utf-8') as fout:
            json.dump(config, fout)

        info_filename = os.path.join(config["summary_dir"], "train.info.pkl")
        try:
            reader.start()
            batch = reader.next_batch()
            while batch:
                epoch = reader.cur_epoch
                step_start_time = time.time()
                train_model.update_learning_rate(session, scheme=config["scheme"])
                # [B, T], [B, r*T', n_mel], [B, r*T', n_linear]
                #char_ids, mel_spec, linear_spec, bucket_id, wave_ids, stop_token_indexes = batch
                if use_stop_token is True:
                    char_ids, mel_spec, bucket_id, wave_ids, stop_tokens = batch
                else:
                    char_ids, mel_spec, bucket_id, wave_ids = batch

#                predicted_mel_spec, attentions, stop_tokens, train_loss, summary_str, _ = train_model.step(
                if use_linear_spec is True:
                    predicted_mel_spec, attentions, train_loss, summary_str, _ = train_model.step(
                        session, char_ids, mel_spec, bucket_id)
                else:
                    predicted_mel_spec, attentions, train_loss, summary_str, _ = train_model.step(
                        session, char_ids, mel_spec, bucket_id)
                #session, char_ids, mel_spec, bucket_id, stop_token_indexes)


                loss_sum.append(loss_sum[-1] + train_loss)
                cur_global_step = int(session.run(train_model.global_step)) - 1

                # print loss
                if cur_global_step % config["print_loss_every"] == 0:
                    print("epoch {}, gs={}: loss={:.6f}, lr={:.5f}, time={:.3f}".format(
                        epoch, cur_global_step, train_loss, session.run(train_model.learning_rate), time.time()-step_start_time))

                # detect loss divergence
                if np.isnan(train_loss) or utils.is_diverging(loss_sum):
                    print("Model divergence detected, restore from last saved model and reduce learning rate")
                    lr = session.run(train_model.learning_rate) * 0.7
                    path = os.path.join(config["summary_dir"], "model.ckpt-{}".format(last_saved_global_step))
                    loss_sum, learning_rates, last_saved_global_step = _recover_model(path)
                    learning_rates[last_saved_global_step] = lr
                    train_model.update_learning_rate(session, lr=lr)
                    continue

                # add summary
                if cur_global_step % config["add_summary_every"] == 0:
                    summary_writer.add_summary(summary_str, cur_global_step)

                # save check point
                if cur_global_step % config["check_point_every"] == 0:
                    learning_rates[cur_global_step] = session.run(train_model.learning_rate)
                    pickle.dump([loss_sum, learning_rates], open(info_filename, 'wb'))
                    saver.save(session, checkpoint_path, global_step=cur_global_step)
                    last_saved_global_step = cur_global_step

                # generate waves using test model, and save waves and attention heatmaps
                if cur_global_step % config["generate_wave_every"] == 0:
                    print("Saving attention map of training batch...")
                    epoch_dir = os.path.join(config["generated_dir"], str(cur_global_step).zfill(10))
                    utils.ensure_dir(epoch_dir)
                    batch_size = config["batch_size"]
                    #test_char_ids, test_mel_spec, test_linear_spec, test_bucket_id, test_wave_ids, test_stop_tokens = test_batch
                    if use_stop_token is True:
                        test_char_ids, test_mel_spec, test_bucket_id, test_wave_ids, test_stop_tokens = test_batch
                    else:
                        test_char_ids, test_mel_spec, test_bucket_id, test_wave_ids = test_batch
                    test_start_time = time.time()
                    #predicted_mel_spec, attentions, stop_tokens = test_model.step(
                    if use_linear_spec is True:
                        predicted_mel_spec, attentions= test_model.step(
                            session, test_char_ids, test_mel_spec, test_bucket_id)
                    else:
                        predicted_mel_spec, attentions = test_model.step(
                            session, test_char_ids, test_mel_spec, test_bucket_id)
                    print("test time for one batch = {:.3f} seconds".format(time.time()-test_start_time))

                    # save mel
                    #mel_filename = os.path.join(epoch_dir, 'mel_')
                    #np.save(mel_filename, predicted_mel_spec)
                    # np.save(
                    #     'E:/projects/tacotron/GoldenBridge/private/tacotron2/testtraining/miaohong_recording/mel/stop_token_' + str(
                    #         cur_global_step), stop_tokens)

                    def generate_and_save(indexes):
                        for i in indexes:
                            wave_id = test_wave_ids[i]
                            prefix = "train." if i < batch_size/2 else "test."
                            wave_filename = os.path.join(epoch_dir, prefix + wave_id + ".wav")
                            pkl_filename = os.path.join(epoch_dir, prefix + wave_id + ".attn.pkl")
                            jpg_filename = os.path.join(epoch_dir, prefix + wave_id + ".attn.png")
                            to_save = "[wave_id, test_char_ids[i], predicted_mel_spec[i], predicted_linear_spec[i]]"
                            print("saving wav file", wave_filename)

                            # output mels
                            mel_filename = os.path.join(epoch_dir, prefix + wave_id + ".mel")
                            np.save(mel_filename, predicted_mel_spec[i])

                            # if use_linear_spec is True:
                                # post_process_func(predicted_linear_spec[i], wave_filename, raise_power=1.0)
                                # wave_filename = wave_filename.replace(".wav", ".raise.wav")
                                # post_process_func(predicted_linear_spec[i], wave_filename, raise_power=1.2)
                            #else:
                                #data_utils_normalize.post_process_mel_spec(predicted_mel_spec[i], wave_filename, raise_power=1.0)
                                #wave_filename = wave_filename.replace(".wav", ".raise.wav")
                                #data_utils_normalize.post_process_mel_spec(predicted_mel_spec[i], wave_filename, raise_power=1.2)

                            utils.plot_attention(attentions[i], jpg_filename, X_label="decoder", Y_label="encoder")
                            # pickle.dump([to_save] + eval(to_save), open(pkl_filename, "wb"))

                    threads = []
                    # do not generate wave instead we may compare mel spectrom
                    for i in range(n_thread):
                        indexes = range(i, len(test_wave_ids), n_thread)
                        t = threading.Thread(target=generate_and_save, args=(indexes,))
                        threads.append(t)
                        t.start()
                    for t in threads:
                        t.join()
                    sys.stdout.flush()
                batch = reader.next_batch()
        except KeyboardInterrupt:
            print("Keyboard interrupted")
        except Exception as e:
            print("Exception happened during training, message:", e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
        finally:
            coord.request_stop()
            coord.join()

def prepare_for_training(args):
    """Prepare directories, update and return config for Philly"""
    global CURRENT_DIRECTORY
    global START_DATE_STRING

    # update configs for Philly
    if args.isphilly is True:
        config_filename = args.configfile
    else:
        config_filename = os.path.join(args.datadir, args.configfile)
    config = utils.load_config(config_filename)

    config["is_philly"] = args.isphilly
    config["data_directory"] = args.datadir

    # create directories for log
    if args.isphilly:
        endi = CURRENT_DIRECTORY.find("@")
        starti = CURRENT_DIRECTORY.rfind("\\", 0, endi) + 1
        aether_id = CURRENT_DIRECTORY[starti: endi]
        assert len(aether_id) > 0
        log_dir = os.path.join(config["data_directory"], "log", aether_id)
        utils.ensure_dir(log_dir)
        print("aether_id:", aether_id)
        config["aether_id"] = aether_id

        # args.outputdit=/hdfs/ipgsp/sys/jobs/application_1501012364013_22503/models
        log_dir = args.outputdir
    else:
        log_dir = os.path.join(CURRENT_DIRECTORY, "log")
    # check if this is a retry on Philly
    is_retry = False
    if args.isphilly and os.path.isdir(log_dir):
        is_retry = True
        try:
            dirs = os.listdir(log_dir)
            if len(dirs) != 1:
                print("Find previous: there are more than one directories under log directory, using the last(lastest) one.")
            if len(dirs[-1]) != 19 or len(dirs[-1].split("-")) != 6:
                raise Exception("Find previous: the last one is not a time-format directory.")
            time_dir = os.path.join(log_dir, dirs[-1])
            summary_dir = os.path.join(time_dir, "summary")
            checkpoint_path = tf.train.latest_checkpoint(summary_dir)
            if checkpoint_path is None:
                raise Exception("Find previous: no checkpoint found under summary directory")

            generated_dir = os.path.join(time_dir, "generated")
            if not os.path.isdir(generated_dir):
                raise Exception("Find previous: generated dir not found under log dir")

            config["restore_from"] = checkpoint_path
        except Exception as e:
            print("Exception happend when try to recover from previous run on Philly, message:", e)
            is_retry = False
    
    if not is_retry:
        time_dir = os.path.join(log_dir, START_DATE_STRING)
        summary_dir = os.path.join(time_dir, "summary")
        generated_dir = os.path.join(time_dir, "generated")
        utils.ensure_dir(log_dir)
        utils.ensure_dir(time_dir)
        utils.ensure_dir(summary_dir)
        utils.ensure_dir(generated_dir)
        
    # redirect standard output
    sys.stdout = codecs.open(os.path.join(time_dir, "log.txt"), 'a+', 'utf-8')
    if is_retry:
        print("Recovering from previous run on Philly")

    print("config read from file")
    pprint(config)
    print("command line arguments")
    pprint(args)

    config["is_retry"] = is_retry
    config["log_dir"] = time_dir
    config["summary_dir"] = summary_dir
    config["generated_dir"] = generated_dir    

    if args.training_info_filename:
        config["training_info_filename"] = args.training_info_filename

    # update arguments from command line
    if not is_retry and args.restore_from:
        config["restore_from"] = args.restore_from
    if args.learning_rates:
        config["learning_rates"] = eval(args.learning_rates)
    if args.wave_zip_filename:
        config["wave_zip_filename"] = args.wave_zip_filename
    if args.ngpu:
        config["n_gpu"] = int(args.ngpu)
    if args.batch_size:
        config["batch_size"] = args.batch_size
    if args.case_sensitive is not None:
        config["case_sensitive"] = args.case_sensitive
    if args.use_phone is not None:
        config["use_phone"] = args.use_phone
    config["use_stress"] = args.use_stress
    if args.reduce_factor:
        config["reduce_factor"] = args.reduce_factor
    if args.mel_spec_size:
        config["mel_spec_size"] = args.mel_spec_size
    else:
        config["mel_spec_size"] = hp.mel_spec_size
    if args.linear_spec_size:
        config["linear_spec_size"] = args.linear_spec_size
    else:
        config["linear_spec_size"] = hp.linear_spec_size
    if args.buckets:
        config["buckets"] = eval(args.buckets)
    config["normalize"] = args.normalize
    config["n_thread"] = args.n_thread
    if args.attention_type:
        config["attention_type"] = args.attention_type
    config["stop_gradient"] = args.stop_gradient
    config["scheme"] = args.scheme
    config["warmup_steps"] = args.warmup_steps

    # complete path
    config["training_info_filename"] = os.path.join(config["data_directory"], config["training_info_filename"])
    config["n_max_wave"] = args.n_max_wave

    path = config["restore_from"] # something like /home/miach/log/model.ckpt-12000
    if path is not None and path != "":
        path = os.path.dirname(path)
        config_filename = os.path.join(path, "config.json")
        with codecs.open(config_filename, 'r', 'utf-8') as fin:
            config_temp = json.load(fin)
        config["test_wave_ids"] = config_temp["test_wave_ids"]
    else:
        config["test_wave_ids"] = None

    if args.other:
        items = args.other.strip().split(":")
        assert len(items) % 3 == 0
        for i in range(0, len(items), 3):
            itype, ikey, ivalue = items[i: i+3]
            config[ikey] = eval(itype)(ivalue)
    
    print("updated final config")
    pprint(config) #pretty print
    return config

def main():
    args = utils.get_train_arguments()
    config = prepare_for_training(args)
    train(config)

if __name__ == "__main__":
    main()
