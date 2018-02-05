# coding: utf-8

import codecs
import json
import os

import service_eval


class Model:

    def __init__(self, restore_from="sample_data/Model/model.ckpt-1000", buckets="[[100,200]]"):
        log_dir = os.path.dirname(restore_from)
        config_filename = os.path.join(log_dir, "config.json")
        with codecs.open(config_filename, 'r', 'utf-8') as fin:
            self.config = json.load(fin)
            print(self.config)
        self.session, self.model, self.post = service_eval.generate_model(self.config, restore_from, buckets)

    def eval(self, text, mel=False):
        '''
            return a data dict, key is text key, value is audio
        '''
        return service_eval.feed_batch(self.session, self.model, self.config, self.post, text, mel)
