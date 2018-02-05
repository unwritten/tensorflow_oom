# coding=utf-8

import components
import decoder
import tensorflow as tf
from hyper_params import HyperParams as hp

class Tacotron2(object):
    """ implementation of NATURAL TTS SYNTHESIS BY CONDITIONING WAVENET ON MEL SPECTROGRAM PREDICTIONS
    """
    def __init__(self, n_gpu, config, is_training=True, variable_scope="Tacotron2", reuse=None, gpu_GLA=False):
        """
        Args:
            n_gpu: number of GPUs, set to 1 if is_training==False
            config: configuratoin dictionary
            is_training: is training or not
            reuse: reuse variable scope or not
        """
        self.config = config
        self.init_lr = config["learning_rates"][0][1]
        self.warmup_steps = config.get("warmup_steps", 10000)

        self.is_training = is_training
        self.gpu_GLA = gpu_GLA

        self.use_linear_spec = config["use_linear_spec"]

        buckets = config["buckets"]
        mel_spec_size = config["mel_spec_size"]
        linear_spec_size = config["linear_spec_size"]
        keep_prob = self.config["keep_prob"] if self.is_training else 1.0

        encoder_size, decoder_size = buckets[-1]

        # define placeholder and split them for GPUs
        # with tf.device("/job:localhost/replica:0/task:0/device:XLA_GPU:0"):
        with tf.device("/cpu:0"):
            # placeholders for encoder inputs
            self.encoder_inputs = [] # [(B)] * encoder_size
            self.tower_encoder_inputs = [[] for i in range(n_gpu)]
            for i in range(encoder_size):
                inputs = tf.placeholder(tf.int32, shape=[None], name="encoder_inputs_{}".format(i))
                self.encoder_inputs.append(inputs)

                inputs_splitted = tf.split(inputs, n_gpu, 0)
                for j, inp in enumerate(inputs_splitted):
                    self.tower_encoder_inputs[j].append(inp)

            # placeholders for decoder outputs
            self.decoder_outputs = [] # [(B, n_mel)] * decoder_size
            self.tower_decoder_outputs = [[] for i in range(n_gpu)]
            for i in range(decoder_size):
                outputs = tf.placeholder(tf.float32, shape=[None, mel_spec_size], name="decoder_outputs_{}".format(i))
                self.decoder_outputs.append(outputs)

                outputs_splitted = tf.split(outputs, n_gpu, 0)
                for j, outp in enumerate(outputs_splitted):
                    self.tower_decoder_outputs[j].append(outp)


            # get decoder inputs from decoder outputs
            self.tower_decoder_inputs = [[] for i in range(n_gpu)]
            for i in range(n_gpu):
                batch_size_tensor = tf.shape(self.tower_decoder_outputs[i][0])[0]
                self.tower_decoder_inputs[i] = [tf.zeros([batch_size_tensor, mel_spec_size])] # go frame with all zeros
                for outputs in self.tower_decoder_outputs[i][:-1]:
                    self.tower_decoder_inputs[i].append(outputs)

            # placeholders for stop_token_indexes
            # self.stop_token_outputs = [] # [(B, n_mel)] * decoder_size
            # self.tower_stop_token_outputs = [[] for i in range(n_gpu)]
            # for i in range(decoder_size):
            #     outputs = tf.placeholder(tf.float32, shape=[None, 1], name="stop_token_indexes_{}".format(i))
            #     self.stop_token_outputs.append(outputs)
            #
            #     outputs_splitted = tf.split(outputs, n_gpu, 0)
            #     for j, outp in enumerate(outputs_splitted):
            #         self.tower_stop_token_outputs[j].append(outp)
        print("placeholders created")
        
        feed_previous = (not is_training)

        if not is_training:
            with tf.variable_scope(variable_scope, reuse=reuse):
                with tf.device("/gpu:0"):
                    with tf.name_scope("test_gpu_0"):

                        # seq2seq model
                        #outputs, attentions, stop_tokens, n_predicted = self.seq2seq(
                        outputs, attentions = self.seq2seq(
                            self.tower_encoder_inputs[0],
                            self.tower_decoder_inputs[0],
                            #self.tower_stop_token_outputs[0],
                            feed_previous=feed_previous)

                        # reshape for post process
                        mel = tf.stack(outputs, axis=1) # [B, T', r*n_mel]
                        mel = tf.reshape(mel, [-1, decoder_size, mel_spec_size])

                        # post-net
                        postnet_conv_layers = config["postnet_conv_layers"]
                        postnet_conv_filters = config["postnet_conv_filters"]
                        postnet_conv_kernel_size = config["postnet_conv_kernel_size"]
                        n_conv1d_hiddens = [postnet_conv_filters] * postnet_conv_layers
                        mel_residual = components.conv_stack(mel, n_conv1d_hiddens=n_conv1d_hiddens,
                                   keep_probs=[keep_prob] * postnet_conv_layers, kernel_size=postnet_conv_kernel_size,
                                   activations=tf.nn.tanh,
                                   last_linear=True, is_training=False)

                        # project to mel size
                        mel_residual = tf.reshape(mel_residual, [-1, n_conv1d_hiddens[-1]])
                        mel_residual = components.linear_layer(mel_residual, n_conv1d_hiddens[-1], mel_spec_size, bias=False,
                                                                variable_scope="linear")
                        mel_residual = tf.reshape(mel_residual, [-1, decoder_size, mel_spec_size])

                        # add mel with its residual
                        mel += mel_residual

                        mel = tf.reshape(mel, [-1, decoder_size, mel_spec_size])
                        self.mel_outputs = mel # [B, r*T', n_mel]

                        attention = tf.stack(attentions, axis=-1) # [B, T, T']
                        self.attentions = attention # [B, T, T']

                        # mel to linear
                        if self.use_linear_spec is True:
                            n_conv1d_hiddens = [256, mel_spec_size]
                            postnet_input = mel
                            postnet_output = components.CBHG(postnet_input, K=8, n_conv1d_hiddens=n_conv1d_hiddens, is_training=is_training)
                            postnet_output = tf.reshape(postnet_output, [-1, 256])

                            linear = components.linear_layer(postnet_output, 256, linear_spec_size, bias=True, variable_scope="linear_project")
                            linear = tf.reshape(linear, [-1, decoder_size, linear_spec_size])
                            self.linear_outputs = linear  # [B, r*T', n_linear]

                        # stop_tokens_stacked = tf.stack(stop_tokens, axis=1)  # [B, T', r*n_mel]
                        # self.stop_token_predicted_outputs = stop_tokens_stacked
                        # to-do:
                        # if gpu_GLA:
                        #    self.wav_outputs = data_utils_normalize.inv_spectrogram_tensorflow(self.linear_outputs)
            return

        # for train graph
        self.global_step_placeholder = tf.placeholder(tf.int32, shape=[], name="global_step_placeholder")
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.update_global_step_op = tf.assign(self.global_step, self.global_step_placeholder)

        self.learning_rate_placeholder = tf.placeholder(tf.float32, shape=[], name="learning_rate_placeholder")
        self.learning_rate = tf.get_variable("learning_rate", [], initializer=tf.constant_initializer(0.001), trainable=False)
        self.update_learning_rate_op = tf.assign(self.learning_rate, self.learning_rate_placeholder)

        # with β1 = 0.9, β2 = 0.999,  = 10−6
        # a learning rate of 10−3
        # exponentially decaying to 10−5
        # starting after
        # 50,000 iterations
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-06)

        ##self.stop_token_optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-06)

        #  apply L2 regularization with weight 10−6
        self.regularizer = tf.contrib.layers.l2_regularizer(scale=0.000001)

        self.tower_mel_outputs = [] # [(B, r*T', n_mel)] * n_gpu
        self.tower_linear_outputs = []  # [(B, r*T', n_linear)] * n_gpu
        self.tower_attentions = [] # [(B, T', T)] * n_gpu
        self.tower_losses1 = [] # loss for mel spec
        self.tower_losses2 = [] # loss for mel residual adjusted mel
        self.tower_losses3 = []
        self.tower_l2 = [] # loss for l2 reg
        self.tower_losses = [] # for total loss = mel_loss + linear_loss
        #self.tower_stop_token_losses = []

        #self.tower_stop_token_predicted_outputs = []

        self.tower_grads_and_vars = []
       # self.tower_stop_token_grads_and_vars = []
        for i in range(n_gpu):
            with tf.variable_scope(variable_scope, reuse=True if i > 0 else reuse):
                with tf.device("/gpu:{}".format(i)):
                    with tf.name_scope("trainig_gpu_{}".format(i)):
                        print("building train graph on GPU {}".format(i))
                        # seq2seq
                        #outputs, attentions, stop_tokens, n_predicted = self.seq2seq(
                        outputs, attentions= self.seq2seq(
                            self.tower_encoder_inputs[i],
                            self.tower_decoder_inputs[i],
                            #self.tower_stop_token_outputs[i],
                            feed_previous=feed_previous)

                        # reshape for post process
                        mel = tf.stack(outputs, axis=1) # [B, T', r*n_mel]
                        mel = tf.reshape(mel, [-1, decoder_size, mel_spec_size])

                        # stop token
                        # stop_tokens_stacked = tf.stack(stop_tokens, axis=1)  # [B, T', r*n_mel]
                        # stop_tokens_stacked = tf.reshape(stop_tokens_stacked, [-1, decoder_size, 1])
                        # stop_token_expected = tf.reshape(self.tower_stop_token_outputs[i], [-1, decoder_size, 1])
                        #
                        # loss_stop_token = tf.losses.mean_squared_error(stop_token_expected, stop_tokens_stacked)
                        #
                        # self.tower_stop_token_losses.append(loss_stop_token)
                        # self.tower_stop_token_predicted_outputs.append(stop_tokens_stacked)

                        # end of stop token

                        # post-net
                        postnet_conv_layers = config["postnet_conv_layers"]
                        postnet_conv_filters = config["postnet_conv_filters"]
                        postnet_conv_kernel_size = config["postnet_conv_kernel_size"]
                        n_conv1d_hiddens = [postnet_conv_filters] * postnet_conv_layers
                        mel_residual = components.conv_stack(mel, n_conv1d_hiddens=n_conv1d_hiddens,
                                   keep_probs=[keep_prob] * postnet_conv_layers, kernel_size=postnet_conv_kernel_size,
                                   activations=tf.nn.tanh,
                                   last_linear=True, is_training=True)

                        # project to mel size
                        mel_residual = tf.reshape(mel_residual, [-1, n_conv1d_hiddens[-1]])

                        mel_residual = components.linear_layer(mel_residual, n_conv1d_hiddens[-1], mel_spec_size, bias=True,
                                                                variable_scope="linear")
                        mel_residual = tf.reshape(mel_residual, [-1, decoder_size, mel_spec_size])

                        mel_postnet = mel + mel_residual

                        mel_gold = tf.stack(self.tower_decoder_outputs[i], axis=1) # [B, T', r*n_mel]
                        loss1 = tf.losses.mean_squared_error(mel_gold, mel)

                        mel_postnet = tf.reshape(mel_postnet, [-1, decoder_size, mel_spec_size])
                        attention = tf.stack(attentions, axis=-1) # [B, T, T']
                        self.tower_mel_outputs.append(mel_postnet)
                        self.tower_attentions.append(attention)

                        loss2 = tf.losses.mean_squared_error(mel_gold, mel_postnet)

                        # mel to linear
                        if self.use_linear_spec is True:
                            n_conv1d_hiddens = [256, mel_spec_size]
                            if config["stop_gradient"]:
                                postnet_input = tf.stop_gradient(mel)
                            else:
                                postnet_input = mel_postnet

                            postnet_output = components.CBHG(postnet_input, K=8, n_conv1d_hiddens=n_conv1d_hiddens,
                                                         is_training=is_training)
                            postnet_output = tf.reshape(postnet_output, [-1, 256])

                            linear = components.linear_layer(postnet_output, 256, linear_spec_size, bias=True,
                                                         variable_scope="linear_project")
                            linear = tf.reshape(linear, [-1, decoder_size, linear_spec_size])
                            self.tower_linear_outputs.append(linear)

                            loss3 = tf.losses.mean_squared_error(self.tower_linear_spec[i], linear)

                            # mel to linear

                        for v in tf.trainable_variables():
                             if not ('bias' in v.name) and not ('stop-token-linear' in v.name):
                                 tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, v)

                        reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                        reg_term = tf.contrib.layers.apply_regularization(self.regularizer, reg_variables)

                        if self.use_linear_spec is True:
                            loss = loss1 + loss2 + reg_term + loss3
                        else:
                            loss = loss1 + loss2 + reg_term

                        self.tower_losses1.append(loss1)
                        self.tower_losses2.append(loss2)
                        if self.use_linear_spec is True:
                            self.tower_losses3.append(loss3)
                        self.tower_losses.append(loss)
                        self.tower_l2.append(reg_term)

                        # stop token grads
                        # stop_token_list = []
                        # mel_list = []
                        # for v in tf.trainable_variables():
                        #     if 'stop-token-linear' in v.name:
                        #         stop_token_list.append(v)
                        #     else:
                        #         mel_list.append(v)

                        #stop_token_grads_and_vars = self.stop_token_optimizer.compute_gradients(loss_stop_token, tf.get_variable("stop-token-var-list"))
                        # stop_token_grads_and_vars = self.stop_token_optimizer.compute_gradients(loss_stop_token, stop_token_list)
                        # self.tower_stop_token_grads_and_vars.append((stop_token_grads_and_vars))

                        # grads
                        ##grads_and_vars = self.optimizer.compute_gradients(loss, mel_list)
                        grads_and_vars = self.optimizer.compute_gradients(loss)
                        self.tower_grads_and_vars.append(grads_and_vars)

        print("computing averaged gradients")
        with tf.device("/cpu:0"):            
            self.mean_loss1 = tf.add_n(self.tower_losses1) / len(self.tower_losses1)
            self.mean_loss2 = tf.add_n(self.tower_losses2) / len(self.tower_losses2)
            if self.use_linear_spec is True:
                self.mean_loss3 = tf.add_n(self.tower_losses3) / len(self.tower_losses3)
            self.mean_loss = tf.add_n(self.tower_losses) / len(self.tower_losses)
            self.mean_l2 = tf.add_n(self.tower_l2) / len(self.tower_l2)

            self.mel_outputs = tf.concat(self.tower_mel_outputs, axis=0) # [B,= r*T', n_mel]
            if self.use_linear_spec is True:
                self.linear_outputs = tf.concat(self.tower_linear_outputs, axis=0)  # [B, r*T', n_linear]
            self.attentions = tf.concat(self.tower_attentions, axis=0) # [B, T, T']

            # compute averaged gradients
            self.mean_grads_and_vars = []
            for grads_and_vars in zip(*self.tower_grads_and_vars):
                grads = []
                for grad, var in grads_and_vars:
                    if grad is not None:
                        grads.append(tf.expand_dims(grad, 0))
                # print("len(grads)={}".format(len(grads)))
                if len(grads) == 0:
                    self.mean_grads_and_vars.append((None, grads_and_vars[0][1]))
                    continue
                mean_grad = tf.reduce_mean(tf.concat(grads, 0), 0)
                self.mean_grads_and_vars.append((mean_grad, var))

            # gradient clipping
            gradients = [grad for grad, var in self.mean_grads_and_vars]
            params = [var for grad, var in self.mean_grads_and_vars]
            clipped_gradients, norm = tf.clip_by_global_norm(gradients, hp.max_gradient_norm)

        self.train_op = self.optimizer.apply_gradients(zip(clipped_gradients, params),
                                                                     self.global_step)

        #with tf.device("/cpu:0"):
        #     self.mean_loss_stop_token = tf.add_n(self.tower_stop_token_losses) / len(self.tower_stop_token_losses)
        #
        #     self.stop_token_predicted_outputs = tf.concat(self.tower_stop_token_predicted_outputs, axis=0)  # [B,= r*T', n_mel]
        #
        #     self.stop_token_mean_grads_and_vars = []
        #     for stop_token_grads_and_vars in zip(*self.tower_stop_token_grads_and_vars):
        #         grads = []
        #         for grad, var in stop_token_grads_and_vars:
        #             print(var)
        #             if grad is not None:
        #                 grads.append(tf.expand_dims(grad, 0))
        #
        #         if len(grads) == 0:
        #             self.stop_token_mean_grads_and_vars.append((None, stop_token_grads_and_vars[0][1]))
        #             continue
        #         mean_grad = tf.reduce_mean(tf.concat(grads, 0), 0)
        #         self.stop_token_mean_grads_and_vars.append((mean_grad, var))
        #
        #     # gradient clipping
        #     st_gradients = [grad for grad, var in self.stop_token_mean_grads_and_vars]
        #     st_params = [var for grad, var in self.stop_token_mean_grads_and_vars]
        #     st_clipped_gradients, norm = tf.clip_by_global_norm(st_gradients, hp.max_gradient_norm)
        #
        # self.st_train_op = self.stop_token_optimizer.apply_gradients(zip(st_clipped_gradients, st_params))

        # summaries
        tf.summary.scalar("mean_loss1", self.mean_loss1)
        tf.summary.scalar("mean_loss2", self.mean_loss2)
        if self.use_linear_spec is True:
            tf.summary.scalar("mean_loss3", self.mean_loss3)
        tf.summary.scalar("mean_l2", self.mean_l2)
        tf.summary.scalar("mean_loss", self.mean_loss)
        ##tf.summary.scalar("mean_loss_stop_token", self.mean_loss_stop_token)
        tf.summary.scalar("learning_rate", self.learning_rate)
        self.merged_summary = tf.summary.merge_all()
        print("building model finished")

    #def seq2seq(self, encoder_inputs, decoder_inputs, tower_stop_token_outputs, feed_previous=False):
    def seq2seq(self, encoder_inputs, decoder_inputs, feed_previous=False):
        """Ensure to reuse_variables when this function is called the second time
        Args:
            encoder_inputs: character ids [[B]] * T
            decoder_inputs: mel-scale spectrogram, [[B, mel_spec_size]] * T'
        Returns:
            (outputs, alignments), where
                outputs: [shape [B, mel_spec_size]] * T'
                alignments: [[B, T]] * T'
        """
        char_vocab_size = self.config["char_vocab_size"]
        encoder_sequence_length = len(encoder_inputs)
        keep_prob = self.config["keep_prob"] if self.is_training else 1.0
        mel_spec_size = self.config["mel_spec_size"]
        encoder_conv_layers = self.config["encoder_conv_layers"]

        # use_stop_token = self.config["use_stop_token"]
        # if self.is_training:
        #     use_stop_token = False
        
        with tf.variable_scope("encoder"):

            # character embedding
            char_embeddings = components.get_weight_variable("char_embeddings", [char_vocab_size, hp.char_embed_size])
            input_embedding = tf.nn.embedding_lookup(char_embeddings, encoder_inputs) # shape [T, B, embed_size]

            # permute to shape [Batch, Time, embedding_size]
            input_embedding =  tf.transpose(input_embedding, perm=[1, 0, 2]) # shape [B, T, embed_size]
			
			#tf.Print("input_embedding shape")
			input_embedding = tf.Print(input_embedding, [input_embedding], message='debug batch')

            # 1d convolution
            conv_output = components.conv_stack(input_embedding, keep_probs=[keep_prob] * encoder_conv_layers, is_training=self.is_training)

            # bi-directional LSTM
            bi_lstm_output =  components.bidirectional_lstm(conv_output, is_training=self.is_training)

            attention_states = bi_lstm_output

        with tf.variable_scope("decoder"):

            self.decoder_cell = decoder.Tacotron2Decoder(attention_states, self.config,
                                                                       is_training=self.is_training)
            outputs = []
            alignments = []
            #stop_tokens = []
            previous_outputs = None
            batch_size_tensor = tf.shape(encoder_inputs[0])[0]
            cur_state = self.decoder_cell.zero_state(batch_size_tensor, tf.float32)
            pre_alignments = tf.one_hot(tf.zeros((batch_size_tensor,), tf.int32), encoder_sequence_length, dtype=tf.float32) # [B, T]

            n_predicted = 0

            for i, cur_inputs in enumerate(decoder_inputs):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                    if feed_previous:
                        cur_inputs = previous_outputs
                #cur_outputs, cur_state, alignment, stop_token_context = self.decoder_cell(cur_inputs, cur_state, pre_alignments)
                cur_outputs, cur_state, alignment = self.decoder_cell(cur_inputs, cur_state,
                                                                                          pre_alignments)
                outputs.append(cur_outputs)
                alignments.append(alignment)
                pre_alignments = alignment
                previous_outputs = cur_outputs

                # stop_token_batch, context_size = stop_token_context.get_shape().as_list()
                # stop_token_linear = components.linear_layer(stop_token_context, context_size, 1,
                #                                             bias=True,
                #                                             variable_scope="stop-token-linear")
                #
                # stop_token_sigmoid = tf.sigmoid(stop_token_linear, name="stop-token-sigmoid")
                #
                # stop_tokens.append(stop_token_sigmoid)
                #
                # # I would like to disable this to let the process smooth
                # # then we can output all decoder steps and a predicted N
                # if use_stop_token is True:
                #     if tf.cast(stop_token_sigmoid, tf.float32) > 0.9:
                #         if  n_predicted == 0:
                #             n_predicted = i + 1
                #   break

        #return outputs, alignments, stop_tokens, n_predicted
        return outputs, alignments

    #def step(self, session, char_ids, mel_spec, bucket_id, stop_token_indexes):
    def step(self, session, char_ids, mel_spec, bucket_id):
        """
        Args:
            session: tf.Session()
            char_ids: character ids, numpy array with shape [batch_size, encoder_size], dtype=tf.int32
            mel_spec: mel-scale spectrogram, numpy array with shape [batch_size, decoder_size, mel_spec_size], dtype=tf.float32
            bucket_id: bucket id
            n_mel: el frame count
        Returns:
            (predicted_mel, predicted_linear, attentions)
        """
        buckets = self.config["buckets"]
        encoder_size, decoder_size = buckets[bucket_id]

        batch_size = len(char_ids)
        # assert len(mel_spec) == batch_size
        # assert len(linear_spec) == batch_size

        encoder_inputs = char_ids
        decoder_outputs = [mel_spec[:, i:(i+1), :].reshape([batch_size, -1]) for i in range(decoder_size)]

        #stop_token_inputs = [stop_token_indexes[:, i:(i + 1), :].reshape([batch_size, -1]) for i in range(decoder_size)]

        if len(encoder_inputs[0]) != encoder_size:
            raise ValueError("Encoder inputs length must be equal to the one in bucket, {} != {}.".format(len(encoder_inputs), encoder_size))
        if len(decoder_outputs) != decoder_size:
            raise ValueError("Decoder outputs length must be equal to the one in bucket, {} != {}.".format(len(decoder_outputs), decoder_size))

        input_feeds = {} # input feeds that the graph needed
        output_feeds = {} # outputs that the graph should compute
        # encoder input feeds
        for i in range(encoder_size):
            input_feeds[self.encoder_inputs[i].name] = encoder_inputs[:, i]
        # decoder output feeds
        for i in range(decoder_size):
            input_feeds[self.decoder_outputs[i].name] = decoder_outputs[i]
        # mel and linear spec feeds

        #for i in range(decoder_size):
        #    input_feeds[self.stop_token_outputs[i].name] = stop_token_inputs[i]

        #output_feeds = [self.mel_outputs, self.attentions, self.stop_token_predicted_outputs] # outputs and attentions
        if self.use_linear_spec is True:
            output_feeds = [self.mel_outputs, self.attentions]  # outputs and attentions
        else:
            output_feeds = [self.mel_outputs, self.attentions]  # outputs and attentions

        # output_feeds += [self.mean_loss1, self.mean_loss2, self.mean_loss] # losses
        if self.is_training:
            ##output_feeds += [self.mean_loss, self.merged_summary, self.train_op, self.st_train_op]
            output_feeds += [self.mean_loss, self.merged_summary, self.train_op]
        elif self.gpu_GLA:
            output_feeds += [self.wav_outputs]

        outputs = session.run(output_feeds, input_feeds)

        return outputs

    # update learning rate according to global steps
    def update_learning_rate(self, session, lr=None, scheme="manual"):
        """scheme:
            manual: update learning rates manually
            noam: https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/trainer_utils.py
            renoam: rectified noam
        """
        if lr is not None:
            feed_dict = {self.learning_rate_placeholder: lr}
            session.run(self.update_learning_rate_op, feed_dict=feed_dict)
            return
        cur_global_step = session.run(self.global_step) + 1.0
        cur_learning_rate = None
        scheme = scheme.lower()
        if scheme == "manual":
            for min_step, lr in self.config["learning_rates"]:
                if cur_global_step >= min_step*1000:
                    cur_learning_rate = lr
            if cur_learning_rate <= session.run(self.learning_rate):
                feed_dict = {self.learning_rate_placeholder: cur_learning_rate}
                session.run(self.update_learning_rate_op, feed_dict=feed_dict)
        elif scheme == "noam":
            cur_learning_rate = self.init_lr * min(cur_global_step / self.warmup_steps, (self.warmup_steps/cur_global_step) ** 0.5)
            feed_dict = {self.learning_rate_placeholder: cur_learning_rate}
            session.run(self.update_learning_rate_op, feed_dict=feed_dict)
        elif scheme == "renoam":
            cur_learning_rate = self.init_lr * min(1.0, (self.warmup_steps/cur_global_step)**0.5) # keep initial learning rate unchanged during warmup steps
            feed_dict = {self.learning_rate_placeholder: cur_learning_rate}
            session.run(self.update_learning_rate_op, feed_dict=feed_dict)
        elif scheme == "exponentially":
            # a learning rate of 10−3
            # exponentially decaying to 10−5
            # starting after
            # 50,000 iterations
            cur_learning_rate = self.init_lr
            if cur_global_step > self.warmup_steps:
                exp_learning_rate = tf.train.exponential_decay(self.init_lr, cur_global_step,
                                                           self.warmup_steps, 0.96)
                cur_learning_rate = max(exp_learning_rate, 0.00001)

            feed_dict = {self.learning_rate_placeholder: cur_learning_rate}
            session.run(self.update_learning_rate_op, feed_dict=feed_dict)
        else:
            raise Exception("unsupported schem {}".format(scheme))

    def update_global_step(self, session, gs):
        feed_dict = {self.global_step_placeholder: gs}
        session.run(self.update_global_step_op, feed_dict)
