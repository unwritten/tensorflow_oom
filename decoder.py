# coding=utf-8

import components
import attention
import tensorflow as tf
import zoneout_lstm

RNNCell = tf.contrib.rnn.RNNCell
GRUCell = tf.contrib.rnn.GRUCell

class Tacotron2Decoder(RNNCell):
    """Use location sensitive attention attention mechanism"""
    def __init__(self, attention_states, config, variable_scope="Tacotron2DecoderCell", is_training=True):
        self._attention_states = attention_states
        self.config = config
        self._mel_spec_size = config["mel_spec_size"]

        lstm_layers = config["decoder_cell_lstm_layers"]
        lstm_units = config["decoder_cell_lstm_units"]
        lstm_hiddens = [lstm_units] * lstm_layers
        prenet_layers = config["decoder_prenet_layers"]
        prenet_units = config["decoder_prenet_units"]

        # In order to introduce output
        # variation at inference time, dropout with probability 0.5 is applied
        # only to layers in the pre-net of the autoregressive decoder
        keep_prob = config["keep_prob"]

        attention_kernel_size = config["attention_kernel_size"]
        attention_filter_length = config["attention_filter_length"]

        batch_size, self.attention_length, self.attention_state_size = attention_states.get_shape().as_list()

        with tf.variable_scope(variable_scope):

            # prenet
            self._prenet = components.PreNet(self._mel_spec_size, n_hiddens=[prenet_units] * prenet_layers, keep_probs=[keep_prob] * prenet_layers)

            # attention mechanism
            self._attention_mechanism = attention.ChorowskiAttention(
                attention_kernel_size,
                attention_filter_length,
                self.attention_state_size,
                attention_states)

            # uni-directional LSTM
            self._lstm_cells = []
            for i, n_hidden in enumerate(lstm_hiddens):
                prefix = "lstm_layer_{}".format(i)
                with tf.variable_scope(prefix):
                    lstm_cell = zoneout_lstm.ZoneoutWrapper(
                        tf.nn.rnn_cell.LSTMCell(n_hidden, use_peepholes=True, state_is_tuple=True), is_training=is_training)
                    self._lstm_cells.append((lstm_cell))

            # attention RNN cell
            cell = GRUCell(256)

            # attention cell
            self._attention_rnn_cell = attention.LocationSensitiveAttentionWrapper(cell,
                self._lstm_cells, self._attention_mechanism, is_training=is_training)

        self.call_counter = 0

    @property
    def state_size(self):
        # attention rnn state comes first
        return sum([cell.state_size for cell in self._decoder_rnn_cells]) + 256

    @property
    def output_size(self):
        return self.config["reduce_factor"] * self.config["mel_spec_size"]

    def zero_state(self, batch_size, dtype=tf.float32):
        with tf.name_scope(type(self).__name__ + "zero_state", values=[batch_size]):
            state = []
            state.append(self._attention_rnn_cell.zero_state(batch_size, dtype))
            state += [cell.zero_state(batch_size, dtype) for cell in self._lstm_cells]
            return state

    def __call__(self, inputs, state, pre_alignments, variable_scope="Tacotron2DecoderCell"):
        """Run cell on inputs.
        Args:
            inputs: tensor with shape [batch_size, n_input]
            state: tensor with shape [batch_size, self.state_size]
        Returns:
            tuple: (output, state), where,
                output.shape = [batch_size, self.output_size]
                state.shape = [batch_size, self.state_size]
        """

        self.call_counter += 1

        with tf.variable_scope(variable_scope):

            # prenet
            prenet_output = self._prenet(inputs)

            # attention rnn
            attention_rnn_outputs, new_attention_rnn_state, context, alignments = self._attention_rnn_cell(prenet_output, state, pre_alignments)

            # concatinate LSTM output and context vector
            linear_input = tf.concat([attention_rnn_outputs[-1], context], axis=1)

            stop_token_input = linear_input

            linear_input_shape = linear_input.get_shape().as_list()
            batch_size, n_input = linear_input_shape

            # linear transform to produce a prediction of the target mel spectrogram frame
            linear_output = components.linear_layer(linear_input, n_input, self._mel_spec_size, bias=True, variable_scope="linear")

        #return linear_output, new_attention_rnn_state, alignments, stop_token_input
        return linear_output, new_attention_rnn_state, alignments