# coding=utf-8

from __future__ import print_function
from __future__ import division

import tensorflow as tf
import zoneout_lstm

GRUCell = tf.contrib.rnn.GRUCell

def get_weight_variable(name, shape, initializer=None, regularizer=None):
    if initializer is None:
        if len(shape) == 2:
            initializer = tf.contrib.layers.xavier_initializer()
        else:    
            initializer = tf.contrib.layers.xavier_initializer_conv2d()
    variable = tf.get_variable(name, shape, initializer=initializer, regularizer=regularizer)
    return variable

def get_bias_variable(name, shape, initializer=None, regularizer=None):
    initializer = initializer or tf.constant_initializer(0.0) # 0.1?
    variable = tf.get_variable(name, shape, initializer=initializer, regularizer=regularizer)
    return variable

def linear_layer(x, n_input, n_output, bias=True, variable_scope="linear"):
    """
    perform a linear transformation on x
    Args:
        x: [batch_size, n_input]
        n_input: input size
        n_output: output size
        bias: use bias or not
    Returns:
        y: a tensor with shape [batch_size, n_output]
    """
    with tf.variable_scope(variable_scope):
        W = get_weight_variable("W", [n_input, n_output])
        x_shape = x.get_shape().as_list()
        assert x_shape[-1] == n_input

        y = tf.matmul(x, W)
        if bias:
            b = get_bias_variable("b", [n_output])
            y = y + b
    return y

def batch_norm(x, decay=0.999, is_training=True, variable_scope="bn"):
    """Batch normalization
    Args:
        x: input tensor, shape [B, ...], x rank in [2, 3, 4]
        is_training: is training or not
    Returns:
        tensor with the same shape as x
    """
    with tf.variable_scope(variable_scope):
        x_rank = x.get_shape().ndims
        assert x_rank>=2 and x_rank<=4
        if x_rank == 2:
            x = tf.expand_dims(x, axis=1)
            x = tf.expand_dims(x, axis=2)
        elif x_rank == 3:
            x = tf.expand_dims(x, axis=1)

        y = tf.contrib.layers.batch_norm(
            inputs=x,
            decay=decay,
            center=True,
            scale=True,
            activation_fn=None,
            updates_collections=None,
            is_training=is_training,
            zero_debias_moving_mean=True,
            fused=True)
        if x_rank == 2:
            y = tf.squeeze(y, axis=[1, 2])
        elif x_rank == 3:
            y = tf.squeeze(y, axis=1)
    return y

def conv1d(x, kernel_size, n_out_channels, is_training=True, bn=True, use_bias=False, variable_scope="conv1d"):
    """1D convolution
    Args:
      x: shape [B, T, n_in_channels]
      kernel_size: int, kernel size (or window size)
      n_out_channels: An int. Output dimension.
      is_training: A boolean. This is passed to an argument of `batch_normalize`.
      bn: A boolean. If True, `batch_normalize` is applied.
      act: A boolean. If True, `ReLU` is applied.
      use_bias: A boolean. If True, units for bias are added.
      
    Returns
      A 3d tensor with shape of [batch_size, length, n_out_channels]
    """
    params = {
        "inputs" : x,
        "filters" : n_out_channels,
        "kernel_size" : kernel_size,
        "dilation_rate" : 1,
        "padding" : "SAME",
        "activation" : None,
        # "kernel_initializer":tf.contrib.layers.xavier_initializer(),
        "kernel_initializer" : tf.truncated_normal_initializer(stddev=0.01),
        "use_bias" : use_bias}

    with tf.variable_scope(variable_scope):                     
        y = tf.layers.conv1d(**params)
        if bn:
            y = batch_norm(y, is_training=is_training)
    return y

def conv2d(x, kernel_shape, strides, padding, variable_scope="conv2d"):
    """ 2D convolution on CPU
    Args:
        x: input tensor with shape [batch_size, width, height, n_in_channels]
        kernel_shape: filter size [kernel_width, kernel_height, n_in_channels, n_out_channels]
        strides: strides of convolution
        padding: either "same" or "valid", case insensitive
    Returns:
        tensor with shape: [batch_size, width, height, n_out_channels]
    """
    with tf.variable_scope(variable_scope):
        W = get_weight_variable("W", kernel_shape)
        b = get_bias_variable("b", kernel_shape[-1:])
        y = tf.nn.conv2d(x, filter=W, strides=strides, padding=padding) + b
    return y

class PreNet(object):
    def __init__(self, n_input, n_hiddens=[256, 256], keep_probs=[0.5, 0.5], activations=None, variable_scope="PreNet"):
        """
        Args:
            n_hiddens: number of units of hidden layers, example: [256, 128]
            keep_probs: keep probabilities of hidden layers, example: [0.5, 0.5]
            activations: activation functions of hidden layers, example: [tf.nn.relu, tf.nn.relu]
        Returns:
            tensor with shape [batch_size, n_output=n_hiddens[-1]]
        """
        print("Composing", variable_scope)
        self.n_input = n_input
        self.n_hiddens = n_hiddens
        self.keep_probs = keep_probs or [0.5] * len(n_hiddens)
        self.activations = activations or ([tf.nn.relu] * len(n_hiddens))
    
        assert len(self.activations) == len(n_hiddens)
        assert len(self.keep_probs) == len(n_hiddens)    
    
        self.Ws = []
        self.bs = []
        with tf.variable_scope(variable_scope):
            for i, n_output in enumerate(n_hiddens):
                prefix = "layer_{}".format(i)
                with tf.variable_scope(prefix):
                    self.Ws.append(get_weight_variable("W", [n_input, n_output]))
                    self.bs.append(get_bias_variable("b", [n_output]))
                n_input = n_output

    def __call__(self, x):
        """
        Args:
            x: input tensor with shape [batch_size, n_input]
        """
        assert x.get_shape().as_list()[-1] == self.n_input

        for i, (W, b, keep_prob, activation) in enumerate(zip(self.Ws, self.bs, self.keep_probs, self.activations)):
            x = tf.matmul(x, W) + b
            x = tf.nn.dropout(activation(x), keep_prob=self.keep_probs[i])
        return x

def conv_stack(x, n_conv1d_hiddens=[512, 512, 512], keep_probs=[0.5, 0.5, 0.5], kernel_size=5, activations=tf.nn.relu,
               last_linear=False, is_training=True, variable_scope="ConvStack"):
    """convolution stack with shape 5 * 1

    Args:
        x: input tensor with shape [B, T, n_input]
        K: maximum window size of stride-1 convolution banks, CBHG performs convolutions of window sizes from 1 to K
        n_conv1d_hiddens: n_hiddens for conv1d layers, [128, 128] for encoder CBHG, and [256, 128] for postnet CBHG
        n_out_channels: number of output channels of convolution layers
        n_highway_layers: number of highway layers
        n_gru_units: number of units of bidirectional GRU
        variable_scope: variable scope
    Returns
        a tensor of shape [batch_size, T, n_output=2*n_gru_units]
    """
    print("Composing", variable_scope)

    with tf.variable_scope(variable_scope):
        x_shape = x.get_shape().as_list()  # tensor: [B, T, n_input]
        batch_size, sequence_length, n_input = x_shape
        print("x_shape ", x_shape)
        # assert n_input == n_conv1d_hiddens[-1]  # ensure residual connectoins can be added

        conv = x
        n_conv = len(n_conv1d_hiddens)
        for i, conv_out_channels in enumerate(n_conv1d_hiddens):
            # 1D convolution , 512 filters, length 5 default
            conv = conv1d(conv, kernel_size, conv_out_channels, is_training=is_training, bn=False, use_bias=True,
                          variable_scope="conv1d_stack_layer_{}".format(i))

            # batch normalization
            conv = batch_norm(conv, is_training=is_training, variable_scope="bn_conv1d_stack_layer-{}".format(i))

            # nonlineariry: ReLU or tanh
            if(i == n_conv -1):
                if (last_linear == False):
                    conv = activations(conv)
            else:
                conv = activations(conv)

            # regularize with dropout
            # to-do:
            # when last layer is linear, should we still use dropout or just skip it?
            if is_training:
                conv = tf.nn.dropout(conv, keep_prob=keep_probs[i])

        print("result.shape =", conv.get_shape().as_list())
        return conv

def bidirectional_lstm(x, hidden_units=256, is_training=True, variable_scope="bi-lstm"):
    """bidirectional lstm with zoneout regularizer

    Args:
        x: input tensor with shape [B, T, n_input]
        K: maximum window size of stride-1 convolution banks, CBHG performs convolutions of window sizes from 1 to K
        n_conv1d_hiddens: n_hiddens for conv1d layers, [128, 128] for encoder CBHG, and [256, 128] for postnet CBHG
        n_out_channels: number of output channels of convolution layers
        n_highway_layers: number of highway layers
        n_gru_units: number of units of bidirectional GRU
        variable_scope: variable scope
    Returns
        a tensor of shape [batch_size, T, n_output=2*n_gru_units]
    """
    print("Composing", variable_scope)

    with tf.variable_scope(variable_scope):
        x_shape = x.get_shape().as_list()  # tensor: [B, T, n_input]
        batch_size, sequence_length, n_input = x_shape
        print("x_shape ", x_shape)

        lstm_cell_forward = zoneout_lstm.ZoneoutWrapper(
            tf.nn.rnn_cell.LSTMCell(hidden_units,
                                use_peepholes= True, state_is_tuple=True), is_training=is_training)
        lstm_cell_backward = zoneout_lstm.ZoneoutWrapper(
            tf.nn.rnn_cell.LSTMCell(hidden_units,
                                use_peepholes= True, state_is_tuple=True), is_training=is_training)

        # bigru_outputs = (outputs_forward, outputs_backward), where both outputs_forward and outputs_backward have shape [batch_size, sequence_length, cell.output_size]
        outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_cell_forward,
                                                                      cell_bw=lstm_cell_backward, inputs=x,
                                                                      dtype=tf.float32)

        result = tf.concat(outputs, axis=-1) # shape [B, T, 2*n_gru_units]
        print("result.shape =", result.get_shape().as_list())
        return result

def unidirectional_lstm_stack(x, n_hiddens=[1024, 1024],
         is_training=True, variable_scope="unidirectional_lstm_stack"):
    """ 2 uni-directional LSTM layers with 1024 units

    Args:
        x: input tensor with shape [B, T, n_input]
        K: maximum window size of stride-1 convolution banks, CBHG performs convolutions of window sizes from 1 to K
        n_conv1d_hiddens: n_hiddens for conv1d layers, [128, 128] for encoder CBHG, and [256, 128] for postnet CBHG
        n_out_channels: number of output channels of convolution layers
        n_highway_layers: number of highway layers
        n_gru_units: number of units of bidirectional GRU
        variable_scope: variable scope
    Returns
        a tensor of shape [batch_size, T, n_output=2*n_gru_units]
    """
    print("Composing", variable_scope)

    with tf.variable_scope(variable_scope):
        x_shape = x.get_shape().as_list()  # tensor: [B, T, n_input]
        batch_size, sequence_length, n_input = x_shape
        print("x_shape ", x_shape)

        input = x
        for i, n_hidden in enumerate(n_hiddens):
            prefix = "layer_{}".format(i)
            with tf.variable_scope(prefix):
                lstm_cell = zoneout_lstm.ZoneoutWrapper(
                    tf.nn.rnn_cell.LSTMCell(n_hidden, use_peepholes=True, state_is_tuple=True))
                outputs, states = tf.nn.dynamic_rnn(lstm_cell, input, dtype=tf.float32)
                input = outputs

        result = input
        print("result.shape =", result.get_shape().as_list())
        return result

def max_pooling1d(x, pool_size, strides, padding="SAME", variable_scope="max_pooling1d"):
    with tf.variable_scope(variable_scope):
        max_pooled = tf.layers.max_pooling1d(x, pool_size, strides, padding="SAME")
    return max_pooled


def highway(x, n_layers=4, variable_scope="highway"):
    """
    Args:
        x: input tensor with shape [batch_size, n_input]
        n_layers: number of highway layers
    Returns:
        a tensor with the same shape as x
    """
    x_shape = x.get_shape().as_list()
    assert len(x_shape) == 2
    batch_size, n_input = x_shape

    with tf.variable_scope(variable_scope):
        for i in range(n_layers):
            prefix = "layer_{}".format(i)
            with tf.variable_scope(prefix):
                h_x = tf.nn.relu(linear_layer(x, n_input, n_input, bias=True, variable_scope="linear"))  # hidden units

                gate = tf.nn.sigmoid(linear_layer(x, n_input, n_input, bias=True, variable_scope="gate"))
                y = h_x * gate + x * (1.0 - gate)
                x = y

    return x

def CBHG(x, K=16, n_conv1d_hiddens=[128, 128], n_out_channels=128, n_highway_layers=4, n_gru_units=128,
         is_training=True, variable_scope="CBHG"):
    """
    Args:
        x: input tensor with shape [B, T, n_input]
        K: maximum window size of stride-1 convolution banks, CBHG performs convolutions of window sizes from 1 to K
        n_conv1d_hiddens: n_hiddens for conv1d layers, [128, 128] for encoder CBHG, and [256, 128] for postnet CBHG
        n_out_channels: number of output channels of convolution layers
        n_highway_layers: number of highway layers
        n_gru_units: number of units of bidirectional GRU
        variable_scope: variable scope
    Returns
        a tensor of shape [batch_size, T, n_output=2*n_gru_units]
    """
    print("Composing", variable_scope)

    with tf.variable_scope(variable_scope):
        x_shape = x.get_shape().as_list()  # tensor: [B, T, n_input]
        batch_size, sequence_length, n_input = x_shape
        print("x_shape ", x_shape)
        assert n_input == n_conv1d_hiddens[-1]  # ensure residual connectoins can be added

        convs = []
        # 1D convolutional banks
        for i in range(1, K + 1):
            conv = conv1d(x, i, n_out_channels, is_training=is_training, bn=False, use_bias=True,
                          variable_scope="conv1d-banks-{}".format(i))
            conv_act = tf.nn.relu(conv)  # [B, T, n_out_channels]
            convs.append(conv_act)

        # stack convolution outputs together
        convs_stacked = tf.concat(convs, axis=-1)  # shape: [batch_size, sequence_length, n_out_channels*K]
        convs_stacked = batch_norm(convs_stacked, is_training=is_training, variable_scope="bn-banks")
        print("convs_stacked.shape :", convs_stacked.get_shape().as_list())

        # max pooling along time, window size is fixed to 2
        max_pooled = max_pooling1d(convs_stacked, pool_size=2, strides=1, padding="SAME")  # [B, T, n_out_channels*K]
        print("max_pooled.shape ", max_pooled.get_shape().as_list())

        conv = max_pooled
        conv_in_channels = n_out_channels * K
        for i, conv_out_channels in enumerate(n_conv1d_hiddens):
            conv = conv1d(conv, 3, conv_out_channels, is_training=True, bn=False, use_bias=True,
                          variable_scope="conv1d_layer_{}".format(i))
            if i < len(n_conv1d_hiddens) - 1:  # linear for the last layer
                conv = tf.nn.relu(conv)
            conv = batch_norm(conv, is_training=is_training, variable_scope="bn-conv1d-layer-{}".format(i))
            conv_in_channels = conv_out_channels

        # residual
        highway_input = x + conv  # shape [B, T, n_input]

        # highway
        highway_input = tf.reshape(highway_input, [-1, n_input])
        highway_output = highway(highway_input, n_layers=4)  # [batch_size*sequence_length, n_input]

        # Bidirectional GRU
        gru_input = tf.reshape(highway_output, [-1, sequence_length, n_input])
        print("gru_input.shape =", gru_input.get_shape().as_list())

        # TODO: to place variables of GRUCell on CPU, but ops on GPU?
        gru_cell_forward = GRUCell(n_gru_units)
        gru_cell_backward = GRUCell(n_gru_units)

        # bigru_outputs = (outputs_forward, outputs_backward), where both outputs_forward and outputs_backward have shape [batch_size, sequence_length, cell.output_size]
        bigru_outputs, bigru_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=gru_cell_forward,
                                                                      cell_bw=gru_cell_backward, inputs=gru_input,
                                                                      dtype=tf.float32)

        result = tf.concat(bigru_outputs, axis=-1)  # shape [B, T, 2*n_gru_units]
        print("result.shape =", result.get_shape().as_list())
        return result
