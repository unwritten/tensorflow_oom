# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A powerful dynamic attention wrapper object.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import tensorflow as tf

from tensorflow.contrib.rnn import RNNCell
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.layers import base as layers_base
from tensorflow.python.layers import core as layers_core
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest

import components

_zero_state_tensors = rnn_cell_impl._zero_state_tensors  # pylint: disable=protected-access


class AttentionMechanism(object):
  pass


def _prepare_memory(memory, memory_sequence_length, check_inner_dims_defined):
  """Convert to tensor and possibly mask `memory`.

  Args:
    memory: `Tensor`, shaped `[batch_size, max_time, ...]`.
    memory_sequence_length: `int32` `Tensor`, shaped `[batch_size]`.
    check_inner_dims_defined: Python boolean.  If `True`, the `memory`
      argument's shape is checked to ensure all but the two outermost
      dimensions are fully defined.

  Returns:
    A (possibly masked), checked, new `memory`.

  Raises:
    ValueError: If `check_inner_dims_defined` is `True` and not
      `memory.shape[2:].is_fully_defined()`.
  """
  memory = nest.map_structure(
      lambda m: ops.convert_to_tensor(m, name="memory"), memory)
  if check_inner_dims_defined:
    def _check_dims(m):
      if not m.get_shape()[2:].is_fully_defined():
        raise ValueError("Expected memory %s to have fully defined inner dims, "
                         "but saw shape: %s" % (m.name, m.get_shape()))
    nest.map_structure(_check_dims, memory)
  if memory_sequence_length is None:
    seq_len_mask = None
  else:
    seq_len_mask = array_ops.sequence_mask(
        memory_sequence_length,
        maxlen=array_ops.shape(nest.flatten(memory)[0])[1],
        dtype=nest.flatten(memory)[0].dtype)
  def _maybe_mask(m, seq_len_mask):
    rank = m.get_shape().ndims
    rank = rank if rank is not None else array_ops.rank(m)
    extra_ones = array_ops.ones(rank - 2, dtype=dtypes.int32)
    if memory_sequence_length is not None:
      seq_len_mask = array_ops.reshape(
          seq_len_mask,
          array_ops.concat((array_ops.shape(seq_len_mask), extra_ones), 0))
      return m * seq_len_mask
    else:
      return m
  return nest.map_structure(lambda m: _maybe_mask(m, seq_len_mask), memory)


class _BaseAttentionMechanism(AttentionMechanism):
  """A base AttentionMechanism class providing common functionality.

  Common functionality includes:
    1. Storing the query and memory layers.
    2. Preprocessing and storing the memory.
  """

  def __init__(self, query_layer, memory, memory_sequence_length=None,
               memory_layer=None, check_inner_dims_defined=True, name=None):
    """Construct base AttentionMechanism class.

    Args:
      query_layer: Callable.  Instance of `tf.layers.Layer`.  The layer's depth
        must match the depth of `memory_layer`.  If `query_layer` is not
        provided, the shape of `query` must match that of `memory_layer`.
      memory: The memory to query; usually the output of an RNN encoder.  This
        tensor should be shaped `[batch_size, max_time, ...]`.
      memory_sequence_length (optional): Sequence lengths for the batch entries
        in memory.  If provided, the memory tensor rows are masked with zeros
        for values past the respective sequence lengths.
      memory_layer: Instance of `tf.layers.Layer` (may be None).  The layer's
        depth must match the depth of `query_layer`.
        If `memory_layer` is not provided, the shape of `memory` must match
        that of `query_layer`.
      check_inner_dims_defined: Python boolean.  If `True`, the `memory`
        argument's shape is checked to ensure all but the two outermost
        dimensions are fully defined.
      name: Name to use when creating ops.
    """
    if (query_layer is not None
        and not isinstance(query_layer, layers_base.Layer)):  # pylint: disable=protected-access
      raise TypeError(
          "query_layer is not a Layer: %s" % type(query_layer).__name__)
    if (memory_layer is not None
        and not isinstance(memory_layer, layers_base.Layer)):  # pylint: disable=protected-access
      raise TypeError(
          "memory_layer is not a Layer: %s" % type(memory_layer).__name__)
    self._query_layer = query_layer
    self._memory_layer = memory_layer
    with ops.name_scope(
        name, "BaseAttentionMechanismInit", nest.flatten(memory)):
      self._values = _prepare_memory(
          memory, memory_sequence_length,
          check_inner_dims_defined=check_inner_dims_defined)
      self._keys = (
          self.memory_layer(self._values) if self.memory_layer  # pylint: disable=not-callable
          else self._values)
      temp = self._keys

  @property
  def memory_layer(self):
    return self._memory_layer

  @property
  def query_layer(self):
    return self._query_layer

  @property
  def values(self):
    return self._values

  @property
  def keys(self):
    return self._keys

class ChorowskiAttention(_BaseAttentionMechanism):
  """Implements Chorowski-style (location sensitive) attention.

  Basiclly, it extends the attention in:

  Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio.
  "Neural Machine Translation by Jointly Learning to Align and Translate."
  ICLR 2015. https://arxiv.org/abs/1409.0473

  with convolution of previous alignment to achieve location sensitive attention in:

  Jan Chorowski, Dzmitry Bahdanau, Dmitriy Serdyuk, Kyunghyun Cho, Yoshua Bengio.
  "Attention-Based Models for Speech Recognition"
  NIPS 2014, http://papers.nips.cc/paper/5847-attention-based-models-for-speech-recognition.pdf
  """

  def __init__(self, kernel_size, filter_length, num_units, memory, memory_sequence_length=None, name="ChorowskiAttention"):
    """Construct the Attention mechanism.

    Args:
      num_units: The depth of the query mechanism.
      memory: The memory to query; usually the output of an RNN encoder.  This
        tensor should be shaped `[batch_size, max_time, ...]`.
      memory_sequence_length (optional): Sequence lengths for the batch entries
        in memory.  If provided, the memory tensor rows are masked with zeros
        for values past the respective sequence lengths.
      name: Name to use when creating ops.
    """
    super(ChorowskiAttention, self).__init__(
    # in the original formula, W and V are just matrix, here set use_bias as False explicitly
    # which is different with the previous implementation on Bahdanau attention
        query_layer=layers_core.Dense(num_units, name="query_layer", use_bias=False),
        memory_layer=layers_core.Dense(num_units, name="memory_layer", use_bias=False),
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        name=name)
    self._kernel_size = kernel_size
    self._filter_length = filter_length
    self._alignment_layer = layers_core.Dense(num_units, name="alignment_layer", use_bias=False),
    self._num_units = num_units
    self._name = name

  def alignment_layer(self, input):
    return self._alignment_layer[0](input)

  def __call__(self, query, prev_alignment):
    """Score the query based on the encoder hidden state, alignment of last time step,
       decoder state of last time step

    Args:
      query: Tensor of dtype matching `self.values` and shape
        `[batch_size, query_depth]`.
    Returns:
      score: Tensor of dtype matching `self.values` and shape
        `[batch_size, T]`.
    """
    with ops.name_scope(None, "ChorowskiAttention", [query]):

      # W * S(i-1)
      processed_query = self.query_layer(query) if self.query_layer else query
      dtype = processed_query.dtype

      # Reshape from [batch_size, ...] to [batch_size, 1, ...] for broadcasting.
      processed_query_expanded = array_ops.expand_dims(processed_query, 1)

      # Reshape from [batch_size, memory_time] to [batch_size, 1, memory_time]
      alignments = array_ops.expand_dims(prev_alignment, 1)
      alignments_transposed = tf.transpose(alignments, perm=[0, 2, 1])  # shape [B, T, embed_size]

      # f(i) = F * a(i-1)
      conv = components.conv1d(alignments_transposed, self._filter_length, self._kernel_size, bn=False, use_bias=False,
                                    variable_scope="conv1d")

      # conv_squeezed = array_ops.squeeze(conv, [1])
      conv_reshape = tf.reshape(conv, [-1, self._kernel_size])

      # U * f(i)
      projected_alignment = self.alignment_layer(conv_reshape)

      # reshape to [batch_size, T, size]
      key_shape = self.keys.get_shape().as_list()  # tensor: [B, T, n_input]
      batch_size, sequence_length, n_input = key_shape
      location = tf.reshape(projected_alignment, [-1, sequence_length, n_input])

      # Bias added prior to the nonlinearity
      b = variable_scope.get_variable(
          "attention_b", [self._num_units], dtype=dtype,
          initializer=init_ops.zeros_initializer())

    score = math_ops.reduce_sum(
        math_ops.tanh(self.keys + processed_query_expanded + location + b), [2])

    return score # [B, T]

class DynamicAttentionWrapperState(
    collections.namedtuple(
        "DynamicAttentionWrapperState", ("cell_state", "attention"))):
  """`namedtuple` storing the state of a `DynamicAttentionWrapper`.

  Contains:

    - `cell_state`: The state of the wrapped `RNNCell`.
    - `attention`: The attention emitted at the previous time step.
  """
  pass

class LocationSensitiveAttentionWrapper(RNNCell):
  """Wraps another `RNNCell` with attention.
  """

  def __init__(self,
               rnn_cell,
               cells,
               attention_mechanism,
               attention_size = 256,
               is_training=True,
               cell_input_fn=None,
               probability_fn=None,
               sigmoid_noise_std_dev=1.0,
               output_attention=True,
               name=None):
    """Construct the `DynamicAttentionWrapper`.

    Args:
      cell: An instance of `RNNCell`.
      attention_mechanism: An instance of `AttentionMechanism`.
      attention_size: Python integer, the depth of the attention (output)
    """
    for cell in cells:
        if not isinstance(cell, RNNCell):
          raise TypeError(
              "cell must be an RNNCell, saw type: %s" % type(cell).__name__)
    if not isinstance(attention_mechanism, AttentionMechanism):
      raise TypeError(
          "attention_mechanism must be a AttentionMechanism, saw type: %s"
          % type(attention_mechanism).__name__)
    if cell_input_fn is None:
      cell_input_fn = (
          lambda inputs, attention: array_ops.concat([inputs, attention], -1))
    else:
      if not callable(cell_input_fn):
        raise TypeError(
            "cell_input_fn must be callable, saw type: %s"
            % type(cell_input_fn).__name__)
    if probability_fn is None:
      probability_fn = nn_ops.softmax
    else:
      if not callable(probability_fn):
        raise TypeError(
            "probability_fn must be callable, saw type: %s"
            % type(probability_fn).__name__)

    self._cell = rnn_cell
    self._cells = cells
    self._attention_mechanism = attention_mechanism
    self._attention_size = attention_size
    self._cell_input_fn = cell_input_fn
    self._probability_fn = probability_fn
    self._attention_size = attention_size
    self._attention_layer = layers_core.Dense(
        attention_size, bias_initializer=None)
    self.is_training = is_training
    self._sigmoid_noise_std_dev = sigmoid_noise_std_dev

  @property
  def output_size(self):
    return self._attention_size

  @property
  def state_size(self):
    return DynamicAttentionWrapperState(
        cell_state=self._cells.state_size,
        attention=self._attention_size)

  def zero_state(self, batch_size, dtype):
    with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
      return DynamicAttentionWrapperState(
          cell_state=self._cell.zero_state(batch_size, dtype),
          attention=_zero_state_tensors(
              self._attention_size, batch_size, dtype))

  def __call__(self, inputs, states, prev_alignment, scope=None):
    """Perform a step of attention-wrapped RNN.

    Args:
      inputs: (Possibly nested tuple of) Tensor, the input at this time step.
      state: An instance of `DynamicAttentionWrapperState` containing
        tensors from the previous time step.
      scope: Must be `None`.

    Returns:
      A tuple `(attention, next_state)`, where:

      - `attention` is the attention passed to the layer above.
      - `next_state` is an instance of `DynamicAttentionWrapperState`
         containing the state calculated at this time step.

    Raises:
      NotImplementedError: if `scope` is not `None`.
    """
    if scope is not None:
      raise NotImplementedError("scope not None is not supported")

    # get the first last time step's state as the attention input state of last time step
    prev_input_state = states[0]

    # concat prenet output and last step's attention as attention rnn cell's input
    cell_inputs = self._cell_input_fn(inputs, prev_input_state.attention)
    cell_state = prev_input_state.cell_state

    cell_output, next_cell_state = self._cell(cell_inputs, cell_state)

    # step1: generate energy using location sensitive attention
    # using previous state, previous alignment
    # energy = self._attention_mechanism(c_prev, prev_alignment)
    score = self._attention_mechanism(cell_output, prev_alignment)

    # this is a naive implementation, instead I used the rensorflow source like style as below
    # step2: calculate alignment with energy using softmax
    # alignments = self._probability_fn(score)
    # test

    if self.is_training: # add noise before sigmoid during training to encourage discrete alignments
      score = score + self._sigmoid_noise_std_dev * tf.random_normal(tf.shape(score))
      prob = self._probability_fn(score) # [B, T]
    else: # use hard alignments for testing
      prob = tf.cast(tf.greater(score, 0.0), score.dtype)

    batch_size = tf.shape(prev_alignment)[0]
    pre_alignments_shifted = tf.concat([tf.zeros((batch_size, 1)), prev_alignment[:, :-1]], 1)
    prob_shifted = tf.concat([tf.zeros((batch_size, 1)), prob[:, :-1]], 1)

    alignments = (1.0 - prob) * prev_alignment + prob_shifted * pre_alignments_shifted

    # test


    # Reshape from [batch_size, memory_time] to [batch_size, 1, memory_time]
    alignments_reshape = array_ops.expand_dims(alignments, 1)

    # step3 calculate context, also called the glompse
    # alignment * encoder_output
    context = math_ops.matmul(alignments_reshape, self._attention_mechanism.values)

    # Reshape to [batch_size, memory_time]
    context_squeezed = array_ops.squeeze(context, [1])

    # calculate attention
    attention = self._attention_layer(
        array_ops.concat([cell_output, context_squeezed], 1))

    next_state = DynamicAttentionWrapperState(
        cell_state=next_cell_state,
        attention=attention)

    # concatenate pre net's output of previous frame output
    # and current context vector
    lstm_inputs = self._cell_input_fn(inputs, context_squeezed)

    new_state = []
    new_output = []

    new_state.append(next_state)
    new_output.append(cell_output)

    # update cell(LSTM) state
    for i, cell in enumerate(self._cells):
        with tf.variable_scope("lstm_{}".format(i)):
            lstm_output, next_lstm_state = cell(lstm_inputs, states[i + 1])
            lstm_inputs = lstm_output
            new_state.append(next_lstm_state)
            new_output.append(lstm_output)

    return new_output, new_state, context_squeezed, alignments