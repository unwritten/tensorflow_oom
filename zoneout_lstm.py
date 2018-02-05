# implementation of paper Zoneout: Regularizing RNNs by Randomly Preserving Hidden Activations
# from https://github.com/teganmaharaj/zoneout
#
# I modified it to make it workable in current python and tensorflow env
#
# This is a minimal gist of what you'd have to
# add to TensorFlow code to implement zoneout.

# To see this in action, see zoneout_seq2seq.py

import tensorflow as tf

# as in tacotron2, LSTM layers are regularized using zoneout with probability 0.1.
z_prob_cells = 0.1
z_prob_states = 0.1

# Wrapper for the TF RNN cell
# For an LSTM, the 'cell' is a tuple containing state and cell
# We use TF's dropout to implement zoneout
class ZoneoutWrapper(tf.nn.rnn_cell.RNNCell):
  """Operator adding zoneout to all states (states+cells) of the given cell."""

  def __init__(self, cell, zoneout_prob=(z_prob_cells, z_prob_states), is_training=True, seed=None):
    if not isinstance(cell, tf.nn.rnn_cell.RNNCell):
      raise TypeError("The parameter cell is not an RNNCell.")
    if (isinstance(zoneout_prob, float) and
        not (zoneout_prob >= 0.0 and zoneout_prob <= 1.0)):
      raise ValueError("Parameter zoneout_prob must be between 0 and 1: %d"
                       % zoneout_prob)
    self._cell = cell
    self._zoneout_prob = zoneout_prob
    self._seed = seed
    self.is_training = is_training
  
  @property
  def state_size(self):
    return self._cell.state_size

  @property
  def output_size(self):
    return self._cell.output_size

  def __call__(self, inputs, state, scope=None):
    if isinstance(self.state_size, tuple) != isinstance(self._zoneout_prob, tuple):
      raise TypeError("Subdivided states need subdivided zoneouts.")
    if isinstance(self.state_size, tuple) and len(tuple(self.state_size)) != len(tuple(self._zoneout_prob)):
      raise ValueError("State and zoneout need equally many parts.")
    output, new_state = self._cell(inputs, state, scope)

    (c_prev, h_prev) = state
    (c_new, h_new) = new_state
    (z_prob_cell, z_prob_state) = self._zoneout_prob

    if isinstance(self.state_size, tuple):
      if self.is_training:
          c = (1 - z_prob_cell) * tf.nn.dropout(c_new - c_prev, (1 - z_prob_cell), seed=self._seed) + c_prev
          h = (1 - z_prob_state) * tf.nn.dropout(h_new - h_prev, (1 - z_prob_state), seed=self._seed) + h_prev
          new_state = tf.nn.rnn_cell.LSTMStateTuple(c, h)
      else:
          c = z_prob_cell * c_prev + (1 - z_prob_cell) * c_new
          h = z_prob_state * h_prev + (1 - z_prob_state) * h_new
          new_state = tf.nn.rnn_cell.LSTMStateTuple(c, h)

    else:
      if self.is_training:
          for new_state_part, state_part, state_part_zoneout_prob in zip(new_state, state, self._zoneout_prob):
              new_state = (1 - state_part_zoneout_prob) * tf.nn.dropout.dropout(
                        new_state_part - state_part, (1 - state_part_zoneout_prob), seed=self._seed) + state_part
      else:
          for new_state_part, state_part, state_part_zoneout_prob in zip(new_state, state, self._zoneout_prob):
              new_state = state_part_zoneout_prob * state_part + (1 - state_part_zoneout_prob) * new_state_part
    return output, new_state