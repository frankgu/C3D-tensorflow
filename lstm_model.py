"""Builds the LSTM network.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

# LSTM hidden unit size
HIDDEN_SIZE = 200
# The probability to keep the input
KEEP_PROB = 0.9

LSTM_NUM_STEPS = 2
NUM_LAYERS = 2
BATCH_SIZE = 10
NUM_CLASSES = 24


def data_type():
  return tf.float16 if FLAGS.use_fp16 else tf.float32


class LSTM(object):
  """Bidirectional LSTM neural network.

  Use this function to create the LSTM nerual network model

  Args:
    is_training: Boolean, whether to apply a dropout layer or not
    inputs: video_input class instance 
      input_data: List. lstm_num_steps * [batch_size, input_num(features)]
      targets: Tensor, corresponding one hot vector groudtrue for 
        input, Size:[batch_size, num_classes]
    is_video: Boolean, whether the input is a video or not, default is
      false
  """
  def __init__(self, is_training, inputs_):
    self._input = inputs_
    # Define lstm cells
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(
      HIDDEN_SIZE, forget_bias=0.0, state_is_tuple=True)
    if is_training:
      lstm_cell = tf.contrib.rnn.DropoutWrapper(
        lstm_cell, output_keep_prob=KEEP_PROB)
    cell = tf.contrib.rnn.MultiRNNCell(
      [lstm_cell] * NUM_LAYERS, state_is_tuple=True)

    self._initial_state = cell.zero_state(BATCH_SIZE, data_type())

    if is_training:
      inputs = [tf.nn.dropout(single_input, KEEP_PROB) 
                    for single_input in self._input.bilstm_inputs]

    self._outputs, state = tf.nn.rnn(lstm_cell, inputs, self._initial_state)

    with tf.variable_scope('lstm_var'):
      softmax_w = tf.get_variable('softmax_w', [HIDDEN_SIZE, NUM_CLASSES], 
                                  data_type())
      softmax_b = tf.get_variable('softmax_b', [NUM_CLASSES], data_type())

    # Linear activation, use the last element of the rnn outputs
    self._logits = tf.matmul(self._outputs[-1], softmax_w) + softmax_b
    