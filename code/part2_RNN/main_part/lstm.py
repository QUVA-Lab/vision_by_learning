################################################################################
# MIT License
#
# Copyright (c) 2018
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2018
# Date Created: 2018-09-04
################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


'''

Tom Runia (2018-09-04): This file contains the code from last year. The LSTM equations
are the same of course and implementing them is very similar to the code for
the Vanilla RNN. 

'''


class LSTM(object):

    def __init__(self, input_length, input_dim, num_hidden, num_classes, batch_size):

        self._input_length = input_length
        self._input_dim    = input_dim
        self._num_hidden   = num_hidden
        self._num_classes  = num_classes
        self._batch_size   = batch_size

        initializer_weights = tf.variance_scaling_initializer()
        initializer_biases  = tf.constant_initializer(0.0)

        # Placeholders serving as model input
        self._inputs  = tf.placeholder(tf.float32, [batch_size, input_length, input_dim], name="batch_inputs")
        self._targets = tf.placeholder(tf.int64, [batch_size], name="batch_targets")

        with tf.variable_scope("lstm_input_modulation_gate"):
            # Weights and biases for gate 'g'
            self._Wgx = tf.get_variable("U", shape=[self._input_dim, self._num_hidden], initializer=initializer_weights)
            self._Wgh = tf.get_variable("V", shape=[self._num_hidden, self._num_hidden], initializer=initializer_weights)
            self._bg = tf.get_variable("b", shape=[self._num_hidden], initializer=initializer_biases)

        with tf.variable_scope("lstm_input_gate"):
            # Weights and biases for gate 'i'
            self._Wix = tf.get_variable("U", shape=[self._input_dim, self._num_hidden], initializer=initializer_weights)
            self._Wih = tf.get_variable("V", shape=[self._num_hidden, self._num_hidden], initializer=initializer_weights)
            self._bi = tf.get_variable("b", shape=[self._num_hidden], initializer=initializer_biases)

        with tf.variable_scope("lstm_forget_gate"):
            # Weights and biases for gate 'f'
            self._Wfx = tf.get_variable("U", shape=[self._input_dim, self._num_hidden], initializer=initializer_weights)
            self._Wfh = tf.get_variable("V", shape=[self._num_hidden, self._num_hidden], initializer=initializer_weights)
            self._bf = tf.get_variable("b", shape=[self._num_hidden], initializer=initializer_biases)

        with tf.variable_scope("lstm_output_gate"):
            # Weights and biases for gate 'f'
            self._Wox = tf.get_variable("U", shape=[self._input_dim, self._num_hidden], initializer=initializer_weights)
            self._Woh = tf.get_variable("V", shape=[self._num_hidden, self._num_hidden], initializer=initializer_weights)
            self._bo = tf.get_variable("b", shape=[self._num_hidden], initializer=initializer_biases)

        with tf.variable_scope("output_layer"):
            # Final linear output layer
            self._Wout = tf.get_variable("W", shape=[self._num_hidden, self._num_classes], initializer=initializer_weights)
            self._bout = tf.get_variable("b", shape=[self._num_classes], initializer=initializer_biases)

        self._logits = self._compute_logits()
        self._loss = self._compute_loss()
        self._accuracy = tf.contrib.metrics.accuracy(self.targets, self.predictions)

    def _rnn_step(self, lstm_state_tuple, x):

        # Previous hidden state h_{t-1} and memory state c_{t-1}
        h_prev, c_prev = tf.unstack(lstm_state_tuple)

        # Apply the gates
        g = tf.tanh(tf.matmul(x, self._Wgx) + tf.matmul(h_prev, self._Wgh) + self._bg, name="input_modulation")
        i = tf.sigmoid(tf.matmul(x, self._Wix) + tf.matmul(h_prev, self._Wih) + self._bi, name="input_gate")
        f = tf.sigmoid(tf.matmul(x, self._Wfx) + tf.matmul(h_prev, self._Wfh) + self._bf, name="forget_gate")
        o = tf.sigmoid(tf.matmul(x, self._Wox) + tf.matmul(h_prev, self._Woh) + self._bo, name="output_gate")

        # Update memory cell
        c = tf.multiply(g, i) + tf.multiply(c_prev, f)

        # Update hidden state
        h = tf.multiply(tf.tanh(c), o)

        # Return the new memory state h_t and memory state c_t
        return tf.stack([h, c])

    def _compute_logits(self):
        '''
        Computes the logits of a vanilla RNN cell using tf.scan operation.
        The output only contains the logits and state at the final step.

        :return: logits, tf.Tensor of size [batch_size, num_classes]
        :return: state,  tf.Tensor of size [batch_size, num_hidden]
        '''

        # Time needs to be the first dimension for tf.scan, hence we
        # transpose the input tensor to dimension [T,N,D].
        input_steps = tf.transpose(self.inputs, perm=(1,0,2))

        # Forward-pass through vanilla RNN cell
        with tf.variable_scope("rnn"):
            # INSTRUCTORS NOTE: setting the initial state is important
            lstm_state_tuple = tf.zeros([2, self._batch_size, self._num_hidden], name="lstm_state_tuple")
            states = tf.scan(self._rnn_step, input_steps, initializer=lstm_state_tuple, name='states')

        # Compute the logits at the final timestep
        with tf.variable_scope("output"):
            h_final, _ = tf.unstack(states[-1])
            logits = tf.matmul(h_final, self._Wout)
            logits = tf.nn.bias_add(logits, self._bout)

        return logits

    def _compute_loss(self):
        '''
        Computes the cross-entropy loss with respect to the targets.
        We compute the loss over the final step, not over all steps.

        :return: loss, tf.Tensor of scalar size
        '''
        targets_one_hot = tf.one_hot(self.targets, depth=self._num_classes)
        loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=targets_one_hot, logits=self.logits)
        loss = tf.reduce_mean(loss)
        return loss

    @property
    def inputs(self):
        return self._inputs

    @property
    def targets(self):
        return self._targets

    @property
    def logits(self):
        return self._logits

    @property
    def loss(self):
        return self._loss

    @property
    def probabilities(self):
        return tf.nn.softmax(self.logits)

    @property
    def predictions(self):
        return tf.argmax(self.probabilities, axis=1)

    @property
    def accuracy(self):
        return self._accuracy
