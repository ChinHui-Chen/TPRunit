"Define a PTR cell."
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf

class PTRCell(tf.contrib.rnn.RNNCell):

    def __init__(self, nSymbols = 100, nRoles = 10, dSymbols = 10, dRoles = 10, dEmb = 100, initializer=None, recurrent_initializer=None, batch_size = 32):
        self._nSymbols = nSymbols
        self._nRoles = nRoles
        self._dSymbols = dSymbols
        self._dRoles = dRoles
        self._dEmb = dEmb
        self._initializer = initializer
        self._recurrent_initializer = recurrent_initializer
        self._batch_size = batch_size

    @property
    def state_size(self):
        "Return the total state size of the cell"
        return self._dSymbols * self._dRoles

    @property
    def output_size(self):
        "Return the total output size of the cell"
        return self._dSymbols * self._dRoles

    def zero_state(self, batch_size, dtype):
        return tf.zeros([batch_size, self._dSymbols * self._dRoles], dtype=dtype)

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__, initializer=self._initializer):
           
            W_s_in = tf.get_variable('W_s_in', [self._dEmb, self._nSymbols])
            W_s_rec = tf.get_variable('W_s_rec', [self._dSymbols*self._dRoles, self._nSymbols])
            b_s = tf.get_variable('b_s', [self._nSymbols])

            W_r_in = tf.get_variable('W_r_in', [self._dEmb, self._nRoles])
            W_r_rec = tf.get_variable('W_r_rec', [self._dSymbols*self._dRoles, self._nRoles])
            b_r = tf.get_variable('b_r', [self._nRoles])

            a_s_t = tf.sigmoid( tf.matmul(inputs, W_s_in) + tf.matmul(state, W_s_rec) + b_s )
            a_r_t = tf.sigmoid( tf.matmul(inputs, W_r_in) + tf.matmul(state, W_r_rec) + b_r )

            S = tf.get_variable('S', [self._nSymbols, self._dSymbols], initializer=self._recurrent_initializer)
            R = tf.get_variable('R', [self._nRoles, self._dRoles], initializer=self._recurrent_initializer)

            S_matmul = tf.matmul(a_s_t, S)
            S_matmul = tf.reshape( S_matmul, [-1, self._dSymbols, 1] )
            R_matmul = tf.matmul(a_r_t, R)
            R_matmul = tf.reshape( R_matmul, [-1, 1, self._dRoles] )
            output = tf.matmul( S_matmul, R_matmul )
            output = tf.reshape(output, [-1, self._dSymbols*self._dRoles])

            return output, output

