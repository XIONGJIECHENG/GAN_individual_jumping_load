# -*- coding: utf-8 -*-
"""
Created on Sun Jul 15 14:34:21 2018

@author: DELL
此文件是针对具体循环网络进行建模
"""
import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell

import config as cf


def discriminator(Inputs, y, seq_len, reuse=False):
    with tf.variable_scope("Discriminator", reuse=reuse):
        num_neurons = cf.DISC_STATE_SIZE
        # backwards compatability
        cell = GRUCell(num_neurons)
        y_un = tf.reshape(y,[cf.BATCH_SIZE, cf.Y_SIZE, 1])
        y_un1 = tf.concat(axis=2, values=[y_un, y_un])
        inputs = tf.concat(axis=1, values=[Inputs, y_un1])     
        inputs = tf.unstack(tf.transpose(inputs, [1,0,2]))
        output, state = tf.contrib.rnn.static_rnn(
            cell,
            inputs,
            dtype=tf.float32)

        last = output[-1]

        weight = tf.get_variable("W", shape=[num_neurons, 1],
                                 initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
        bias = tf.get_variable("b", shape=[1], initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))

        prediction = tf.matmul(last, weight) + bias

        return prediction


def generator(n_samples, y, seq_len=None):
    with tf.variable_scope("Generator"):
        num_neurons = cf.GEN_STATE_SIZE
        cell = GRUCell(num_neurons+cf.Y_SIZE)

        # this is separate to decouple train and test
        z, _ = get_noise()
        inference_initial_states = tf.concat(axis=1, values=[z, y])
        
        sm_weight = tf.Variable(tf.random_uniform([num_neurons+cf.Y_SIZE, 2], minval=-0.1, maxval=0.1))
        sm_bias = tf.Variable(tf.random_uniform([2], minval=-0.1, maxval=0.1))

        char_input = make_noise([cf.BATCH_SIZE, 2], mean=0.0, stddev=1.0)
        #char_input = tf.Variable(tf.random_uniform([1], minval=-0.1, maxval=0.1))
        #char_input = tf.reshape(tf.tile(char_input, [n_samples]), [n_samples, 1])

        inference_op = get_inference_op(cell, char_input, seq_len, sm_bias, sm_weight, inference_initial_states,
                                            num_neurons, reuse=False)
        return inference_op

def get_inference_op(cell, char_input, seq_len, sm_bias, sm_weight, states, num_neurons, reuse=False):
    inference_pred = []
    for i in range(seq_len):
        with tf.variable_scope("rnn", reuse=reuse):
            output, state = tf.contrib.rnn.static_rnn(
            cell,
            [char_input],
            initial_state = states,
            dtype=tf.float32)
            GRU_output = tf.matmul(state, sm_weight) + sm_bias
            inference_pred.append(GRU_output)
            char_input = GRU_output
            states = state
            reuse = True
    return tf.reshape(tf.concat(inference_pred, axis=1),[cf.BATCH_SIZE, seq_len, 2])


def get_noise():
    noise_shape = [cf.BATCH_SIZE, cf.GEN_STATE_SIZE]
    return make_noise(shape=noise_shape, stddev=cf.NOISE_STDEV), noise_shape

def make_noise(shape, mean=0.0, stddev=10.0):
    return tf.random_normal(shape, mean, stddev)

def params_with_name(name):
    return [p for p in tf.trainable_variables() if name in p.name]






