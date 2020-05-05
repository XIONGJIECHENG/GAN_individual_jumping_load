# -*- coding: utf-8 -*-
"""
Created on Sun Jul 15 13:52:34 2018

@author: DELL
此文件主要是为了定义目标函数和目标优化操作
"""
import tensorflow as tf
import config as cf
from model import generator, discriminator, params_with_name


def get_optimization_ops(disc_cost, gen_cost):
    gen_params = params_with_name('Generator')
    disc_params = params_with_name('Discriminator')
    print("Generator Params: %s" % gen_params)
    print("Disc Params: %s" % disc_params)
    gen_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(gen_cost,
                                                                                             var_list=gen_params)
    disc_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(disc_cost,
                                                                                              var_list=disc_params)
    return disc_train_op, gen_train_op


def define_objective(real_inputs_discrete, y, seq_length):
    real_inputs = real_inputs_discrete
    train_pred = generator(cf.BATCH_SIZE, y, seq_len=seq_length)
    disc_real = discriminator(real_inputs, y, seq_length, reuse=False)
    disc_fake = discriminator(train_pred, y, seq_length, reuse=True)
    disc_cost, gen_cost = loss_d_g(disc_fake, disc_real, train_pred, real_inputs, y, seq_length, discriminator)
    return disc_cost, gen_cost

def loss_d_g(disc_fake, disc_real, fake_inputs, real_inputs, y, seq_length, discriminator):
    disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
    gen_cost = -tf.reduce_mean(disc_fake)

    # WGAN lipschitz-penalty
    alpha = tf.random_uniform(
        shape=[tf.shape(real_inputs)[0], 1, 1],
        minval=0.,
        maxval=1.
    )
    differences = fake_inputs - real_inputs
    interpolates = real_inputs + (alpha * differences)
    gradients = tf.gradients(discriminator(interpolates, y, seq_length, reuse=True), [interpolates,y ])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2]))
    gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
    disc_cost += cf.LAMBDA * gradient_penalty

    return disc_cost, gen_cost
