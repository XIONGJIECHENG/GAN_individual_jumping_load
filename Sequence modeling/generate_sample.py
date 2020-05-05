# -*- coding: utf-8 -*-
"""
Created on Sun Jul 15 17:23:32 2018

@author: DELL
此文件是根据训练好的权重生成样本
"""
import tensorflow as tf
import config as cf
import os
import numpy as np

from model import generator

import pro_matdata

cf.BATCH_SIZE = 42000

SEQ_LEN = 100
y = tf.placeholder(tf.float32, shape=[None, cf.Y_SIZE])
y_sample = np.zeros(shape=[cf.BATCH_SIZE , cf.Y_SIZE])
for k in range(cf.Y_SIZE):
    y_sample[k*3000:(k+1)*3000,k]=1


inference_op = generator(cf.BATCH_SIZE, y, seq_len=SEQ_LEN)
#inference_op = tf.reshape(inference_op, [cf.BATCH_SIZE, SEQ_LEN])




saver = tf.train.Saver()

with tf.Session() as session:
    internal_checkpoint_dir = os.path.join((cf.SAVE_CHECKPOINTSLOG), str(32))
    saver.restore(session, tf.train.latest_checkpoint(internal_checkpoint_dir, "checkpoint"))
    sequential_output = session.run(inference_op, feed_dict={y:y_sample})

pro_matdata.save_sample(sequential_output)






