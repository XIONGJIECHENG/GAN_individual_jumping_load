# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 21:11:15 2018

@author: DELL

利用RNN和GAN对信号进行生成,此文件是主文件是训练的主文件
"""
import tensorflow as tf
import config as cf
from singlelength_run import run

stages = range(cf.START_SEQ, cf.END_SEQ+1)

for i in range(len(stages)):
    prev_seq_length = stages[i-1] if i>0 else 0
    seq_length = stages[i]
    tf.reset_default_graph()
    iterations = cf.ITERATIONS_PER_SEQ_LENGTH
    run(iterations, seq_length, seq_length == stages[0], prev_seq_length)

    
    