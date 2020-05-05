# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 21:57:41 2018

@author: DELL
此文件是针对一个固定长度样本的训练
"""


import config as cf
from objective import get_optimization_ops, define_objective
import tensorflow as tf
import os 
from pro_matdata import get_data


def run(iterations, seq_length, is_first,  prev_seq_length):
    restore_path = os.path.join((cf.SAVE_CHECKPOINTSLOG), str(prev_seq_length))
    save_path = os.path.join((cf.SAVE_CHECKPOINTSLOG), str(seq_length))
    real_inputs_discrete = tf.placeholder(tf.float32, shape=[cf.BATCH_SIZE, seq_length,2])
    y = tf.placeholder(tf.float32, shape=[None, cf.Y_SIZE])
    disc_cost, gen_cost = define_objective(real_inputs_discrete, y, seq_length)
    disc_train_op, gen_train_op = get_optimization_ops(disc_cost, gen_cost)
    saver = tf.train.Saver(tf.trainable_variables())
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        if not is_first:
            print("Loading previous checkpoint...")
            internal_checkpoint_dir = restore_path
            saver.restore(session, tf.train.latest_checkpoint(internal_checkpoint_dir, "checkpoint"))
            
            
        for iteration in range(iterations):
          

            # Train critic
            for i in range(cf.CRITIC_ITERS):
                _data, _y = get_data(cf.BATCH_SIZE, seq_length)   ###get_data需要编写
                _disc_cost, _ = session.run(
                    [disc_cost, disc_train_op],
                    feed_dict={real_inputs_discrete: _data, y:_y}
                )
            # Train G
            for i in range(cf.GEN_ITERS):
                _gen_cost, _ = session.run(
                    [gen_cost, gen_train_op],
                    feed_dict={y:_y})
                

            if  iteration%100 ==0:
                print("iteration %s/%s"%(iteration, iterations))
                print("disc cost %f"%_disc_cost)
                print("gen cost %f"%_gen_cost)
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            if iteration % cf.SAVE_CHECKPOINTS_EVERY == cf.SAVE_CHECKPOINTS_EVERY-1:
                saver.save(session, save_path+'\\my_model.ckpt', global_step=iteration)
        session.close()
    
                                                                                                          
