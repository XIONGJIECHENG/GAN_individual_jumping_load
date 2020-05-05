# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 11:17:57 2018

@author: DELL
"""

import tensorflow as tf
import numpy as np
import os
import scipy.io as sio  
import time

lam = 10
mb_size = 128
X_dim = 100
z_dim = 100
y_dim = 14
h_dim = 100
n_disc = 5
lr = 1e-4

data=sio.loadmat('E:\\jump_rnn-gan\\pulse_try\\IMPULSE.mat')

def convert_to_one_hot(y, C):
    return np.eye(C)[y.reshape(-1)]

def get_data(batch_size):
    Load = data['Impulse']
    dataxy = Load[np.random.randint(Load.shape[0], size=batch_size),:]
    x = dataxy[:,:-1]
    y = dataxy[:,-1]-1
    y_int = y.astype(int) 
    y_hot = convert_to_one_hot(y_int, y_dim)
    return x, y_hot

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

X = tf.placeholder(tf.float32, shape=[None, X_dim])
y = tf.placeholder(tf.float32, shape=[None, y_dim])

D_W1 = tf.Variable(xavier_init([X_dim+y_dim, h_dim]))
D_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

D_W2 = tf.Variable(xavier_init([h_dim, 1]))
D_b2 = tf.Variable(tf.zeros(shape=[1]))

theta_D = [D_W1, D_W2, D_b1, D_b2]


z = tf.placeholder(tf.float32, shape=[None, z_dim])

G_W1 = tf.Variable(xavier_init([z_dim+y_dim, h_dim]))
G_b1 = tf.Variable(tf.zeros(shape=[h_dim]))



G_W2 = tf.Variable(xavier_init([h_dim, X_dim]))
G_b2 = tf.Variable(tf.zeros(shape=[X_dim]))

G_W3 = tf.Variable(tf.truncated_normal([10, 1, 1]))

theta_G = [G_W1, G_W2, G_b1, G_b2, G_W3]



def sample_z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


def G(z, y):
    inputs = tf.concat(axis=1, values=[z, y])
    G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)
    G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)

    G_h3 = tf.reshape(G_h2, [-1, 1, 100])
    G_h4 = tf.nn.conv1d(G_h3, G_W3, stride=1, padding='SAME', data_format="NCW")
    G_h5 = tf.nn.relu(tf.reshape(G_h4, [-1, 100]))
    return G_h5


def D(X, y):
    inputs = tf.concat(axis=1, values=[X, y])
    D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)
    out = tf.matmul(D_h1, D_W2) + D_b2
    return out


G_sample = G(z, y)
D_real = D(X, y)
D_fake = D(G_sample, y)

eps = tf.random_uniform([mb_size, 1], minval=0., maxval=1.)
X_inter = eps*X + (1. - eps)*G_sample
grad = tf.gradients(D(X_inter, y), [X_inter, y])[0]
grad_norm = tf.sqrt(tf.reduce_sum((grad)**2, axis=1))
grad_pen = lam * tf.reduce_mean((grad_norm - 1)**2)

D_loss = tf.reduce_mean(D_fake) - tf.reduce_mean(D_real) + grad_pen
G_loss = -tf.reduce_mean(D_fake)

D_solver = (tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5)
            .minimize(D_loss, var_list=theta_D))
G_solver = (tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5)
            .minimize(G_loss, var_list=theta_G))

start_time = time.time()
saver = tf.train.Saver()

sess = tf.Session()
sess.run(tf.global_variables_initializer())
#saver.restore(sess, 'E:\\jump_rnn-gan\\pulse_try\\c6\\my_model-1000000')

if not os.path.exists('E:\\jump_rnn-gan\\pulse_try\\Checkpoint_0430\\'):
    os.makedirs('E:\\jump_rnn-gan\\pulse_try\\Checkpoint_0430\\')

if not os.path.exists('E:\\jump_rnn-gan\\pulse_try\\smi_result\\'):
    os.makedirs('E:\\jump_rnn-gan\\pulse_try\\smi_result\\')
    
i = 0
for it in range(1000001):
    for _ in range(n_disc):
        X_mb, y_mb = get_data(mb_size)

        _, D_loss_curr = sess.run(
            [D_solver, D_loss],
            feed_dict={X: X_mb, z: sample_z(mb_size, z_dim), y:y_mb})

    _, G_loss_curr = sess.run(
        [G_solver, G_loss],
        feed_dict={z: sample_z(mb_size, z_dim), y:y_mb})

    if it % 1000 == 0:
        t2 = time.time()-start_time
        print('Iter: {}; time: {:.4} D loss: {:.4}; G_loss: {:.4}'
              .format(it, t2, D_loss_curr, G_loss_curr))

        if it % 10000 == 0:
            y_sample = np.zeros(shape=[140, y_dim])
            y_sample[0:10, 0] = 1
            y_sample[10:20, 1] = 1
            y_sample[20:30, 2] = 1
            y_sample[30:40, 3] = 1
            y_sample[40:50, 4] = 1
            y_sample[50:60, 5] = 1
            y_sample[60:70, 6] = 1
            y_sample[70:80, 7] = 1
            y_sample[80:90, 8] = 1
            y_sample[90:100, 9] = 1
            y_sample[100:110, 10] = 1
            y_sample[110:120, 11] = 1
            y_sample[120:1300, 12] = 1
            y_sample[130:140, 13] = 1
            samples = sess.run(G_sample, feed_dict={z: sample_z(140, z_dim), y:y_sample})
            sio.savemat('E:\\jump_rnn-gan\\pulse_try\\smi_result\\Result{}.mat'.format(str(i)), {'Gene_load': samples}) 
            saver.save(sess, 'E:\\jump_rnn-gan\\pulse_try\\Checkpoint_0430\\'+'my_model', global_step=it)
            i += 1