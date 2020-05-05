# -*- coding: utf-8 -*-
"""
Created on Sun Jul 15 16:51:18 2018

@author: DELL
此文件是为了随机生成训练样本，以及保存训练样本
"""
import scipy.io as sio
import numpy as np
import time
import config as cf

def convert_to_one_hot(y, C):
    return np.eye(C)[y.reshape(-1)]


data=sio.loadmat('E:\\jump_rnn-gan\\python_con_amp\\T_DATA_AMP.mat')


def get_data(batch_size, seq_len):
    Data_AMP = data['Data_AMP']
    Data_T = data['Data_T']
    Data_AMP_T = np.array([Data_AMP,Data_T])
    Data_AMP_T = Data_AMP_T.swapaxes(0,1)
    Data_AMP_T = Data_AMP_T.swapaxes(2,1)
    
    index = np.random.randint(Data_AMP_T.shape[0], size=batch_size)
    
    xy = Data_AMP_T[index]
    x = xy[:,:seq_len,:]
    y = xy[:,-1,0]-1
    y_int = y.astype(int) 
    y_hot = convert_to_one_hot(y_int, cf.Y_SIZE)
    return x, y_hot

def save_sample(samples):
    sio.savemat('E:\\jump_rnn-gan\\python_con_amp\\smi_result\\Result{}.mat'.format(str(time.time())), {'smi_data': samples})  


