from tfun.loss import loss
from tfun.trainer import trainer
from tfun.util import create_one_hot, create_conv_layer, create_projection_layer, LoadH5, save_hook
import tfun.global_config as global_cfg

import tensorflow as tf
import numpy as np

import re
import glob
import os.path

import time
import pdb

def create_fcn():
    # The net structure is the same with fcn_pretrain with the last layer removed
    x = tf.placeholder(global_cfg.dtype, name='input_img') 
    is_train = tf.placeholder(np.bool, name='is_train') # For dropout train/test

    keep_rate = 0.9
    x = tf.cond(is_train,
            lambda: tf.nn.dropout(x, keep_prob = keep_rate, seed=global_cfg.rng_seed),
            lambda: x,
            name='dropout0')

    # CONV STACK 1
    kernel_shape = (3, 3, 3, 32)
    with tf.variable_scope('conv1_1') as scope:
        x = create_conv_layer(x, kernel_shape, use_bias=True, activation=tf.nn.relu, name=scope.name)
    
    kernel_shape = (3, 3, 32, 32)
    with tf.variable_scope('conv1_2') as scope:
        x = create_conv_layer(x, kernel_shape, use_bias=True, activation=tf.nn.relu, name=scope.name)
   
    x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1,2,2,1], padding = 'VALID', name = 'pool1')

    keep_rate = 0.5
    x = tf.cond(is_train,
            lambda: tf.nn.dropout(x, keep_prob = keep_rate, seed=global_cfg.rng_seed),
            lambda: x,
            name='dropout1')

    # CONV STACK 2
    kernel_shape = (3, 3, 32, 64)
    with tf.variable_scope('conv2_1') as scope:
        x = create_conv_layer(x, kernel_shape, use_bias=True, activation=tf.nn.relu, name=scope.name)
    
    kernel_shape = (3, 3, 64, 64)
    with tf.variable_scope('conv2_2') as scope:
        x = create_conv_layer(x, kernel_shape, use_bias=True, activation=tf.nn.relu, name=scope.name)

    x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1,2,2,1], padding = 'VALID', name = 'pool2')

    keep_rate = 0.5
    x = tf.cond(is_train,
            lambda: tf.nn.dropout(x, keep_prob = keep_rate, seed=global_cfg.rng_seed),
            lambda: x,
            name='dropout2')

    # CONV STACK 3
    kernel_shape = (3, 3, 64, 128)
    with tf.variable_scope('conv3_1') as scope:
        x = create_conv_layer(x, kernel_shape, use_bias=True, activation=tf.nn.relu, name=scope.name) 
    
    kernel_shape = (3, 3, 128, 128)
    with tf.variable_scope('conv3_2') as scope:
        x = create_conv_layer(x, kernel_shape, use_bias=True, activation=tf.nn.relu, name=scope.name)

    x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1,2,2,1], padding = 'VALID', name = 'pool3')

    keep_rate = 0.6
    x = tf.cond(is_train,
            lambda: tf.nn.dropout(x, keep_prob = keep_rate, seed=global_cfg.rng_seed),
            lambda: x,
            name='dropout3')


    # CONV STACK 4
    kernel_shape = (3, 3, 128, 256)
    with tf.variable_scope('conv4_1') as scope:
        x = create_conv_layer(x, kernel_shape, use_bias=True, activation=tf.nn.relu, name=scope.name)
    
    kernel_shape = (3, 3, 256, 256)
    with tf.variable_scope('conv4_2') as scope:
        x = create_conv_layer(x, kernel_shape, use_bias=True, activation=tf.nn.relu, name=scope.name)

    x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1,2,2,1], padding = 'VALID', name = 'pool4')

    keep_rate = 0.5
    x = tf.cond(is_train,
            lambda: tf.nn.dropout(x, keep_prob = keep_rate, seed=global_cfg.rng_seed),
            lambda: x,
            name='dropout4')

    # CONV STACK 5
    kernel_shape = (3, 3, 256, 512)
    with tf.variable_scope('conv5_1') as scope:
        x = create_conv_layer(x, kernel_shape, use_bias=True, activation=tf.nn.relu, name=scope.name)

    x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1,2,2,1], padding = 'VALID', name = 'pool5')

    x = tf.cond(is_train,
            lambda: tf.nn.dropout(x, keep_prob = keep_rate, seed=global_cfg.rng_seed),
            lambda: x,
            name='dropout5')
    
    # CONV STACK 6
    kernel_shape = (3, 3, 512, 512)
    with tf.variable_scope('conv6_1') as scope:
        x = create_conv_layer(x, kernel_shape, use_bias=True, activation=tf.nn.relu, name=scope.name)

    kernel_shape = (3, 3, 512, 512)
    with tf.variable_scope('conv6_2') as scope:
        x = create_conv_layer(x, kernel_shape, use_bias=True, activation=tf.nn.relu, name=scope.name)
 
    # Added liner projection layerr after conv4, conv5 and conv6
    x4 = create_projection_layer('conv4_2:0', (256, 4), name='projection_4')
    x5 = create_projection_layer('conv5_1:0', (512, 4), name='projection_5')
    x6 = create_projection_layer('conv6_2:0', (512, 4), name='projection_6')

    # Creating transposed conv layers
    kernel_shape = (3, 3, 4, 4)

    # Do a succession of 5 transposed conv
    for i in range(1, 6):
        with tf.variable_scope('t_conv6_%d' % i) as scope:
            kernel = create_var('weight', shape=kernel_shape, initializer=tf.contrib.layers.xavier_initializer())
            
            x6_shape = tf.shape(x6)
            x6_shape[1] = x6_shape[1]*(2**5)
            x6_shape[2] = x6_shape[2]*(2**5)
            x6 = tf.nn.conv2d_transpose(x6, kernel, x6_shape, strides=[1, 2, 2, 1], name=scope.name)
    
    # Do a succession of 4 transposed conv
    for i in range(1, 5):
        with tf.variable_scope('t_conv5_%d' % i) as scope:
            kernel = create_var('weight', shape=kernel_shape, initializer=tf.contrib.layers.xavier_initializer())
            
            x5_shape = tf.shape(x5)
            x5_shape[1] = x5_shape[1]*(2**4)
            x5_shape[2] = x5_shape[2]*(2**4)
            x5 = tf.nn.conv2d_transpose(x5, kernel, x5_shape, strides=[1, 1, 1, 1], name=scope.name)
    
    # Do a succession of 3 transpoed conv
    for i in range(1, 4):
        with tf.variable_scope('t_conv4') as scope:
            kernel = create_var('weight', shape=kernel_shape, initializer=tf.contrib.layers.xavier_initializer())
            
            x4_shape = tf.shape(x4)
            x4_shape[1] = x4_shape[1]*(2**4)
            x4_shape[2] = x4_shape[2]*(2**4)
            x4 = tf.nn.conv2d_transpose(x4, kernel, x4_shape, strides=[1, 1, 1, 1], name=scope.name)
    
    # x4 x5 and x6 should have the same shape now
    x = x4+x5+x6
    return x

if __name__ == "__main__":
    # Load data
    x = 1
