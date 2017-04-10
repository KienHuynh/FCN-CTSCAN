import numpy as np
import tensorflow as tf
import pdb
import global_config as global_cfg

def create_one_hot(target_vector, num_class, dtype=np.float32):
    """
    Generate one-hot 4D tensor from a target vector of length N (num sample)
    The one-hot tensor will have the shape of (N x 1 x 1 x num_class)

    :param target_vector: Index vector, values are ranged from 0 to num_class-1

    :param num_class: number of classes/labels
    :return: target vector as a 4D tensor
    """
    one_hot = np.eye(num_class+1, num_class, dtype=dtype)
    one_hot = one_hot[target_vector]
    result = np.reshape(one_hot, (target_vector.shape[0], 1, 1, num_class))
    
    return result

def create_var(name, shape=None, initializer=None, trainable=True):
    """create_var
    Create a tensor variable
    If GPU should be used, specify it with global_cfg.device = '/gpu:0'
    :param name: name of the variable
    :param shape: the shape of the variable, tuple or list of int
    :param initializer: an tf initializer instance or a numpy array
    :param trainable: specify if the var should be trained in the main loop
    """ 
    with tf.device(global_cfg.device):
        dtype = global_cfg.dtype
        
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype, trainable=trainable)
        return var

def create_conv_layer(x, kernel_shape, use_bias, stride=[1,1,1,1], padding='SAME', activation=tf.nn.relu, wkey='weight', bkey='bias', name=None):

    kernel = create_var('weight', shape=kernel_shape, initializer=tf.contrib.layers.xavier_initializer())
    conv_result = tf.nn.conv2d(x, kernel, stride, padding=padding)
    if (use_bias):
        bias_shape = [1,1,1,kernel_shape[3]]
        bias = create_var('bias', shape=bias_shape, initializer=tf.constant_initializer(np.zeros(bias_shape)))
        conv_result = conv_result + bias
    if (activation != None):
        conv_result = activation(conv_result, name=name)
    
    return conv_result

