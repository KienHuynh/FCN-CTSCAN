from tfun.loss import loss
from tfun.trainer import trainer
from tfun.util import create_one_hot, create_var, create_conv_layer, create_linear_layer
import tfun.global_config as global_cfg

import tensorflow as tf
import gzip
import pdb

import six.moves.cPickle as pickle

import numpy as np

import time
from datetime import datetime
def loadMNIST(data_path): 
    with gzip.open(data_path, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)
    
    test_set_x, test_set_y = test_set
    valid_set_x, valid_set_y = valid_set
    train_set_x, train_set_y = train_set

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]

    return rval

class logger_hook(tf.train.SessionRunHook):
    def __init__(self, loss):
        super(logger_hook, self).__init__()
        self.loss = loss

    def begin(self):
        self.step = -1
        self.t0 = time.time()
        
    def before_run(self, run_context):
        self.step += 1
        return tf.train.SessionRunArgs(self.loss)

    def after_run(self, run_context, run_values):
        if self.step % global_cfg.log_freq == 0:
            t1 = time.time()
            duration = t1 - self.t0
            self.t0 = t1
            loss_value = run_values.results
            examples_per_sec = global_cfg.log_freq * global_cfg.batch_size / duration
            sec_per_batch = float(duration / global_cfg.log_freq)
            format_str = ('%s: step %d, loss = %.3f (%.1f examples/sec; %.3f sec/batch')
            print(format_str % (datetime.now(), self.step, loss_value, examples_per_sec, sec_per_batch))
            
if __name__ == "__main__":
    global_cfg.device = '/cpu:0'
    global_cfg.batch_size = 128

    data_path = '../data/mnist.pkl.gz'
    dataset = loadMNIST(data_path)
    train_X, train_Y = dataset[0]
    val_X, val_Y = dataset[1]
    test_X, test_Y = dataset[2]
    
    global_cfg.num_train = train_X.shape[0]
    global_cfg.num_val = val_X.shape[0]
    global_cfg.num_test = test_X.shape[0]

    train_X = np.reshape(train_X, (global_cfg.num_train, 28, 28, 1))
    train_Y = create_one_hot(train_Y, 10)

    val_X = np.reshape(val_X, (global_cfg.num_val, 28, 28, 1))
    val_Y = create_one_hot(val_Y, 10)

    test_X = np.reshape(test_X, (global_cfg.num_test, 28, 28, 1))
    test_Y = create_one_hot(test_Y, 10)

    # Create placeholder
    x = tf.placeholder(global_cfg.dtype, name='input_img')
    y = tf.placeholder(global_cfg.dtype, name='target')
    # Create the model 
    kernel_shape = (3,3,1,64) 
    with tf.variable_scope('conv1') as scope:
        x = create_conv_layer(x, kernel_shape, use_bias=True, activation=tf.nn.relu, name=scope.name)
    
    kernel_shape = (3,3,64,64)
    with tf.variable_scope('conv2') as scope:
        x = create_conv_layer(x, kernel_shape, use_bias=True, activation=tf.nn.relu, name=scope.name)
    
    x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1,2,2,1], padding = 'VALID', name = 'pool1')
    
    kernel_shape = (3, 3, 64, 128) 
    with tf.variable_scope('conv3') as scope:
        x = create_conv_layer(x, kernel_shape, use_bias=True, activation=tf.nn.relu, name=scope.name)

    kernel_shape = (3, 3, 128, 128) 
    with tf.variable_scope('conv4') as scope:
        x = create_conv_layer(x, kernel_shape, use_bias=True, activation=tf.nn.relu, name=scope.name)
     
    x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1,2,2,1], padding = 'VALID', name = 'pool2')

    kernel_shape = (3, 3, 128, 256)
    with tf.variable_scope('conv5') as scope:
        x = create_conv_layer(x, kernel_shape, use_bias=True, activation=tf.nn.relu, name=scope.name)
     
    kernel_shape = (3, 3, 256, 256) 
    with tf.variable_scope('conv6') as scope:
        x = create_conv_layer(x, kernel_shape, use_bias=True, activation=tf.nn.relu, name=scope.name)
     
    x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1,2,2,1], padding = 'VALID', name = 'pool3')

    with tf.variable_scope('fc') as scope:
        x = create_linear_layer(x, (3*3*256, 10), use_bias=True, name=scope.name)
 
    l = loss()
    l.softmax_log_loss(x, train_Y)
    l.l2_loss('weight', lm=0.0005)
    total_loss = l.total_loss()

    t = trainer(total_loss)
    t.create_adam_optimizer(0.001)
    t_ = t.get_trainer()
    with tf.train.MonitoredTrainingSession(
            checkpoint_dir=global_cfg.train_dir,
            hooks=[tf.train.StopAtStepHook(last_step=global_cfg.max_step),
                tf.train.NanTensorHook(total_loss),
                logger_hook(total_loss)]
            ) as sess:
        while not sess.should_stop():
            # create mini batches
            num_ite_per_epoch = int(
                    np.ceil(float(global_cfg.num_train)/float(global_cfg.batch_size))
                    )
            for i in range(num_ite_per_epoch):
                batch_range = range(i*global_cfg.batch_size, (i+1)*global_cfg.batch_size)
                if (batch_range[-1] > global_cfg.num_train):
                    batch_range = range(i*global_cfg.batch_size, global_cfg.num_train)
                batch_x = train_X[batch_range, :, :, :]
                batch_y = train_Y[batch_range, :, :, :]
                sess.run(t_, feed_dict={'input_img:0': batch_x, 'target:0': batch_y}) 
