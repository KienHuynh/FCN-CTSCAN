from tfun.loss import loss
from tfun.trainer import trainer
from tfun.util import create_one_hot, create_conv_layer, create_projection_layer, LoadH5, save_hook, create_var
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
   
    drop4 = tf.cond(is_train,
            lambda: tf.nn.dropout(x, keep_prob = keep_rate, seed=global_cfg.rng_seed),
            lambda: x,
            name='dropout4')

    x = tf.nn.max_pool(drop4, ksize=[1, 2, 2, 1], strides=[1,2,2,1], padding = 'VALID', name = 'pool4')

    # CONV STACK 5
    kernel_shape = (3, 3, 256, 512)
    with tf.variable_scope('conv5_1') as scope:
        x = create_conv_layer(x, kernel_shape, use_bias=True, activation=tf.nn.relu, name=scope.name) 
    
    drop5 = tf.cond(is_train,
            lambda: tf.nn.dropout(x, keep_prob = keep_rate, seed=global_cfg.rng_seed),
            lambda: x,
            name='dropout5')

    x = tf.nn.max_pool(drop5, ksize=[1, 2, 2, 1], strides=[1,2,2,1], padding = 'VALID', name = 'pool5')

    # CONV STACK 6
    kernel_shape = (3, 3, 512, 512)
    with tf.variable_scope('conv6_1') as scope:
        x = create_conv_layer(x, kernel_shape, use_bias=True, activation=tf.nn.relu, name=scope.name)

    kernel_shape = (3, 3, 512, 512)
    with tf.variable_scope('conv6_2') as scope:
        x = create_conv_layer(x, kernel_shape, use_bias=True, activation=tf.nn.relu, name=scope.name)
    
    drop6 = tf.cond(is_train,
            lambda: tf.nn.dropout(x, keep_prob = keep_rate, seed=global_cfg.rng_seed),
            lambda: x,
            name='dropout6')


    # Added liner projection layerr after conv4, conv5 and conv6

    with tf.variable_scope('project4'):
        x4 = create_projection_layer(drop4, (256, 4), name=scope.name)
    with tf.variable_scope('project5'):
        x5 = create_projection_layer(drop5, (512, 4), name=scope.name)
    with tf.variable_scope('project6'):
        x6 = create_projection_layer(drop6, (512, 4), name=scope.name)

    # Creating transposed conv layers
    kernel_shape = (3, 3, 4, 4)
    
    keep_rate = [0.6, 0.7, 0.8, 0.9, 1]
    # Do a succession of 5 transposed conv
    for i in range(1, 6):
        with tf.variable_scope('t_conv6_%d' % i) as scope:
            kernel = create_var('weight%d' % i, shape=kernel_shape, initializer=tf.contrib.layers.xavier_initializer())
            x6 = tf.cond(is_train,
                lambda: tf.nn.dropout(x6, keep_prob = keep_rate[i-1], seed=global_cfg.rng_seed),
                lambda: x6,
                name='dropout%d' % i)
            x6_shape = tf.shape(x6)
            new_shape = (x6_shape[0], x6_shape[1]*2, x6_shape[1]*2, x6_shape[3])
            x6 = tf.nn.conv2d_transpose(x6, kernel, new_shape, strides=[1, 2, 2, 1], name=scope.name)
   
    keep_rate = [0.6, 0.75, 0.85, 1]
    # Do a succession of 4 transposed conv
    for i in range(1, 5):
        with tf.variable_scope('t_conv5_%d' % i) as scope:
            kernel = create_var('weight%d' % i, shape=kernel_shape, initializer=tf.contrib.layers.xavier_initializer())
            x5 = tf.cond(is_train,
                lambda: tf.nn.dropout(x5, keep_prob = keep_rate[i-1], seed=global_cfg.rng_seed),
                lambda: x5,
                name='dropout%d' % i)
            x5_shape = tf.shape(x5)
            new_shape = (x5_shape[0], x5_shape[1]*2, x5_shape[1]*2, x5_shape[3])
            x5 = tf.nn.conv2d_transpose(x5, kernel, new_shape, strides=[1, 2, 2, 1], name=scope.name)
    
    keep_rate = [0.6, 0.8, 1]
    # Do a succession of 3 transpoed conv
    for i in range(1, 4):
        with tf.variable_scope('t_conv4') as scope:
            kernel = create_var('weight%d' % i, shape=kernel_shape, initializer=tf.contrib.layers.xavier_initializer())
            x4= tf.cond(is_train,
                lambda: tf.nn.dropout(x4, keep_prob = keep_rate[i-1], seed=global_cfg.rng_seed),
                lambda: x4,
                name='dropout%d' % i)

            x4_shape = tf.shape(x4)
            new_shape = (x4_shape[0], x4_shape[1]*2, x4_shape[1]*2, x4_shape[3])
            x4 = tf.nn.conv2d_transpose(x4, kernel, new_shape, strides=[1, 2, 2, 1], name=scope.name)
    
    # x4 x5 and x6 should have the same shape now
    x = x4+x5+x6
    return x

if __name__ == "__main__":
    global_cfg.train_dir = '../../data/trained/fcn_ctscan/'
    global_cfg.device = '/gpu:0'
    global_cfg.batch_size = 4
    global_cfg.ckpt_name = 'fcn_ctscan%s.ckpt' 
    # Load data
    data_path = '../../data/patients_new/'
    dict_obj = LoadH5(data_path + 'patient_train_val_480_step2_clahe_localnorm.h5')

    train_X = np.asarray(dict_obj['train_X'], dtype=global_cfg.dtype)
    train_Y = np.asarray(dict_obj['train_Y'], dtype=global_cfg.dtype)
    val_X = np.asarray(dict_obj['val_X'], dtype=global_cfg.dtype)
    val_Y = np.asarray(dict_obj['val_Y'], dtype=global_cfg.dtype)
    
    train_weight = np.ones_like(train_Y[:,0:4,:,:])
    S = train_weight.shape
    label_5 = train_Y[:,4,:,:].reshape(S[0], 1, S[2], S[3])
    label_5 = label_5.repeat(S[1], 1).astype(np.bool)
    label_4 = train_Y[:,3,:,:].reshape(S[0], 1, S[2], S[3])
    label_4 = label_4.repeat(S[1], 1).astype(np.bool)
    train_weight[label_5] = 0 # Does not take label 5 into account when training
    train_weight[label_4] = 0.2 # The <Other> class only weights about 70% of it should be
    train_Y = train_Y[:,0:4,:,:]

    del label_4

    val_weight = np.ones_like(val_Y[:, 0:4, :, :])
    S = val_weight.shape
    label_5 = val_Y[:,4,:,:].reshape(S[0], 1, S[2], S[3])
    label_5 = label_5.repeat(S[1], 1).astype(np.bool)
    val_weight[label_5] = 0
    val_Y = val_Y[:,0:4,:,:]

    del label_5

    # Swap dims to fit NHWC format
    train_X = np.transpose(train_X, [0,2,3,1])
    train_Y = np.transpose(train_Y, [0,2,3,1])
    train_weight = np.transpose(train_weight, [0,2,3,1])
    val_X = np.transpose(val_X, [0,2,3,1])
    val_Y = np.transpose(val_Y, [0,2,3,1])
    val_weight = np.transpose(val_weight, [0,2,3,1])

    global_cfg.num_train = train_X.shape[0]
    global_cfg.num_val = val_X.shape[0]

    # Create train and val global_step to go with summary
    train_global_step = tf.Variable(0, trainable=False, name='train_global_step')
    val_global_step = tf.Variable(0, trainable=False, name='val_global_step')
    tgs_op = tf.assign(train_global_step, train_global_step + 1) 
    vgs_op = tf.assign(val_global_step, val_global_step + 1) 

    net_output = create_fcn()
    y = tf.placeholder(global_cfg.dtype, name='target')
    weight = tf.placeholder(global_cfg.dtype, name='weight')
    l = loss()
    l.l2_loss('weight', 0.0001)
    l.softmax_log_loss(net_output, y, weight)
    total_loss = l.total_loss()
    
    t = trainer(total_loss)
    t.create_adam_optimizer(lr=0.0005)
    t_ = t.get_trainer()

    num_ite_per_epoch = int(
            np.ceil(float(global_cfg.num_train)/float(global_cfg.batch_size))
            )
    num_val_ite = int(
            np.ceil(float(global_cfg.num_val)/float(global_cfg.batch_size))
            )
    
    tf.global_variables_initializer()

    merged = tf.summary.merge_all()
    summary_writer_train = tf.summary.FileWriter(global_cfg.train_dir + '/train', total_loss.graph)
    summary_writer_val = tf.summary.FileWriter(global_cfg.train_dir + '/val')
    
    saver = tf.train.Saver()
    ckpt_list = glob.glob((global_cfg.train_dir + (global_cfg.ckpt_name % '*')) + '*')
    if (len(ckpt_list) > 0):
        ckpt_list = sorted(ckpt_list) 
        ckpt = ckpt_list[-1]
        # Get the save number
        save_num = int(re.findall(r'\d+', ckpt)[-1])
        ckpt = global_cfg.train_dir + (global_cfg.ckpt_name % ('%07d' % save_num))
    else:
        # If nothing found, start from the pre-trained net
        pretrain_dir = '../../data/trained/fcn_pretrain1/'
        ckpt_list = glob.glob(pretrain_dir + 'fcn_pretrain*')
        if (len(ckpt_list) > 0):
            # Load pretrained weights
            ckpt_list = sorted(ckpt_list)
            ckpt = ckpt_list[-1]
            save_num = int(re.findall(r'\d+', ckpt)[-1])
        else:
            # With this, an entire new model will be created
            ckpt = None
            save_num = 0
    
    sess_config = tf.ConfigProto(allow_soft_placement = True)
    log_str = '%s loss: %.3f at step %d/%d, speed is %.2f sample/sec'
    
    with tf.train.MonitoredTrainingSession(
            checkpoint_dir=global_cfg.train_dir,
            hooks=[tf.train.NanTensorHook(total_loss),
                save_hook(saver, global_cfg.train_dir + global_cfg.ckpt_name, global_cfg.save_freq, save_num)],
            config=sess_config
            ) as sess:
        
        for s in range(save_num, global_cfg.max_step):
            t0 = time.time()
            i = s % num_ite_per_epoch
            
            # Uniform sampling over the entire train set to avoid early overfitting
            batch_range = range(i, global_cfg.num_train, num_ite_per_epoch)  
            batch_x = train_X[batch_range, :, :, :]
            batch_y = train_Y[batch_range, :, :, :]
            batch_weight = train_weight[batch_range, :, :, :]
            
            train_summary, train_loss, _, tgs = sess.run(
                    [merged, 'total_loss:0', t_, tgs_op],
                    feed_dict={'input_img:0': batch_x, 'target:0': batch_y, 'weight:0': batch_weight, 'is_train:0': 1})
            
            # If use tboard, start adding summary
            if (global_cfg.use_tboard):
                #pdb.set_trace()
                summary_writer_train.add_summary(train_summary, tgs)

            # Print live log on screen
            if (s % global_cfg.log_freq == 0):
               print(log_str % ('Train', 
                   train_loss,
                   s,
                   global_cfg.max_step, 
                   float(global_cfg.batch_size*global_cfg.log_freq)/(time.time()-t0)))
          
            # Validation
            if ((s+1) % global_cfg.val_freq == 0):    
                for v_s in range(num_val_ite):
                    t0 = time.time()
                    batch_range = range(v_s, global_cfg.num_val, num_val_ite) 
                    batch_x = val_X[batch_range, :, :, :]
                    batch_y = val_Y[batch_range, :, :, :]
                    batch_weight = val_weight[batch_range, :, :, :]

                    try: 
                        val_summary, val_loss, vgs = sess.run(
                            [merged, 'total_loss:0', vgs_op], 
                            feed_dict={'input_img:0': batch_x, 'target:0': batch_y, 'weight:0': batch_weight, 'is_train:0':0})
                    except Exception as Er:
                        pdb.set_trace()

                    if (global_cfg.use_tboard): 
                        summary_writer_val.add_summary(val_summary, vgs)

                    print(log_str % ('Val', val_loss, v_s, num_val_ite, float(global_cfg.batch_size)/(time.time()-t0))) 

    

    x = 1
