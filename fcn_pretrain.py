from tfun.loss import loss
from tfun.trainer import trainer
from tfun.util import create_one_hot, create_conv_layer, LoadH5
import tfun.global_config as global_cfg

import tensorflow as tf
import numpy as np

import re
import glob
import os.path

import time
import pdb

def create_classifier_dnn():
     
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

    keep_rate = 0.8
    x = tf.cond(is_train,
            lambda: tf.nn.dropout(x, keep_prob = keep_rate, seed=global_cfg.rng_seed),
            lambda: x,
            name='dropout1')

    # CONV STACK 2
    kernel_shape = (3, 3, 32, 64)
    with tf.variable_scope('conv2_1') as scope:
        x = create_conv_layer(x, kernel_shape, use_bias=True, activation=tf.nn.relu, name=scope.name)

    x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1,2,2,1], padding = 'VALID', name = 'pool2')

    keep_rate = 0.7
    x = tf.cond(is_train,
            lambda: tf.nn.dropout(x, keep_prob = keep_rate, seed=global_cfg.rng_seed),
            lambda: x,
            name='dropout2')

    # CONV STACK 3
    kernel_shape = (3, 3, 64, 128)
    with tf.variable_scope('conv3_1') as scope:
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
            name='dropout4')

    # CONV STACK 6
    kernel_shape = (3, 3, 512, 512)
    with tf.variable_scope('conv6_1') as scope:
        x = create_conv_layer(x, kernel_shape, use_bias=True, activation=tf.nn.relu, name=scope.name)

    kernel_shape = (3, 3, 512, 512)
    with tf.variable_scope('conv6_2') as scope:
        x = create_conv_layer(x, kernel_shape, use_bias=True, activation=tf.nn.relu, name=scope.name)

    x = tf.cond(is_train,
            lambda: tf.nn.dropout(x, keep_prob = keep_rate, seed=global_cfg.rng_seed),
            lambda: x,
            name='dropout4')

    # This is basically a fully connected layer under conv2D disguise
    kernel_shape = (2, 2, 512, 4)
    with tf.variable_scope('conv6_3') as scope:
        x = create_conv_layer(x, kernel_shape, use_bias=True, activation=tf.nn.relu, name=scope.name, padding='VALID')
    return x

class save_hook(tf.train.SessionRunHook):
    def __init__(self, saver, file_name, save_freq, num):
        super(save_hook, self).__init__()
        self.saver = saver
        self.file_name = file_name
        self.save_freq = save_freq
        self.num = num
        self.init = False
        

    def before_run(self, run_context):
        file_path = self.file_name % ('%07d' % self.num)
        if (not os.path.exists(file_path)):
            self.init = True

        if (not self.init):
            session = run_context.session
            file_path = self.file_name % ('%07d' % self.num)
            saver.restore(session, file_path)
            self.init = True
            print('Loaded checkpoint at %s' % file_path)

    def after_run(self, run_context, run_values):
        session = run_context.session
        self.num += 1 
        if (self.num % self.save_freq == 0): 
            file_path = self.file_name % ('%07d' % self.num)
            self.saver.save(session, file_path)           
            print('Save checkpoint at %s' % file_path) 

if __name__ == '__main__':
    global_cfg.device = '/gpu:0'  
    global_cfg.train_dir = '../../data/trained/fcn_pretrain/'
    global_cfg.batch_size = 64
    global_cfg.ckpt_name = 'fcn_pretrain%s.ckpt'

    data_path = '../../data/patients_new/'
    dict_obj = LoadH5(data_path + 'patient_train_val_pixel.h5')
    
    # X's shape is [N, in_dim, H, W]
    # Y's shape is [N, num_class, 1, 1]
    train_X = dict_obj['train_X']
    train_Y = dict_obj['train_Y']
    val_X = dict_obj['val_X']
    val_Y = dict_obj['val_Y']
    
    # Convert X and Y to tf shape format
    train_X = train_X.transpose((0, 2, 3, 1))
    train_Y = train_Y.transpose((0, 2, 3, 1))
    val_X = val_X.transpose((0, 2, 3, 1))
    val_Y = val_Y.transpose((0, 2, 3, 1))
    
    # Convert data to desired dtype
    train_X = train_X.astype(global_cfg.dtype)
    train_Y = train_Y.astype(global_cfg.dtype)
    val_X = val_X.astype(global_cfg.dtype)  
    val_Y = val_Y.astype(global_cfg.dtype)

    global_cfg.num_train = train_X.shape[0]
    global_cfg.num_val = val_X.shape[0]

    # Create train and val global_step to go with summary
    train_global_step = tf.Variable(0, trainable=False, name='train_global_step')
    val_global_step = tf.Variable(0, trainable=False, name='val_global_step')
    tgs_op = tf.assign(train_global_step, train_global_step + 1) 
    vgs_op = tf.assign(val_global_step, val_global_step + 1) 
     
    net_output = create_classifier_dnn()
    y = tf.placeholder(global_cfg.dtype, name='target')
    
    # Create loss object
    l = loss()
    l.l2_loss('weight', 0.0001)
    l.softmax_log_loss(net_output, y)
    total_loss = l.total_loss()

    # Create trainer object
    t = trainer(total_loss)
    t.create_adam_optimizer(lr=0.0002)
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
            
            train_summary, train_loss, _, tgs = sess.run(
                    [merged, 'total_loss:0', t_, tgs_op],
                    feed_dict={'input_img:0': batch_x, 'target:0': batch_y, 'is_train:0': 1})
            
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

                    try: 
                        val_summary, val_loss, vgs = sess.run(
                            [merged, 'total_loss:0', vgs_op], 
                            feed_dict={'input_img:0': batch_x, 'target:0': batch_y, 'is_train:0':0})
                    except Exception as Er:
                        pdb.set_trace()

                    if (global_cfg.use_tboard): 
                        summary_writer_val.add_summary(val_summary, vgs)

                    print(log_str % ('Val', val_loss, v_s, num_val_ite, float(global_cfg.batch_size)/(time.time()-t0))) 

