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

import matplotlib.pyplot as plt
import pylab

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
    
    # Reshape these tensor because after tensordot, they all have Dim(None) as shape 
    x4 = tf.reshape(x4, [-1, 60, 60, 4])
    x5 = tf.reshape(x5, [-1, 30, 30, 4])
    x6 = tf.reshape(x6, [-1, 15, 15, 4])
    
    # Creating transposed conv layers
    kernel_shape = (3, 3, 4, 4)
    
    #keep_rate = [0.6, 0.7, 0.8, 0.9, 1]
    # Do a succession of 5 transposed conv
    with tf.variable_scope('bilinear_up6') as scope:
        #kernel = create_var('weight', shape=kernel_shape, initializer=tf.contrib.layers.xavier_initializer())
        #for i in range(1, 6):
        #    x6 = tf.cond(is_train,
        #        lambda: tf.nn.dropout(x6, keep_prob = keep_rate[i-1], seed=global_cfg.rng_seed),
        #        lambda: x6,
        #        name='dropout%d' % i)
        #    x6_shape = tf.shape(x6)
        #    new_shape = (x6_shape[0], x6_shape[1]*2, x6_shape[1]*2, x6_shape[3])
        #    x6 = tf.nn.conv2d_transpose(x6, kernel, new_shape, strides=[1, 2, 2, 1], name=scope.name)
        
        x6_shape = tf.shape(x6)
        new_shape = [x6_shape[1]*(2**5), x6_shape[2]*(2**5)]
        x6 = tf.image.resize_images(x6, [new_shape[0], new_shape[1]])
        x6 = tf.identity(x6, name=scope.name)
 
    with tf.variable_scope('bilinear_up5') as scope:
        x5_shape = tf.shape(x5)
        new_shape = [x5_shape[1]*(2**4), x5_shape[2]*(2**4)]
        x5 = tf.image.resize_images(x5, [new_shape[0], new_shape[1]])
        x5 = tf.identity(x5, name=scope.name)
 
    with tf.variable_scope('bilinear_up4') as scope:
        x4_shape = tf.shape(x4, name='shape1')
        new_shape = [x4_shape[1]*(2**3), x4_shape[2]*(2**3)] 
        x4 = tf.image.resize_images(x4, [new_shape[0], new_shape[1]])
        x4 = tf.identity(x4, name=scope.name)

    # x4 x5 and x6 should have the same shape now
    x = x4+x5+x6
    x = tf.identity(x, name='logprob')
    return x

def train():
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
    t.create_adam_optimizer(lr=0.0001)
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
    
    saver = tf.train.Saver(max_to_keep = 10)
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
           
            x4_shape = sess.run(
                    ['bilinear_up4/shape1:0'],
                    feed_dict={'input_img:0': batch_x, 'target:0': batch_y, 'weight:0': batch_weight, 'is_train:0': 1}) 
            train_summary, train_loss, _, tgs = sess.run(
                    [merged, 'total_loss:0', t_, tgs_op],
                    feed_dict={'input_img:0': batch_x, 'target:0': batch_y, 'weight:0': batch_weight, 'is_train:0': 1})
            
            # If use tboard, start adding summary
            if (global_cfg.use_tboard): 
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


def test():
    global_cfg.train_dir = '../../data/trained/fcn_ctscan/'
    global_cfg.device = '/gpu:0'
    global_cfg.batch_size = 4
    global_cfg.ckpt_name = 'fcn_ctscan%s.ckpt' 
    # Load data
    data_path = '../../data/patients_new/'
    dict_obj = LoadH5(data_path + 'patient_test_480_trainstep2_clahe_localnorm.h5')

    test_X = np.asarray(dict_obj['test_X'], dtype=global_cfg.dtype)  
    test_Y = np.asarray(dict_obj['test_Y'], dtype=global_cfg.dtype)
    
    test_weight = np.ones_like(test_Y[:,0:4,:,:])
    S = test_weight.shape
    label_5 = test_Y[:,4,:,:].reshape(S[0], 1, S[2], S[3])
    label_5 = label_5.repeat(S[1], 1).astype(np.bool) 
    test_weight[label_5] = 0 # Does not take label 5 into account when testing 
    test_Y = test_Y[:,0:4,:,:]
    del label_5

    # Swap dims to fit NHWC format
    test_X = np.transpose(test_X, [0,2,3,1])
    test_Y = np.transpose(test_Y, [0,2,3,1])
    test_weight = np.transpose(test_weight, [0,2,3,1])

    global_cfg.num_test = test_X.shape[0]

    net_output = create_fcn()
    y = tf.placeholder(global_cfg.dtype, name='target')
    weight = tf.placeholder(global_cfg.dtype, name='weight')

    saver = tf.train.Saver()
    ckpt_list = glob.glob((global_cfg.train_dir + (global_cfg.ckpt_name % '*')) + '*')
    if (len(ckpt_list) > 0):
        ckpt_list = sorted(ckpt_list) 
        ckpt = ckpt_list[-1]
        # Get the save number
        save_num = int(re.findall(r'\d+', ckpt)[-1])
        ckpt = global_cfg.train_dir + (global_cfg.ckpt_name % ('%07d' % save_num))
    else:
        print('No trained model found, return...')
        return

    num_ite = int(np.ceil(float(global_cfg.num_test)/float(global_cfg.batch_size)))
    all_pred = np.zeros((global_cfg.num_test, 480, 480))
    with tf.Session() as sess:
        saver.restore(sess, ckpt)
        for i in range(num_ite):
            batch_range = range(i*global_cfg.batch_size, (i+1)*global_cfg.batch_size)
            if (batch_range[-1] >= global_cfg.num_test):
                batch_range = range(batch_range[0], global_cfg.num_test)
            
            batch_x = test_X[batch_range,:,:,:]
            batch_y = test_Y[batch_range,:,:,:]
            batch_w = test_weight[batch_range,:,:,:]
            logprob = sess.run(['logprob:0'], feed_dict={'input_img:0': batch_x, 'target:0': batch_y, 'weight:0': batch_w, 'is_train:0': 0}) 
            all_pred[batch_range,:,:] = np.argmax(logprob[0], axis=3)
         
        test_Y = np.argmax(test_Y, axis=3)+1
        all_pred = all_pred + 1
        true_pred = np.copy(all_pred)
        true_pred[all_pred != test_Y] = 0
        test_weight = test_weight[:,:,:,0]
        
        test_Y = test_Y*test_weight
        true_pred = true_pred*test_weight
        all_pred = all_pred*test_weight 

        # Saving image results
        #for i in range(global_cfg.num_test):
        #    f = plt.figure(1, figsize = (15,5)) 
        #    ax = plt.subplot(1,3,1)
        #    ax.set_title('Input CT image')
        #    plt.imshow(test_X[i, :, :, 1]) 
        #    
        #    ax = plt.subplot(1,3,2)
        #    ax.set_title('Ground truth')
        #    plt.imshow(test_Y[i, :, :])

        #    ax = plt.subplot(1,3,3)
        #    ax.set_title('Prediction')
        #    plt.imshow(all_pred[i, :, :])
        #    
        #    f.savefig('../../data/trained/fcn_ctscan/visual/%04d.png' % i, bbox_inches='tight')
        #    print('Saving images of sample %d' % i) 
        
        # Confusion mat
        conf_mat = np.zeros((4,4), dtype=global_cfg.dtype)
        for l in range(4):
            for k in range(4):
                idx = test_Y==(l+1)
                num_l_as_k = np.sum(all_pred[idx]==k+1)
                conf_mat[l, k] = num_l_as_k
        class_recall = conf_mat.flatten()[0::5]/np.sum(conf_mat, 1)
        class_precision = conf_mat.flatten()[0::5]/np.sum(conf_mat,0)

        # Pixel accuracy
        pixel_acc = float(np.sum(true_pred!=0))/float(np.sum(test_Y!=0)) 
        print('Pixel acc: %.4f' % pixel_acc)

        # Mean accuracy
        mean_acc = [0, 0, 0, 0]
        for l in range(4):        
            mean_acc[l] = float(np.sum(true_pred==(l+1)))/float(np.sum(test_Y==(l+1)))

        mean_acc = np.mean(mean_acc)
        print('Mean acc: %.4f' % mean_acc)

        # Mean IU (Intersection-Uniono) and frequency weighted IU
        mean_IU = [0,0,0,0]
        freq_IU = [0,0,0,0]
        for l in range(4):
            mean_IU[l] = float(np.sum(true_pred==(l+1)))/float(np.sum(test_Y==(l+1)) + np.sum(all_pred==(l+1)) - np.sum(true_pred==(l+1)))
            freq_IU[l] = mean_IU[l]*np.sum(test_Y==(l+1))
    
        print('Mean IU: %.4f' % np.mean(mean_IU))
        print('Frequency weighted IU: %.4f' % (np.sum(freq_IU)/float(np.sum(test_Y!=0))))

if __name__ == "__main__":
    #train()
    test()
