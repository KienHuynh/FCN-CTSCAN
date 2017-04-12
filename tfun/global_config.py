import numpy as np

dtype = np.float32
device = '/gpu:0'
train_dir = '../data/trained/'
use_tboard = True

lkey = 'loss' 
batch_size = 0
num_train = 0
num_test = 0
num_val = 0

max_step = 1000000 
log_freq = 10
val_freq = 160

save_freq = 100
ckpt_name = 'model.ckpt'

rng_seed = 1311
