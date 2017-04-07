import tensorflow as tf

class loss(object):
    def __init__(self, target, loss_cfg, global_cfg, weight=None): 
	self.track_loss = global_cfg.track_loss
        self.lkey = loss_cfg.lkey
        self.wkey = global_cfg.wkey

    def softmax_log_loss(self, X, target, target_weight=None, lm=1):
    	"""softmax_log_loss
	Compute the softmax log loss using X and target then add it to lkey collection
        Class dim is -1

    	:param X: input to be calculated in loss
	:paran target: expected output
        :param target_weight: target target_weight, use to balance between classes/remove pad sammples
    	:param lm: param indicate the important rate of this loss comparing to others
    	""" 
	xdev = X - tf.reduce_max(X, keep_dims=True, reduction_indices=[-1])
        lsm = xdev - tf.log(tf.sum(tf.exp(xdev), keep_dims=True, reduction_indices=[-1]))
        if (target_weight == None):
            target_weight=1
        l = -tf.reduce_mean(weight*target*lsm, name='softmax_log_loss')
        tf.add_to_collection(self.lkey, l)
        if (self.track_loss):
            tf.summary.scalar(l)

    def l2_loss(self, X, lm):
        """l2_loss
	Compute l2 weight decay of all trainable variables with name == self.wkey
	
        :param X: the final computation node
        :param lm: param indicate the important rate of this loss comparing to others
        """
        all_var = tf.trainable_variables()
        for var in all_var:
            if (self.wkey in var.op.name):
                l = tf.multiply(tf.nn.l2_loss(var), lm, name='weight_loss')
                tf.add_to_collection(self.lkey, l)
    

