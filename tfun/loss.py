import tensorflow as tf
from global_config import global_config
global_cfg = global_config()

class loss(object):
    """
    This class work by adding loss values (computed from the net + target values) into the loss collection, named [lkey]
    Before running actual computation, it will collect all added losses from the collection and add them together
    """
    def __init__(self): 
	"""__init__
        Init a loss instance using global_cfg
        """

        self.use_tboard = global_cfg.use_tboard
        self.lkey = global_cfg.lkey 

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
        lsm = xdev - tf.log(tf.reduce_sum(tf.exp(xdev), keep_dims=True, reduction_indices=[-1]))
        if (target_weight == None):
            target_weight=1
        l = -tf.reduce_mean(target_weight*target*lsm, name='softmax_log_loss')
        tf.add_to_collection(self.lkey, l)
        if (self.use_tboard):
            tf.summary.scalar('softmax_log_loss', l)

    def l2_loss(self, wkey, lm):
        """l2_loss
	Compute l2 weight decay of all trainable variables with name == wkey
	 
        :param wkey: string tag, any trainable variable having this tag will be included in l2 loss
        :param lm: param indicate the important rate of this loss comparing to others
        """
        all_var = tf.trainable_variables()
        for var in all_var:
            if (wkey in var.op.name):
                l = tf.multiply(tf.nn.l2_loss(var), lm, name='weight_loss')
                tf.add_to_collection(self.lkey, l)
                if self.use_tboard:
                    tf.summary.scalar(var.op.name + '/weight_loss', l)

    def total_loss(self):
        """total_loss
        Compute total loss from all in the collection

        """
        l = tf.add_n(tf.get_collection(self.lkey), name='total_loss')
        if self.use_tboard:
            tf.summary.scalar('total_loss', l)
        return l
