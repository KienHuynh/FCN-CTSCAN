import tensorflow as tf
from global_config import global_config
global_cfg = global_config()
class trainer(object):
    """trainer
    This class is for creating optimizer and computing gradients
    """
    def __init__(self, global_cfg, loss_output):
        """__init__

        :param global_cfg: global config instance
        :param loss_output: the loss output, computed by calling total_loss in tfun/loss.py
        """
        self.batch_size = global_cfg.batch_size
        self.num_train = global_cfg.num_train
        self.loss_output = loss_output
        self.use_tboard = global_cfg.use_tboard
        self.optimizer = None
        self.grads = None
        self.global_step = 0
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

    def create_sgd_optimizer(self, lr):
        """create_sgd_optimizer

        :param lr: learning rate
        """
        self.optimizer = tf.train.GradientDescentOptimizer(lr)
        
    def create_adam_optimizer(self, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        """create_adam_optimizer

        :param lr: learning rate
        :param beta1: beta1 in the paper
        :param beta2: beta2 in the paper
        :param eps: epsilon in the paper
        """
        self.optimizer = tf.train.AdamOptimizer(lr, beta1, beta2, eps)

    def get_trainer():
        """get_trainer
        Return the appply grad object so that a tf session could run on it
        """
        assert self.optimizer != None, "Please create an optimizer for trainer first before calling get_trainer()"

        # Create grad computation nodes & add them to summary
        self.grads = self.optimizer.compute_gradients(self.loss_output)
        if (self.use_tboard):
            for grad, var in self.grads:
                if grad != None:
                    tf.summary.histogram(var.op.name + '/grad', grad)

        # Add trainable variables to summary histogram
        for var in tf.trainable_variables():
            tf.summary.hisogram(var.op.name, var)

        # Apply grad
        apply_grad_op = self.optimizer.apply_gradients(self.grad, global_step=self.global_step, name='train')

        return apply_grad_op 
