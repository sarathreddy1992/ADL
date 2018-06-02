import tensorflow as tf
import ATN.basic_cnn as bcnn
import ATN.basic_ae as bae
import numpy as nnp
from ATN.util.decorator import lazy_property

class ATN:
    """
    The ATN framework.
    """

    def __init__(self, data, label_gt, p_keep, rerank):
        with tf.variable_scope('autoencoder'):
            self._autoencoder = bae.BasicAE(data)
        with tf.variable_scope('target') as scope:
            self._target_adv = bcnn.BasicCnn(
                self._autoencoder.prediction, label_gt, p_keep
            )
            scope.reuse_variables()
            self._target = bcnn.BasicCnn(data, label_gt, p_keep)
        self.data = data
        self.rerank = rerank
        self.prediction
        self.optimization

    @lazy_property
    def optimization(self):
        loss_beta = 0.01
        learning_rate = 0.0001

        y_pred = self._autoencoder.prediction
        # print('shape of y_pred : ',y_pred)
        y_true = tf.reshape(self.data, [-1, 1024])
#        tf.reshape(self.data, [-1, 1024])
#         print('shape of y_true : ',y_true)

        Lx = loss_beta * tf.reduce_sum(
            tf.sqrt(tf.reduce_sum((y_pred-y_true)**2, 1))
        )
        Ly = tf.reduce_sum(tf.sqrt(tf.reduce_sum(
            (self._target_adv.prediction-self.rerank)**2, 1))
        )
        loss = Lx + Ly

        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(
            loss,
            var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                       "autoencoder"))
        return optimizer, loss

    @lazy_property
    def prediction(self):
        return self._autoencoder.prediction

    def load(self, sess, path, prefix="ATN_"):
        self._autoencoder.load(sess, path, name=prefix+'basic_ae.ckpt')
        self._target.load(sess, path+'/BasicCNN')

    def save(self, sess, path, prefix="ATN_"):
        self._autoencoder.save(sess, path, name=prefix+'basic_ae.ckpt')
        self._target.save(sess, path + '/BasicCNN')
