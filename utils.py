import numpy as np
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.contrib.layers.python.layers import utils

## TensorFlow helper functions

WEIGHT_DECAY_KEY = 'WEIGHT_DECAY'


def _relu(x, leakness=0.0, name=None):
    if leakness > 0.0:
        name = 'lrelu' if name is None else name
        return tf.maximum(x, x * leakness, name='lrelu')
    else:
        name = 'relu' if name is None else name
        return tf.nn.relu(x, name='relu')


def _conv(x, filter_size, out_channel, strides, pad='SAME', name='conv', hyper=None):
    in_shape = x.get_shape()
    with tf.variable_scope(name):
        if hyper:
            kernel = hyper._create_conv_weight(shape=[filter_size, filter_size, in_shape[3], out_channel],
                                               scope='kernel')
        else:
            kernel = tf.get_variable('kernel', [filter_size, filter_size, in_shape[3], out_channel],
                                     tf.float32, initializer=tf.random_normal_initializer(
                    stddev=np.sqrt(2.0 / filter_size / filter_size / out_channel)))
        if kernel not in tf.get_collection(WEIGHT_DECAY_KEY):
            tf.add_to_collection(WEIGHT_DECAY_KEY, kernel)
            # print('\tadded to WEIGHT_DECAY_KEY: %s(%s)' % (kernel.name, str(kernel.get_shape().as_list())))
        # show kernel mean and std
        u, s = tf.nn.moments(tf.layers.flatten(kernel), axes=0)
        tf.summary.scalar(kernel.name + '/mean', u[0])
        tf.summary.scalar(kernel.name + '/std', s[0])
        # show kernel as image
        transposed_kernel = tf.transpose(kernel, [2, 0, 1, 3])
        a, b, c, d = transposed_kernel.get_shape().as_list()
        reshaped_kernel = tf.reshape(transposed_kernel, [1, a * b, c * d, 1])
        tf.summary.image(kernel.name, reshaped_kernel)
        conv = tf.nn.conv2d(x, kernel, [1, strides, strides, 1], pad)
    return conv


def _fc(x, out_dim, name='fc'):
    with tf.variable_scope(name):
        w = tf.get_variable('weights', [x.get_shape()[1], out_dim],
                            tf.float32, initializer=tf.random_normal_initializer(
                stddev=np.sqrt(1.0 / out_dim)))
        if w not in tf.get_collection(WEIGHT_DECAY_KEY):
            tf.add_to_collection(WEIGHT_DECAY_KEY, w)
            # print('\tadded to WEIGHT_DECAY_KEY: %s(%s)' % (w.name, str(w.get_shape().as_list())))
        b = tf.get_variable('biases', [out_dim], tf.float32,
                            initializer=tf.constant_initializer(0.0))
        fc = tf.nn.bias_add(tf.matmul(x, w), b)
    return fc


def _bn(x, is_train, global_step=None, name='bn'):
    moving_average_decay = 0.9
    # moving_average_decay = 0.99
    # moving_average_decay_init = 0.99
    with tf.variable_scope(name):
        decay = moving_average_decay
        # if global_step is None:
        # decay = moving_average_decay
        # else:
        # decay = tf.cond(tf.greater(global_step, 100)
        # , lambda: tf.constant(moving_average_decay, tf.float32)
        # , lambda: tf.constant(moving_average_decay_init, tf.float32))
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2])
        mu = tf.get_variable('mu', batch_mean.get_shape(), tf.float32,
                             initializer=tf.zeros_initializer, trainable=False)
        sigma = tf.get_variable('sigma', batch_var.get_shape(), tf.float32,
                                initializer=tf.ones_initializer, trainable=False)
        beta = tf.get_variable('beta', batch_mean.get_shape(), tf.float32,
                               initializer=tf.zeros_initializer)
        gamma = tf.get_variable('gamma', batch_var.get_shape(), tf.float32,
                                initializer=tf.ones_initializer)
        # BN when training
        update = 1.0 - decay
        # with tf.control_dependencies([tf.Print(decay, [decay])]):
        # update_mu = mu.assign_sub(update*(mu - batch_mean))
        update_mu = mu.assign_sub(update * (mu - batch_mean))
        update_sigma = sigma.assign_sub(update * (sigma - batch_var))
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_mu)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_sigma)

        mean, var = tf.cond(is_train, lambda: (batch_mean, batch_var),
                            lambda: (mu, sigma))
        bn = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-5)

        # bn = tf.nn.batch_normalization(x, batch_mean, batch_var, beta, gamma, 1e-5)

        # bn = tf.contrib.layers.batch_norm(inputs=x, decay=decay,
        # updates_collections=[tf.GraphKeys.UPDATE_OPS], center=True,
        # scale=True, epsilon=1e-5, is_training=is_train,
        # trainable=True)
    return bn


class Hyper(object):
    """Initializer that generates tensors with a uniform distribution.

    Args:
      minval: A python scalar or a scalar tensor. Lower bound of the range
        of random values to generate.
      maxval: A python scalar or a scalar tensor. Upper bound of the range
        of random values to generate.  Defaults to 1 for float types.
      seed: A Python integer. Used to create random seeds. See
        @{tf.set_random_seed}
        for behavior.
      dtype: The data type.
    """

    def __init__(self, f_size=1, in_size=64, out_size=64, z_dim=64, name='Hyper'):
        self.f_size = f_size
        self.in_size = in_size
        self.out_size = out_size
        self.z_dim = z_dim
        self.name = name
        self.kernel_initializer = tf.truncated_normal_initializer(stddev=0.01)
        self.bias_initializer = tf.constant_initializer(0.0)
        # batch norm params
        self._BATCH_NORM_DECAY = 0.997
        self._BATCH_NORM_EPSILON = 1e-5
        with tf.variable_scope(self.name):
            # create embedding
            # load hypyer params
            self.w1 = tf.get_variable('w1', shape=[self.z_dim, self.in_size * self.z_dim],
                                      dtype=tf.float32, initializer=self.kernel_initializer)
            self.b1 = tf.get_variable('b1', shape=[self.in_size * self.z_dim],
                                      dtype=tf.float32, initializer=self.bias_initializer)
            self.w2 = tf.get_variable('w2', shape=[self.z_dim, self.f_size * self.out_size * self.f_size],
                                      dtype=tf.float32, initializer=self.kernel_initializer)
            self.b2 = tf.get_variable('b2', shape=[self.f_size * self.out_size * self.f_size],
                                      dtype=tf.float32, initializer=self.bias_initializer)

    def _create_conv_weight(self, shape, scope=None):
        k1_size, k2_size, dim_in, dim_out = shape
        if not (self.f_size == k1_size and self.f_size == k2_size):
            raise Exception('kernel size error')
        if dim_in % self.in_size != 0:
            raise Exception('dim_in%%in_size=%d' % (dim_in % self.in_size))
        if dim_out % self.out_size != 0:
            raise Exception('dim_out%%out_size=%d' % (dim_out % self.out_size))
        # embedding vector
        # emb = tf.Variable(
        #    tf.random_normal([dim_in // self.in_size, dim_out // self.out_size, self.z_dim], dtype=dtype, stddev=0.01),
        #    name='emb')
        emb = tf.get_variable(name=scope + 'embedding',
                              shape=[dim_in // self.in_size, dim_out // self.out_size, self.z_dim],
                              initializer=tf.truncated_normal_initializer(stddev=0.01))

        in_list = []
        for i in range(dim_in // self.in_size):
            out_list = []
            for j in range(dim_out // self.out_size):
                # row = tf.nn.embedding_lookup(emb, i)
                # z = tf.nn.embedding_lookup(row, j)
                w1, b1, w2, b2 = self.w1, self.b1, self.w2, self.b2
                # z = tf.reshape(z, [-1, self.z_dim])
                z = tf.reshape(emb[i, j, :], [-1, self.z_dim])
                # create conv weight
                a = tf.matmul(z, w1) + b1
                a = tf.reshape(a, [self.in_size, self.z_dim])
                weight = tf.matmul(a, w2) + b2
                weight = tf.reshape(weight, [self.in_size, self.out_size, self.f_size, self.f_size])
                out_list.append(weight)
                # concat
            out_weight = tf.concat(out_list, axis=1)
            in_list.append(out_weight)
        # concat
        conv_weight = tf.concat(in_list, axis=0)
        conv_weight = tf.transpose(conv_weight, [2, 3, 0, 1])
        return conv_weight

    def get_config(self):
        return {
            'f_size': self.f_size,
            'in_size': self.in_size,
            'out_size': self.out_size,
            'z_dim': self.z_dim,
            'name': self.name,
        }

    def conv2d(self, inputs, filters, kernel_size, stride=1, padding='VALID', use_bias=True, scope=None,
               collections=None):
        if kernel_size != self.f_size and kernel_size != [self.f_size, self.f_size]:
            raise Exception('kernel_size must be the same with f_size')
        dim_in = inputs.get_shape().as_list()[-1]
        # create conv weight
        conv_weight = self._create_conv_weight([self.f_size, self.f_size, dim_in, filters], scope=scope)
        strides = [1, stride, stride, 1]
        outputs = tf.nn.conv2d(inputs, conv_weight, strides=strides, padding=padding)
        if use_bias:
            bias = tf.get_variable(name=scope + 'bias', shape=[filters, ], initializer=tf.constant_initializer(0.0))
            outputs += bias
        outputs = self.batch_norm_relu(inputs=outputs, is_training=True, data_format='channel_last', scope=scope)
        if collections is not None:
            utils.collect_named_outputs(collections, tf.get_variable_scope().name + '/conv_weight', conv_weight)
        return outputs

    def conv2d_same(self, inputs, num_outputs, kernel_size, stride, data_format='NHWC', rate=1, scope=None,
                    collections=None):
        if stride == 1:
            return self.conv2d(inputs, num_outputs, kernel_size, stride=1,
                               padding='SAME', scope=scope, collections=collections)
        else:
            kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
            pad_total = kernel_size_effective - 1
            pad_beg = pad_total // 2
            pad_end = pad_total - pad_beg
            padding = [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]]
            if data_format == 'NCHW':
                padding = [padding[0], padding[3], padding[1], padding[2]]
            inputs = tf.pad(inputs, padding)
            return self.conv2d(inputs, num_outputs, kernel_size, stride=stride,
                               padding='VALID', scope=scope, collections=collections)

    def batch_norm_relu(self, inputs, is_training, data_format, scope=None):
        """Performs a batch normalization followed by a ReLU."""
        # We set fused=True for a significant performance boost. See
        # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
        inputs = tf.layers.batch_normalization(
            inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
            momentum=self._BATCH_NORM_DECAY, epsilon=self._BATCH_NORM_EPSILON, center=True,
            scale=True, training=is_training, fused=True, name=scope + 'batchnorm')
        inputs = tf.nn.relu(inputs)
        return inputs
