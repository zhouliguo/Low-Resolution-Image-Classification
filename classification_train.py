from collections import namedtuple
import random
import time
import numpy as np
import tensorflow as tf
import six
import cv2
import os
from tensorflow.python.training import moving_averages
from util import get_image
from util import make_shuffle_idx


HParams = namedtuple('HParams',
                     'batch_size, num_classes, min_lrn_rate, lrn_rate, '
                     'num_residual_units, use_bottleneck, weight_decay_rate, '
                     'relu_leakiness, optimizer')

hps = HParams(batch_size=64, num_classes=10, min_lrn_rate=0.0001, lrn_rate=0.1, num_residual_units=18, use_bottleneck=True, weight_decay_rate=0.0002, relu_leakiness=0.1, optimizer='mom')
extra_train_ops = []
mode='train'

def decay():
    """L2 weight decay loss."""
    costs = []
    for var in tf.trainable_variables():
      if var.op.name.find(r'DW') > 0:
        costs.append(tf.nn.l2_loss(var))
        # tf.summary.histogram(var.op.name, var)

    return tf.multiply(hps.weight_decay_rate, tf.add_n(costs))

def fully_connected(x, out_dim):
    """FullyConnected layer for final output."""
    x = tf.reshape(x, [hps.batch_size, -1])
    w = tf.get_variable(
        'DW', [x.get_shape()[1], out_dim],
        initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
    b = tf.get_variable('biases', [out_dim],
                        initializer=tf.constant_initializer())
    return tf.nn.xw_plus_b(x, w, b)

def global_avg_pool(x):
    assert x.get_shape().ndims == 4
    return tf.reduce_mean(x, [1, 2])

def batch_norm(name, x):
    """Batch normalization."""
    with tf.variable_scope(name):
      params_shape = [x.get_shape()[-1]]

      beta = tf.get_variable(
          'beta', params_shape, tf.float32,
          initializer=tf.constant_initializer(0.0, tf.float32))
      gamma = tf.get_variable(
          'gamma', params_shape, tf.float32,
          initializer=tf.constant_initializer(1.0, tf.float32))

      if mode == 'train':
        mean, variance = tf.nn.moments(x, [0, 1, 2], name='moments')

        moving_mean = tf.get_variable(
            'moving_mean', params_shape, tf.float32,
            initializer=tf.constant_initializer(0.0, tf.float32),
            trainable=False)
        moving_variance = tf.get_variable(
            'moving_variance', params_shape, tf.float32,
            initializer=tf.constant_initializer(1.0, tf.float32),
            trainable=False)

        extra_train_ops.append(moving_averages.assign_moving_average(
            moving_mean, mean, 0.9))
        extra_train_ops.append(moving_averages.assign_moving_average(
            moving_variance, variance, 0.9))
      else:
        mean = tf.get_variable(
            'moving_mean', params_shape, tf.float32,
            initializer=tf.constant_initializer(0.0, tf.float32),
            trainable=False)
        variance = tf.get_variable(
            'moving_variance', params_shape, tf.float32,
            initializer=tf.constant_initializer(1.0, tf.float32),
            trainable=False)
        tf.summary.histogram(mean.op.name, mean)
        tf.summary.histogram(variance.op.name, variance)
      # epsilon used to be 1e-5. Maybe 0.001 solves NaN problem in deeper net.
      y = tf.nn.batch_normalization(
          x, mean, variance, beta, gamma, 0.001)
      y.set_shape(x.get_shape())
      return y

def relu(x, leakiness=0.0):
    """Relu, with optional leaky support."""
    return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

def residual(x, in_filter, out_filter, stride,
                activate_before_residual=False):
    """Residual unit with 2 sub layers."""
    if activate_before_residual:
      with tf.variable_scope('shared_activation'):
        x = batch_norm('init_bn', x)
        x = relu(x, hps.relu_leakiness)
        orig_x = x
    else:
      with tf.variable_scope('residual_only_activation'):
        orig_x = x
        x = batch_norm('init_bn', x)
        x = relu(x, hps.relu_leakiness)

    with tf.variable_scope('sub1'):
      x = conv('conv1', x, 3, in_filter, out_filter, stride)

    with tf.variable_scope('sub2'):
      x = batch_norm('bn2', x)
      x = relu(x, hps.relu_leakiness)
      x = conv('conv2', x, 3, out_filter, out_filter, [1, 1, 1, 1])

    with tf.variable_scope('sub_add'):
      if in_filter != out_filter:
        orig_x = tf.nn.avg_pool(orig_x, stride, stride, 'VALID')
        orig_x = tf.pad(
            orig_x, [[0, 0], [0, 0], [0, 0],
                     [(out_filter-in_filter)//2, (out_filter-in_filter)//2]])
      x += orig_x

    tf.logging.debug('image after unit %s', x.get_shape())
    return x

def bottleneck_residual(x, in_filter, out_filter, stride,
                           activate_before_residual=False):
    """Bottleneck residual unit with 3 sub layers."""
    if activate_before_residual:
      with tf.variable_scope('common_bn_relu'):
        x = batch_norm('init_bn', x)
        x = relu(x, hps.relu_leakiness)
        orig_x = x
    else:
      with tf.variable_scope('residual_bn_relu'):
        orig_x = x
        x = batch_norm('init_bn', x)
        x = relu(x, hps.relu_leakiness)

    with tf.variable_scope('sub1'):
      x = conv('conv1', x, 1, in_filter, out_filter/4, stride)

    with tf.variable_scope('sub2'):
      x = batch_norm('bn2', x)
      x = relu(x, hps.relu_leakiness)
      x = conv('conv2', x, 3, out_filter/4, out_filter/4, [1, 1, 1, 1])

    with tf.variable_scope('sub3'):
      x = batch_norm('bn3', x)
      x = relu(x, hps.relu_leakiness)
      x = conv('conv3', x, 1, out_filter/4, out_filter, [1, 1, 1, 1])

    with tf.variable_scope('sub_add'):
      if in_filter != out_filter:
        orig_x = conv('project', orig_x, 1, in_filter, out_filter, stride)
      x += orig_x

    tf.logging.info('image after unit %s', x.get_shape())
    return x

def stride_arr(stride):
    """Map a stride scalar to the stride array for tf.nn.conv2d."""
    return [1, stride, stride, 1]

def conv(name, x, filter_size, in_filters, out_filters, strides):
    """Convolution."""
    with tf.variable_scope(name):
      n = filter_size * filter_size * out_filters
      kernel = tf.get_variable(
          'DW', [filter_size, filter_size, in_filters, out_filters],
          tf.float32, initializer=tf.random_normal_initializer(
              stddev=np.sqrt(2.0/n)))
      return tf.nn.conv2d(x, kernel, strides, padding='SAME')

def model(images, labels):
    """Build the core model within the graph."""
    with tf.variable_scope('init'):
      x = images
      x = conv('init_conv', x, 3, 3, 16, stride_arr(1))

    strides = [1, 2, 2]
    activate_before_residual = [True, False, False]
    if hps.use_bottleneck:
      res_func = bottleneck_residual
      filters = [16, 64, 128, 256]
    else:
      res_func = residual
      filters = [16, 16, 32, 64]
      # Uncomment the following codes to use w28-10 wide residual network.
      # It is more memory efficient than very deep residual network and has
      # comparably good performance.
      # https://arxiv.org/pdf/1605.07146v1.pdf
      # filters = [16, 160, 320, 640]
      # Update hps.num_residual_units to 4

    with tf.variable_scope('unit_1_0'):
      x = res_func(x, filters[0], filters[1], stride_arr(strides[0]),
                   activate_before_residual[0])
    for i in six.moves.range(1, hps.num_residual_units):
      with tf.variable_scope('unit_1_%d' % i):
        x = res_func(x, filters[1], filters[1], stride_arr(1), False)

    with tf.variable_scope('unit_2_0'):
      x = res_func(x, filters[1], filters[2], stride_arr(strides[1]),
                   activate_before_residual[1])
    for i in six.moves.range(1, hps.num_residual_units):
      with tf.variable_scope('unit_2_%d' % i):
        x = res_func(x, filters[2], filters[2], stride_arr(1), False)

    with tf.variable_scope('unit_3_0'):
      x = res_func(x, filters[2], filters[3], stride_arr(strides[2]),
                   activate_before_residual[2])
    for i in six.moves.range(1, hps.num_residual_units):
      with tf.variable_scope('unit_3_%d' % i):
        x = res_func(x, filters[3], filters[3], stride_arr(1), False)

    with tf.variable_scope('unit_last'):
      x = batch_norm('final_bn', x)
      x = relu(x, hps.relu_leakiness)
      x = global_avg_pool(x)

    with tf.variable_scope('logit'):
      logits = fully_connected(x, hps.num_classes)
      predictions = tf.nn.softmax(logits)

    with tf.variable_scope('costs'):
      xent = tf.nn.softmax_cross_entropy_with_logits(
          logits=logits, labels=labels)
      cost = tf.reduce_mean(xent, name='xent')
      cost += decay()
    return cost

if __name__=="__main__":
    x_s = tf.placeholder(tf.float32, [hps.batch_size, None, None, 3])
    y_s = tf.placeholder(tf.float32, [hps.batch_size, hps.num_classes])

    cost = model(x_s, y_s)
    lrn_rate = tf.Variable(hps.lrn_rate, tf.float32)
    tf.summary.scalar('learning_rate', lrn_rate)

    trainable_variables = tf.trainable_variables()
    grads = tf.gradients(cost, trainable_variables)

    if hps.optimizer == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(lrn_rate)
    elif hps.optimizer == 'mom':
        optimizer = tf.train.MomentumOptimizer(lrn_rate, 0.9)

    global_step=tf.train.get_or_create_global_step()
    apply_op = optimizer.apply_gradients(zip(grads, trainable_variables), global_step=global_step , name='train_step')

    train_ops = [apply_op] + extra_train_ops
    train = tf.group(*train_ops)

    labels = []
    image_paths = []

    folders = os.listdir('./data/cifar-10/train_dir_w5x2')
    for i in range(10):
        image_path = tf.gfile.Glob(os.path.join('./data/cifar-10/train_dir_w5x2/'+folders[i], '*.*g'))
        image_paths += image_path

        label = np.ones(len(image_path), np.int)
        if hps.num_classes == 10:
            label = label*i
        else:
            label = label*int(folders[i])
        labels += list(label)

    shuffle_idx = make_shuffle_idx(len(labels))
    image_paths = [image_paths[i] for i in shuffle_idx]
    labels = [labels[i] for i in shuffle_idx]

    image_batch = np.zeros((hps.batch_size, 64, 64, 3), np.float32)
    label_batch = np.zeros((hps.batch_size, 10), np.float32)
    label_batch_ = np.zeros((hps.batch_size), np.int32)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer()) 
        #saver.restore(sess,'models/res164_90000')

        k = 0
        sum_loss = 0
        for i in range(180000):
            label_batch = label_batch*0
            for j in range(hps.batch_size):
                image_batch[j] = get_image(image_paths[k])
                label_batch[j][labels[k]] = 1.0
                k = k+1
                if k==len(labels):
                    shuffle_idx = make_shuffle_idx(len(labels))
                    image_paths = [image_paths[ii] for ii in shuffle_idx]
                    labels = [labels[ii] for ii in shuffle_idx]
                    k=0
            _, loss = sess.run((train, cost), feed_dict={x_s: image_batch, y_s: label_batch})
            sum_loss = sum_loss+loss

            if i+1 == 80000:
                sess.run(tf.assign(lrn_rate, 0.01))
                print(sess.run(optimizer._learning_rate))
            if i+1 == 120000:
                sess.run(tf.assign(lrn_rate, 0.001))
                print(sess.run(optimizer._learning_rate))
            if i+1 == 160000:
                sess.run(tf.assign(lrn_rate, 0.0001))
                print(sess.run(optimizer._learning_rate))
            if (i+1)%100 == 0:
                if (i+1)%5000 == 0:
                    saver.save(sess, './models/cifar-10/models_w5x2_n_p4/res164_x2_'+str(i+1))
                #print(i, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                print(i, sum_loss/100, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                sum_loss = 0
