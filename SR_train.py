import tensorflow as tf
import numpy as np
import random
import time
import os
import cv2
import matplotlib.pyplot as plt
from common.layers import conv2d_weight_norm
from util import make_shuffle_idx
from util import compute_psnr

scale = 2
num_channels = 3
num_residual_units = 32
num_blocks = 8

x = tf.placeholder(tf.float32, [None, None, None, 3])
y = tf.placeholder(tf.float32, [None, None, None, 3])
weights = tf.placeholder(tf.float32, [None, None, None, 3])

learning_rate = tf.Variable(0.001, dtype=tf.float32)

x = tf.identity(x, 'input')


#Network
def wdsr(x):

    def _residual_block(x, num_channels):
      skip = x
      x = conv2d_weight_norm(
          x,
          num_channels * 4,
          3,
          padding='same',
          name='conv0',
      )
      x = tf.nn.relu(x)
      x = conv2d_weight_norm(
          x,
          num_channels,
          3,
          padding='same',
          name='conv1',
      )
      return x + skip

    def _subpixel_block(x,
                        kernel_size,
                        num_channels=num_channels,
                        scale=scale):
      x = conv2d_weight_norm(
          x,
          num_channels * scale * scale,
          kernel_size,
          padding='same',
      )
      x = tf.depth_to_space(x, scale)
      return x

    MEAN = tf.constant([0.5, 0.5, 0.5], dtype=tf.float32)
    x = x - MEAN
    with tf.variable_scope('skip'):
      skip = _subpixel_block(x, 5)
    with tf.variable_scope('input'):
      x = conv2d_weight_norm(
          x,
          num_residual_units,
          3,
          padding='same',
      )
    for i in range(num_blocks):
      with tf.variable_scope('layer{}'.format(i)):
        x = _residual_block(x, num_residual_units)
    with tf.variable_scope('output'):
      x = _subpixel_block(x, 3)
    x += skip
    x = x + MEAN
    return x

y_train = wdsr(x)

train_loss = tf.losses.absolute_difference(y, y_train, weights)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    #optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
    optimizer = tf.train.AdamOptimizer(learning_rate)

train = optimizer.minimize(train_loss)

saver = tf.train.Saver()

batch_h = 128
batch_w = 128
batch_size = 3072
step = 32

batch_data = np.zeros((batch_size, batch_h, batch_w, 3), np.float32)
batch_labels = np.zeros((batch_size, batch_h*scale, batch_w*scale, 3), np.float32)
batch_weights = np.zeros((batch_size, batch_h*scale, batch_w*scale, 3), np.float32)

labelnames = tf.gfile.Glob(os.path.join('D:/div2k/DIV2K_train_HR', '*.png'))    #Ground truth HR images
labelnum = len(labelnames)
shuffle_idx = make_shuffle_idx(labelnum)
labelnames = [labelnames[i] for i in shuffle_idx]


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    j = 0
    k = 0
    step_num = 0

    for i in range(105*labelnum):
        imagename = labelnames[k][:8]+'\\DIV2K_train_LR_bicubic\\X2'+labelnames[k][23:28]+'x2'+labelnames[k][28:]   #LR images
        weightname = 'D:/div2k/DIV2K_weights_HR_g/'+labelnames[k][23:28]+'.npy' #Weight maps

        label = cv2.imread(labelnames[k]).astype(np.float32)/255
        image = cv2.imread(imagename).astype(np.float32)/255

        npy = np.load(weightname).astype(np.float32)
        mx = np.max(npy)
        mn = np.min(npy)
        npy = (npy-mn)/(mx-mn)
        
        m = np.mean(npy)
        std = np.std(npy)
        std = std/6 #set alpha here
        npy[npy>=(m+std)] = 1
        npy[npy<=(m-std)] = 1
        npy[npy!=1] = 0

        k = k+1

        #For data augment
        fr = random.randint(0,7)
        if fr == 1:
            image = cv2.rotate(image, 0)
            label = cv2.rotate(label, 0)
            npy = cv2.rotate(npy, 0)
        if fr == 2:
            image = cv2.rotate(image, 1)
            label = cv2.rotate(label, 1)
            npy = cv2.rotate(npy, 1)
        if fr == 3:
            image = cv2.rotate(image, 2)
            label = cv2.rotate(label, 2)
            npy = cv2.rotate(npy, 2)
        if fr == 4:
            image = cv2.flip(image, 1)
            label = cv2.flip(label, 1)
            npy = cv2.flip(npy, 1)
        if fr == 5:
            image = cv2.flip(image, 1)
            image = cv2.rotate(image, 0)
            label = cv2.flip(label, 1)
            label = cv2.rotate(label, 0)
            npy = cv2.flip(npy, 1)
            npy = cv2.rotate(npy, 0)
        if fr == 6:
            image = cv2.flip(image, 1)
            image = cv2.rotate(image, 1)
            label = cv2.flip(label, 1)
            label = cv2.rotate(label, 1)
            npy = cv2.flip(npy, 1)
            npy = cv2.rotate(npy, 1)
        if fr == 7:
            image = cv2.flip(image, 1)
            image = cv2.rotate(image, 2)
            label = cv2.flip(label, 1)
            label = cv2.rotate(label, 2)
            npy = cv2.flip(npy, 1)
            npy = cv2.rotate(npy, 2)

        height, width = image.shape[:2]
        h_random = random.randint(32,128)
        w_random = random.randint(32,128)

        #Training
        for h in range(0, height-batch_h+h_random, h_random):
            if h + batch_h > height:
                h = height-batch_h
            
            for w in range(0, width-batch_w+w_random, w_random):
                if w + batch_w > width:
                    w = width-batch_w
                
                batch_data[j] = image[h:h+batch_h, w:w+batch_w]
                batch_labels[j] = label[h*scale:(h+batch_h)*scale, w*scale:(w+batch_w)*scale]
                batch_weights[j] = npy[h*scale:(h+batch_h)*scale, w*scale:(w+batch_w)*scale]

                #For data augment
                fr = random.randint(0,7)
                if fr == 1:
                    batch_data[j] = cv2.rotate(batch_data[j], 0)
                    batch_labels[j] = cv2.rotate(batch_labels[j], 0)
                    batch_weights[j] = cv2.rotate(batch_weights[j], 0)
                if fr == 2:
                    batch_data[j] = cv2.rotate(batch_data[j], 1)
                    batch_labels[j] = cv2.rotate(batch_labels[j], 1)
                    batch_weights[j] = cv2.rotate(batch_weights[j], 1)
                if fr == 3:
                    batch_data[j] = cv2.rotate(batch_data[j], 2)
                    batch_labels[j] = cv2.rotate(batch_labels[j], 2)
                    batch_weights[j] = cv2.rotate(batch_weights[j], 2)
                if fr == 4:
                    batch_data[j] = cv2.flip(batch_data[j], 1)
                    batch_labels[j] = cv2.flip(batch_labels[j], 1)
                    batch_weights[j] = cv2.flip(batch_weights[j], 1)
                if fr == 5:
                    batch_data[j] = cv2.flip(batch_data[j], 1)
                    batch_data[j] = cv2.rotate(batch_data[j], 0)
                    batch_labels[j] = cv2.flip(batch_labels[j], 1)
                    batch_labels[j] = cv2.rotate(batch_labels[j], 0)
                    batch_weights[j] = cv2.flip(batch_weights[j], 1)
                    batch_weights[j] = cv2.rotate(batch_weights[j], 0)
                if fr == 6:
                    batch_data[j] = cv2.flip(batch_data[j], 1)
                    batch_data[j] = cv2.rotate(batch_data[j], 1)
                    batch_labels[j] = cv2.flip(batch_labels[j], 1)
                    batch_labels[j] = cv2.rotate(batch_labels[j], 1)
                    batch_weights[j] = cv2.flip(batch_weights[j], 1)
                    batch_weights[j] = cv2.rotate(batch_weights[j], 1)
                if fr == 7:
                    batch_data[j] = cv2.flip(batch_data[j], 1)
                    batch_data[j] = cv2.rotate(batch_data[j], 2)
                    batch_labels[j] = cv2.flip(batch_labels[j], 1)
                    batch_labels[j] = cv2.rotate(batch_labels[j], 2)
                    batch_weights[j] = cv2.flip(batch_weights[j], 1)
                    batch_weights[j] = cv2.rotate(batch_weights[j], 2)

                j = j+1
                if j == batch_size:
                    j = 0
                    shuffle_idx = make_shuffle_idx(batch_size)
                    batch_data = [batch_data[b_i] for b_i in shuffle_idx]
                    batch_labels = [batch_labels[b_i] for b_i in shuffle_idx]
                    batch_weights = [batch_weights[b_i] for b_i in shuffle_idx]

                    value_sum = 0
                    for batch_i in range(0, batch_size, step):
                        batch_i_up = batch_i+step
                        _, value = sess.run([train, train_loss], feed_dict={x: batch_data[batch_i:batch_i_up], y: batch_labels[batch_i:batch_i_up], weights:batch_weights[batch_i:batch_i_up]})
                        value_sum = value_sum+value
                        step_num = step_num+1
                    print(i, '\tstep_num:' , step_num,'\tloss:', value_sum*step/batch_size, '\tlr:', sess.run(optimizer._lr), time.strftime('\t%Y-%m-%d %H:%M:%S', time.localtime()))
        
        #Validation
        if (i+1) % 2000 == 0:
            sum_value=0
            for val in range(801, 901):
                val_label = cv2.imread('D:/div2k/DIV2K_valid_HR/0'+str(val)+'.png').astype(np.float32)
                val_image = cv2.imread('D:/div2k/DIV2K_valid_LR_bicubic/X2/0'+str(val)+'x2.png').astype(np.float32)/255

                height, width, channel = val_image.shape
                val_image = val_image.reshape((1,height, width, channel))
                
                pred = sess.run(y_train, feed_dict={x: val_image})
                pred[pred>1] = 1
                pred[pred<0] = 0
                pred = np.round(pred*255)
                value = compute_psnr(pred[0], val_label, scale)
                sum_value = sum_value+value
            print('Val_loss:\t', sum_value/100)
        

        if (i+1) % labelnum == 0:
            shuffle_idx = make_shuffle_idx(labelnum)
            labelnames = [labelnames[ii] for ii in shuffle_idx]
            k = 0
            saver.save(sess, 'models/wdsr_w_6_x2_'+str(i+1))
            if i+1 == 40*labelnum:
                sess.run(tf.assign(learning_rate, 0.0005))
            if i+1 == 70*labelnum:
                sess.run(tf.assign(learning_rate, 0.0001))
            if i+1 == 90*labelnum:
                sess.run(tf.assign(learning_rate, 0.00005))
            if i+1 == 100*labelnum:
                sess.run(tf.assign(learning_rate, 0.00001))
