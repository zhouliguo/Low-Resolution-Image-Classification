import numpy as np
import random
import cv2
import os
import tensorflow as tf

filenames = tf.gfile.Glob(os.path.join('D:\\div2k\\DIV2K_train_HR', '*.png'))


for filename in filenames:
    basename = os.path.basename(filename)
    if len(basename) > 8:
        continue
    print(basename)

    image = cv2.imread(filename)
    height, width, channel = image.shape
    image = image.astype(np.float32)
    weights = np.zeros((height, width, channel), np.float32)
    temp = np.zeros((height, width, channel), np.float32)

    #Compute weights for inner area
    temp[1:height-1, 1:width-1] = image[1:height-1, 1:width-1]

    a1 = temp[1:height-1, 1:width-1] - image[0:height-2, 1:width-1]
    a2 = temp[1:height-1, 1:width-1] - image[2:height, 1:width-1]
    a3 = temp[1:height-1, 1:width-1] - image[1:height-1, 0:width-2]
    a4 = temp[1:height-1, 1:width-1] - image[1:height-1, 2:width]
    a = np.array([a1, a2, a3, a4])
    a_index = np.argmax(np.abs(a),0)
    b = a[a_index[1,1],1,1]
    for h in range(height-2):
        for w in range(width-2):
            for c in range(3):
                weights[h+1, w+1, c] = a[a_index[h,w,c],h,w,c]

    #Compute weights for corners and edges
    for h in range(height):
        for w in range(width):
            if h>0 and h<height-1 and w>0 and w<width-1:
                continue
            if h == 0 and w == 0:
                weights[h, w] = image[h, w]-image[h, w+1]
                for c in range(3):
                    if np.abs(image[h, w, c]-image[h+1, w, c]) > np.abs(image[h, w, c]-image[h, w+1, c]):
                        weights[h, w, c] = image[h, w, c]-image[h+1, w, c]
                continue
            if h == height-1 and w == 0:
                weights[h, w] = image[h, w]-image[h, w+1]
                for c in range(3):
                    if np.abs(image[h, w, c]-image[h-1, w, c]) > np.abs(image[h, w, c]-image[h, w+1, c]):
                        weights[h, w, c] = image[h, w, c]-image[h-1, w, c]
                continue
            if h == 0 and w == width-1:
                weights[h, w] = image[h, w]-image[h, w-1]
                for c in range(3):
                    if np.abs(image[h, w, c]-image[h+1, w, c]) > np.abs(image[h, w, c]-image[h, w-1, c]):
                        weights[h, w, c] = image[h, w, c]-image[h+1, w, c]
                continue
            if h == height-1 and w == width-1:
                weights[h, w] = image[h, w]-image[h, w-1]
                for c in range(3):
                    if np.abs(image[h, w, c]-image[h-1, w, c]) > np.abs(image[h, w, c]-image[h, w-1, c]):
                        weights[h, w, c] = image[h, w, c]-image[h-1, w, c]
                continue
            if h == 0:
                weights[h, w] = image[h, w]-image[h, w+1]
                for c in range(3):
                    if np.abs(image[h, w, c]-image[h, w-1, c]) > np.abs(image[h, w, c]-image[h, w+1, c]):
                        weights[h, w, c] = image[h, w, c]-image[h, w-1, c]
                    if np.abs(image[h, w, c]-image[h+1, w, c]) > np.abs(image[h, w, c]-image[h, w-1, c]):
                        weights[h, w, c] = image[h, w, c]-image[h+1, w, c]
                continue
            if h == height-1:
                weights[h, w] = image[h, w]-image[h, w+1]
                for c in range(3):
                    if np.abs(image[h, w, c]-image[h, w-1, c]) > np.abs(image[h, w, c]-image[h, w+1, c]):
                        weights[h, w, c] = image[h, w, c]-image[h, w-1, c]
                    if np.abs(image[h, w, c]-image[h-1, w, c]) > np.abs(image[h, w, c]-image[h, w-1, c]):
                        weights[h, w, c] = image[h, w, c]-image[h-1, w, c]
                continue
            if w == 0:
                weights[h, w] = image[h, w]-image[h, w+1]
                for c in range(3):
                    if np.abs(image[h, w, c]-image[h+1, w, c]) > np.abs(image[h, w, c]-image[h, w+1, c]):
                        weights[h, w, c] = image[h, w, c]-image[h+1, w, c]
                    if np.abs(image[h, w, c]-image[h-1, w, c]) > np.abs(image[h, w, c]-image[h+1, w, c]):
                        weights[h, w, c] = image[h, w, c]-image[h-1, w, c]
                continue
            if w == width-1:
                weights[h, w, c] = image[h, w, c]-image[h, w-1, c]
                for c in range(3):
                    if np.abs(image[h, w, c]-image[h+1, w, c]) > np.abs(image[h, w, c]-image[h, w-1, c]):
                        weights[h, w, c] = image[h, w, c]-image[h+1, w, c]
                    if np.abs(image[h, w, c]-image[h-1, w, c]) > np.abs(image[h, w, c]-image[h+1, w, c]):
                        weights[h, w, c] = image[h, w, c]-image[h-1, w, c]
                continue

    np.save('D:\\div2k\\weights_HR_g\\' + basename[0:-4] + '.npy', weights)
