import random
import numpy as np
import cv2

def image_standardization(image):

    num_pixels = image.size
    image = image.astype(np.float64)
    image_mean = np.mean(image)

    stddev = np.std(image)
    # Apply a minimum normalization that protects us against uniform images.
    min_stddev = 1.0/np.sqrt(num_pixels)
    pixel_value_scale = np.max([stddev, min_stddev])
    pixel_value_offset = image_mean

    image = image - pixel_value_offset
    image = image/pixel_value_scale
    return image

def get_image(path):
    image = cv2.imread(path)
    h, w = image.shape[:2]
    pad = int(h/8)
    pad2 = int(pad/2)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_pad = np.zeros((h+pad, w+pad, 3), np.uint8)

    image_pad[pad2:pad2+h, pad2:pad2+w] = image

    w_shift = random.randint(0, pad)
    h_shift = random.randint(0, pad)
    f = random.randint(0,1)

    image = image_pad[h_shift:h_shift+h, w_shift:w_shift+w]
    if f == 1:
        image = cv2.flip(image, 1)
    #cv2.imshow('image', image)
    #cv2.imshow('image_pad', image_pad)
    #cv2.waitKey()
    image = image_standardization(image)
    return image
    
def make_shuffle_idx(n):
    #random.seed(0)
    order = list(range(n))
    random.shuffle(order)
    return order
    
def compute_psnr(image1,image2, upscale):
    upscale = upscale + 6
    image1 = image1[upscale:-upscale, upscale:-upscale]
    image2 = image2[upscale:-upscale, upscale:-upscale]
    
    diff = image1.astype(np.float) - image2.astype(np.float)
    diff = diff*diff
    diff = np.mean(diff)
    rmse = np.sqrt(diff)
    psnr = 20*np.log10(255/rmse)
    return psnr