import numpy as np
from skimage.io import imread, imsave
from skimage.transform import rescale, resize, downscale_local_mean

img_y = open('../data/manual_03.ntxy', 'r')
img_y = list(img_y)[:128]

for line in img_y:
    _, img, yx, yy = line.split(' ')
    image = imread('../data/frames/03'+img+'.jpg')
    image_rescaled = rescale(image, 1.0 / 5.0)
    imsave('../data/resized/03'+img+'.jpg', image_rescaled);
