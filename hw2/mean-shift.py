import time
import os
import random
import math
import torch
import numpy as np

# run `pip install scikit-image` to install skimage, if you haven't done so
from skimage import io, color
from skimage.transform import rescale

def distance(x, X):
    return torch.sqrt(torch.sum((x - X) ** 2, dim=-1))

def distance_batch(x, X):
    return torch.sqrt(torch.sum(torch.pow(X.reshape(-1, 1, 3) - X.reshape(1, -1, 3), 2), dim=-1))

def gaussian(dist, bandwidth):
    return torch.exp(- 0.5 * (dist / bandwidth) ** 2)

def update_point(weight, X):
    normalizer = torch.sum(weight)
    return torch.sum(weight[:, None] * X, dim=0) / normalizer

def update_point_batch(weight, X):
    # shape (N,)
    normalizer = torch.sum(weight, dim=-1, keepdim=True).expand(-1, 3)
    # shape (N, C)
    new_X = torch.matmul(weight, X)
    # shape (N, C)
    return new_X / normalizer

def meanshift_step(X, bandwidth=2.5):
    X_ = X.clone()
    for i, x in enumerate(X):
        dist = distance(x, X)
        weight = gaussian(dist, bandwidth)
        X_[i] = update_point(weight, X)
    return X_

def meanshift_step_batch(X, bandwidth=2.5):
    dists = distance_batch(None, X)
    weights = gaussian(dists, bandwidth)
    X = update_point_batch(weights, X)
    return X

def meanshift(X):
    X = X.clone()
    for itr in range(10):
        # X = meanshift_step(X)   # slow implementation
        X = meanshift_step_batch(X)   # fast implementation
        print('finished iteration:', itr)
    return X

scale = 0.25    # downscale the image to run faster

# Load image and convert it to CIELAB space
image = rescale(io.imread('cow.jpg'), scale, multichannel=True)
image_lab = color.rgb2lab(image)
shape = image_lab.shape # record image shape
image_lab = image_lab.reshape([-1, 3])  # flatten the image

# Run your mean-shift algorithm
t = time.time()
X = meanshift(torch.from_numpy(image_lab)).detach().cpu().numpy()
# X = meanshift(torch.from_numpy(data).cuda()).detach().cpu().numpy()  # you can use GPU if you have one
t = time.time() - t
print ('Elapsed time for mean-shift: {}'.format(t))

# Load label colors and draw labels as an image
colors = np.load('colors.npz')['colors']
colors[colors > 1.0] = 1
colors[colors < 0.0] = 0

centroids, labels = np.unique((X / 4).round(), return_inverse=True, axis=0)

result_image = colors[labels].reshape(shape)
result_image = rescale(result_image, 1 / scale, order=0, multichannel=True)     # resize result image to original resolution
result_image = (result_image * 255).astype(np.uint8)
io.imsave('result.png', result_image)
