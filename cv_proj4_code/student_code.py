#!/usr/bin/python3

import numpy as np
from torch import nn
import torch
from typing import Tuple
import copy
import pdb
import time
import matplotlib.pyplot as plt
import math


"""
Authors: Vijay Upadhya, John Lambert, Cusuh Ham, Patsorn Sangkloy, Samarth
Brahmbhatt, Frank Dellaert, James Hays, January 2021.

Implement SIFT  (See Szeliski 7.1.2 or the original publications here:
    https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf

Your implementation will not exactly match the SIFT reference. For example,
we will be excluding scale and rotation invariance.

You do not need to perform the interpolation in which each gradient
measurement contributes to multiple orientation bins in multiple cells. 
"""

SOBEL_X_KERNEL = torch.tensor(
    [
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ],dtype=torch.float32)
SOBEL_Y_KERNEL = torch.tensor(
    [
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ],dtype=torch.float32)


#TODO 1
def compute_image_gradients(image_bw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Use convolution with Sobel filters to compute the image gradient at each pixel.

    Args:
        image_bw: A torch tensor of shape (M,N) containing the grayscale image

    Returns:
        Ix: Array of shape (M,N) representing partial derivatives of image w.r.t. x-direction
        Iy: Array of shape (M,N) representing partial derivative of image w.r.t. y-direction
    """

    # Create convolutional layer
    conv2d = nn.Conv2d(
        in_channels=1,
        out_channels=2,
        kernel_size=3,
        bias=False,
        padding=(1,1),
        padding_mode='zeros'
    )

    # Torch parameter representing (2, 1, 3, 3) conv filters

    # There should be two sets of filters: each should have size (1 x 3 x 3)
    # for 1 channel, 3 pixels in height, 3 pixels in width. When combined along
    # the batch dimension, this conv layer should have size (2 x 1 x 3 x 3), with
    # the Sobel_x filter first, and the Sobel_y filter second.
    
    #############################################################################
    # TODO: YOUR CODE HERE #                                                    #
    
    b_w_image = image_bw.unsqueeze(0).unsqueeze(0)
    
    
    k_x = SOBEL_X_KERNEL.unsqueeze(0).unsqueeze(0)
    k_y = SOBEL_Y_KERNEL.unsqueeze(0).unsqueeze(0)
    
    # convolution of b/w image with the corresponding sobel filter in x-direction
    x_direction = nn.functional.conv2d(b_w_image, k_x, padding = 1).squeeze(0).squeeze(0)
    
    # convolution of b/w image with the corresponding sobel filter in y-direction
    y_direction = nn.functional.conv2d(b_w_image, k_y, padding = 1).squeeze(0).squeeze(0)
    
    #############################################################################
    #raise NotImplementedError('`compute_image_gradients` function in ' +
    #    '`student_sift.py` needs to be implemented')
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    
    return x_direction, y_direction


#TODO 2.1
def get_gaussian_kernel_2D_pytorch(ksize: int, sigma: float) -> torch.Tensor:
    """Create a Pytorch Tensor representing a 2d Gaussian kernel
    Args:
        ksize: dimension of square kernel 
        sigma: standard deviation of Gaussian

    Returns:
        kernel: Tensor of shape (ksize,ksize) representing 2d Gaussian kernel
    """
    #############################################################################
    # TODO: YOUR CODE HERE                                                      #           
    #standard_deviation = sigma
    #ksize = 4*standard_deviation+1
    #variance = (standard_deviation)**2
    #k1 = ksize-1
    #mean = (k1*0.5)
    fil_size = ksize
    #x = torch.arange(int(ksize))
    average = fil_size//2
    #kernel = 1/((torch.sqrt(torch.tensor(2 * math.pi)) * standard_deviation))*torch.exp((-(x-mean)**2)/(2*variance))
    oneD_Gauss_kernel = np.arange(fil_size)
    #kernel = kernel/torch.sum(kernel)
    oneD_Gauss_kernel = (1/(sigma*np.sqrt(2*np.pi)))*(np.exp((-1/2)*np.power((oneD_Gauss_kernel - average)/(sigma), 2)))
    #kenny = kernel.numpy()
    oneD_Gauss_kernel = oneD_Gauss_kernel/np.sum(oneD_Gauss_kernel)
    #kernel_2d = np.outer(kenny.T, kenny.T)
    oneD_Gauss_kernel = np.reshape(oneD_Gauss_kernel, (-1, 1))
    kernel = torch.Tensor(np.outer(oneD_Gauss_kernel, oneD_Gauss_kernel))
    #kernel_2d = torch.from_numpy(np.asarray(kernel_2d))
    #kernel_2d = kernel_2d/torch.sum(kernel_2d)
    #return kernel_2d 
    return kernel
    #############################################################################
    #raise NotImplementedError('`get_gaussian_kernel_2D_pytorch` function in ' +
        #'`student_sift.py` needs to be implemented')
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

#TODO 2.2
def second_moments(
    image_bw: torch.tensor,
    ksize: int = 7,
    sigma: float = 10
) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
    """ Compute second moments from image.

    Compute image gradients Ix and Iy at each pixel, then mixed derivatives,
    then compute the second moments (sx2, sxsy, sy2) at each pixel, using
    convolution with a Gaussian filter.
    
    Args:
        image_bw: array of shape (M,N) containing the grayscale image
        ksize: size of 2d Gaussian filter
        sigma: standard deviation of gaussian filter
    
    Returns:
        sx2: array of shape (M,N) containing the second moment in the x direction
        sy2: array of shape (M,N) containing the second moment in the y direction
        sxsy: array of dim (M,N) containing the second moment in the x then the y direction
    """

    sx2, sy2, sxsy = None, None, None
    #############################################################################
    # TODO: YOUR CODE HERE                                                      #                                          #
    #############################################################################
    Ix,Iy = compute_image_gradients(image_bw)

    ker_filter = get_gaussian_kernel_2D_pytorch(ksize, sigma)
    
    ker_filter = torch.unsqueeze(ker_filter, 0)
    
    ker_filter = torch.cat([ker_filter, ker_filter, ker_filter], dim = 0)
    #print(ker_filter.shape)
    
    ker_filter = torch.unsqueeze(ker_filter, 1)
    #print(ker_filter.shape)
    
    Ix_sq = torch.mul(Ix,Ix).unsqueeze(0)
    #print(Ix_sq.shape)
    Iy_sq = torch.mul(Iy,Iy).unsqueeze(0)
    Ix_y = torch.mul(Ix,Iy).unsqueeze(0)
    
    input = torch.cat([Ix_sq, Iy_sq, Ix_y])
    # maintaining consistency in dimension
    input = torch.unsqueeze(input, 0)
    output = nn.functional.conv2d(input, ker_filter, groups = 3, padding = ker_filter.shape[-1]//2).squeeze()
    
    sx2 = output[0]
    sy2 = output[1]
    sxsy = output[2]
 
   
    
    #raise NotImplementedError('`second_moments` function in ' +
        #'`part1_harris_corner.py` needs to be implemented')
    
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return sx2, sy2, sxsy


#TODO 3
def compute_harris_response_map(
    image_bw: torch.tensor,
    ksize: int = 7,
    sigma: float = 5,
    alpha: float = 0.05
):
    """Compute the Harris cornerness score at each pixel (See Szeliski 7.1.1)

    Recall that R = det(M) - alpha * (trace(M))^2
    where M = [S_xx S_xy;
               S_xy  S_yy],
          S_xx = Gk * I_xx
          S_yy = Gk * I_yy
          S_xy  = Gk * I_xy,
    and * is a convolutional operation over a Gaussian kernel of size (k, k).
    (You can verify that this is equivalent to taking a (Gaussian) weighted sum
    over the window of size (k, k), see how convolutional operation works here:
    http://cs231n.github.io/convolutional-networks/)

    Ix, Iy are simply image derivatives in x and y directions, respectively.
    You may find the Pytorch function nn.Conv2d() helpful here.

    Args:
        image_bw: array of shape (M,N) containing the grayscale image
        ksize: size of 2d Gaussian filter
        sigma: standard deviation of gaussian filter
        alpha: scalar term in Harris response score
    
    Returns:
        R: array of shape (M,N), indicating the corner score of each pixel.
    """

    #############################################################################
    # TODO: YOUR CODE HERE                                                      #                                          #
    #############################################################################
    #raise NotImplementedError('`compute_harris_response_map` function in ' +
        #'`student_sift.py` needs to be implemented')
        
    sxx, syy, sxy = second_moments(image_bw, ksize, sigma)
    
    determinant_M = (sxx*syy - sxy*sxy)
    trace_M = (syy + sxx)
    R = determinant_M - alpha*(trace_M)**2
    #R = np.linalg.determinant_M - alpha*np.power(np.trace_M)
    #R = torch.from_numpy(R)

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return R

#TODO 4.1
def maxpool_numpy(R: torch.tensor, ksize: int) -> torch.tensor:
    """ Implement the 2d maxpool operator with (ksize,ksize) kernel size.

    Note: the implementation is identical to my_conv2d_numpy(), except we
    replace the dot product with a max() operator. You can implement with torch or numpy functions but do not use
    torch's exact maxpool 2d function here.
    
    Args:
        R: array of shape (M,N) representing a 2d score/response map

    Returns:
        maxpooled_R: array of shape (M,N) representing the maxpooled 2d score/response map
    """
    #############################################################################
    # TODO: YOUR CODE HERE                                                      #                                          #
    #############################################################################
    #raise NotImplementedError('`maxpool_numpy` function in ' +
        #'`student_sift.py` needs to be implemented')    
    M, N = R.shape #Assign M, N as the dimension of array shape
    pad = ksize//2
    pad_R = np.zeros((M+2*pad, N+2*pad))
    pad_R[pad:pad + M, pad:pad + N] = R

    maxpooled_R = np.zeros_like(R) # zero element matrix with same shape as R
    for index in range(pad, pad + M):
        for jindex in range(pad, pad + N):
            maxpooled_R[index-pad, jindex-pad] = np.amax(pad_R[index-pad:index+pad+1, jindex-pad:jindex+pad+1])
    maxpooled_R = torch.from_numpy(maxpooled_R)
    return maxpooled_R
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################    

#TODO 4.2
def nms_maxpool_pytorch(R: torch.tensor, k: int, ksize: int) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
    """ Get top k interest points that are local maxima over (ksize,ksize) neighborhood.
    
    HINT: One simple way to do non-maximum suppression is to simply pick a
    local maximum over some window size (u, v). This can be achieved using
    nn.MaxPool2d. Note that this would give us all local maxima even when they
    have a really low score compare to other local maxima. It might be useful
    to threshold out low value score before doing the pooling (torch.median
    might be useful here).

    You will definitely need to understand how nn.MaxPool2d works in order to
    utilize it, see https://pytorch.org/docs/stable/nn.html#maxpool2d

    Threshold globally everything below the median to zero, and then
    MaxPool over a 7x7 kernel. This will fill every entry in the subgrids
    with the maximum nearby value. Binarize the image according to
    locations that are equal to their maximum. Multiply this binary
    image, multiplied with the cornerness response values. We'll be testing
    only 1 image at a time.

    Args:
        R: score response map of shape (M,N)
        k: number of interest points (take top k by confidence)
        ksize: kernel size of max-pooling operator
    
    Returns:
        x: array of shape (k,) containing x-coordinates of interest points
        y: array of shape (k,) containing y-coordinates of interest points
        c: array of shape (k,) containing confidences of interest points
    """
    #############################################################################
    # TODO: YOUR CODE HERE                                                      #                                          #
    #############################################################################
    #raise NotImplementedError('`nms_maxpool_pytorch` function in ' +
        #'`student_sift.py` needs to be implemented')
        
    med_value = torch.median(R)
    filter_value = torch.where(R<med_value, torch.zeros_like(R), R)

    input = filter_value.unsqueeze(0).unsqueeze(0)
    mxpl = nn.MaxPool2d(kernel_size=ksize, stride=1, padding=ksize//2)

    mxpl_R = mxpl(input)
    mxpl_R = mxpl_R.squeeze(0).squeeze(0)
    
    mask = filter_value == mxpl_R

    filter_value = (filter_value*mask).numpy()

    location = np.argwhere(filter_value!=0)
    x = location[:, 1]
    y = location[:, 0]

    sorted_out = np.argsort(filter_value[y, x])[::-1][:k]

    x = x[sorted_out]
    y = y[sorted_out]
    confidences = filter_value[y, x]
    
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
    confidences = torch.from_numpy(confidences)
 
    return x, y, confidences 
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################    


#TODO 5.1    
def remove_border_vals(
    img: torch.tensor,
    x: torch.tensor,
    y: torch.tensor,
    c: torch.tensor
) -> Tuple[torch.tensor,torch.tensor,torch.tensor]:
    """
    Remove interest points that are too close to a border to allow SIFT feature
    extraction. Make sure you remove all points where a 16x16 window around
    that point cannot be formed.

    Args:
        img: array of shape (M,N) containing the grayscale image
        x: array of shape (k,)
        y: array of shape (k,)
        c: array of shape (k,)

    Returns:
        x: array of shape (p,), where p <= k (less than or equal after pruning)
        y: array of shape (p,)
        c: array of shape (p,)
    """

    #############################################################################
    # TODO: YOUR CODE HERE                                                      #                                          #
    #############################################################################

    #raise NotImplementedError('`remove_border_vals` in `HarrisNet.py` needs '
        #+ 'to be implemented')
        
    #M, N = img.shape
    #padding = 1

    #removed_idx = np.array(list(set((list(np.argwhere(x<padding-1).flatten())) + (list(np.argwhere(x>=(N-padding)).flatten())) + \
     #   (list(np.argwhere(y<padding-1).flatten())) + (list(np.argwhere(y>=(M-padding)).flatten())))))

    
    #k = x.shape[0]

    #remaining_idx = np.setdiff1d(np.arange(k), removed_idx)
    
    #x = x[remaining_idx]
    #y = y[remaining_idx]
    #c = c[remaining_idx]
    #------------------------------------------------------
    #w = img.shape[0]
    #h = img.shape[1]
    #------------------------------------------------------
    
    m,n = img.shape
    length_of_window = 16 
    minimum_width = length_of_window #
    maximum_width = m-length_of_window #
    minimum_height = length_of_window # 
    maximum_height = n-length_of_window #
    
    index = []
    for pointer in range(len(c)):
        if x[pointer]<minimum_width or y[pointer]<minimum_height or x[pointer]>maximum_width or y[pointer] > maximum_height:
            continue
        else:
            index.append(pointer)
            
    index = torch.LongTensor(index)
    #print(index.shape)
    x = torch.index_select(x,0, index)
    c = torch.index_select(c,0, index)
    y = torch.index_select(y,0, index)
    
    return x, y, c
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################     


#TODO 5.2
def get_harris_interest_points(image_bw: torch.tensor, k: int = 2500) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
    """
    Implement the Harris Corner detector. You will find
        compute_harris_response_map(), nms_maxpool_pytorch(), and remove_border_vals() useful.

    Args:
        image_bw: array of shape (M,N) containing the grayscale image
        k: number of interest points to retrieve

    Returns:
        x: array of shape (p,) containing x-coordinates of interest points
        y: array of shape (p,) containing y-coordinates of interest points
        confidences: array of dim (p,) containing the strength of each interest point
    """
    #############################################################################
    # TODO: YOUR CODE HERE                                                      #                                          #
    #############################################################################
    #raise NotImplementedError('`get_harris_interest_points` in `HarrisNet.py` needs '
        #+ 'to be implemented')
        
    filtersize = 7
    #R = compute_harris_response_map(image_bw)
    #x, y, c = nms_maxpool_pytorch(R, k, ksize)
    #array = c.numpy()
    #norm = np.linalg.norm(array)
    #c = array/norm
    #c = torch.from_numpy(c)
    
    #x, y, c = remove_border_vals(image_bw, x, y, c)
    #---------------------------------------------------------------------
    R = compute_harris_response_map(image_bw)
    x, y, c = nms_maxpool_pytorch(R, k, filtersize)
    #array = c.numpy()
    #norm = np.linalg.norm(array)
    #c = array/norm
    #c = torch.from_numpy(c)
    x, y, c = remove_border_vals(image_bw, x, y, c)
    
    
    return x, y, c
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################       

#TODO 6
def get_magnitudes_and_orientations(Ix: torch.tensor, Iy: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
    """
    This function will return the magnitudes and orientations of the
    gradients at each pixel location. 
    
    Args:
        Ix: array of shape (m,n), representing x gradients in the image
        Iy: array of shape (m,n), representing y gradients in the image
    Returns:
        magnitudes: A numpy array of shape (m,n), representing magnitudes of the
            gradients at each pixel location. Square root of (Ix ^ 2  + Iy ^ 2)
        orientations: A numpy array of shape (m,n), representing angles of
            the gradients at each pixel location. angles should range from 
            -PI to PI. (you may find torch.atan2 helpful here)
    """
    magnitudes = []#placeholder
    orientations = []#placeholder

    #############################################################################
    # TODO: YOUR CODE HERE                                                      #                                          #
    #############################################################################
    Ixx = torch.mul(Ix,Ix)
    Iyy = torch.mul(Iy,Iy)
    
    magnitudes = torch.sqrt(Ixx + Iyy) 
    orientations = torch.atan2(Iy,Ix)

    #raise NotImplementedError('`get_magnitudes_and_orientations` function in ' +
    #   '`part2_sift_descriptor.py` needs to be implemented')


    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return magnitudes, orientations


#TODO 7
def get_gradient_histogram_vec_from_patch(window_magnitudes: torch.tensor, window_orientations: torch.tensor) -> torch.tensor:
    """ Given 16x16 patch, form a 128-d vector of gradient histograms
    
    Key properties to implement:
    (1) a 4x4 grid of cells, each feature_width/4. It is simply the terminology
        used in the feature literature to describe the spatial bins where
        gradient distributions will be described. The grid will extend
        feature_width/2 - 1 to the left of the "center", and feature_width/2 to
        the right. The same applies to above and below, respectively.
    (2) each cell should have a histogram of the local distribution of
        gradients in 8 orientations. Appending these histograms together will
        give you 4x4 x 8 = 128 dimensions. The bin centers for the histogram 
        should be at -7pi/8,-5pi/8,...5pi/8,7pi/8. The histograms should be added
        to the feature vector left to right then row by row (reading order).  

    Do not normalize the histogram here to unit norm -- preserve the histogram
    values. You may find numpy's np.histogram() function to be useful here.

    Args:
        window_magnitudes: (16,16) tensor representing gradient magnitudes of the patch
        window_orientations: (16,16) tensor representing gradient orientations of the patch

    Returns:
        wgh: (128,1) representing weighted gradient histograms for all 16
            neighborhoods of size 4x4 px
    """
    #############################################################################
    # TODO: YOUR CODE HERE                                                      #                                          #
    #############################################################################

    #raise NotImplementedError('`get_gradient_histogram_vec_from_patch` function in ' +
    #    '`student_sift.py` needs to be implemented')
    
    # create an empty array to append with histogram[0]
    weight = np.empty((0,0))
    for index in range(4): 
        for jndex in range(4):
            magnitude_array = window_magnitudes[(4*index):(4*(index+1)),(4*jndex):(4*(jndex+1))]
            orientation_array = window_orientations[(4*index):(4*(index+1)), (4*jndex):(4*(jndex+1))]
            histo_gram = np.histogram(orientation_array, bins = 8, range = (-np.pi, np.pi), weights = magnitude_array)
            weight = np.append(weight, histo_gram[0])
    weight = weight.reshape(128, 1)

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return torch.from_numpy(weight)


#TODO 8
def get_feat_vec(
    x: float,
    y: float,
    magnitudes,
    orientations,
    feature_width: int = 16
) -> torch.tensor:
    """
    This function returns the feature vector for a specific interest point.
    To start with, you might want to simply use normalized patches as your
    local feature. This is very simple to code and works OK. However, to get
    full credit you will need to implement the more effective SIFT descriptor
    (See Szeliski 7.1.2 or the original publications at
    http://www.cs.ubc.ca/~lowe/keypoints/)
    Your implementation does not need to exactly match the SIFT reference.


    Your (baseline) descriptor should have:
    (1) Each feature should be normalized to unit length.
    (2) Each feature should be raised to the 1/2 power, i.e. square-root SIFT
        (read https://www.robots.ox.ac.uk/~vgg/publications/2012/Arandjelovic12/arandjelovic12.pdf)
    
    For our tests, you do not need to perform the interpolation in which each gradient
    measurement contributes to multiple orientation bins in multiple cells
    As described in Szeliski, a single gradient measurement creates a
    weighted contribution to the 4 nearest cells and the 2 nearest
    orientation bins within each cell, for 8 total contributions.
    The autograder will only check for each gradient contributing to a single bin.
    
    Args:
        x: a float, the x-coordinate of the interest point
        y: A float, the y-coordinate of the interest point
        magnitudes: A torch tensor of shape (m,n), representing image gradients
            at each pixel location
        orientations: A torch tensor of shape (m,n), representing gradient
            orientations at each pixel location
        feature_width: integer representing the local feature width in pixels.
            You can assume that feature_width will be a multiple of 4 (i.e. every
                cell of your local SIFT-like feature will have an integer width
                and height). This is the initial window size we examine around
                each keypoint.
    Returns:
        fv: A torch tensor of shape (feat_dim,1) representing a feature vector.
            "feat_dim" is the feature_dimensionality (e.g. 128 for standard SIFT).
            These are the computed features.
    """

    fv = []#placeholder
    #############################################################################
    # TODO: YOUR CODE HERE                                                      #                                          #
    #############################################################################
    #raise NotImplementedError('`get_feat_vec` function in ' +
    #    '`student_sift.py` needs to be implemented')
    
    mag_gradient = magnitudes[y-(feature_width-1)//2:y+feature_width//2+1, 
        x-(feature_width-1)//2:x+feature_width//2+1]
    # similarly we implement for the orientation
    ori_gradient = orientations[y-(feature_width-1)//2:y+feature_width//2+1, 
        x-(feature_width-1)//2:x+feature_width//2+1]
   
    
    fv = get_gradient_histogram_vec_from_patch(mag_gradient, ori_gradient)
    # numpy linear algebra . normalize(arg, order)
    
    var = (int)(feature_width/4)
    #print(var)
    normalize = np.linalg.norm(fv.reshape(1, var**2*8))
    
    if (normalize != 0):
        fv = fv/normalize
    fv = np.asarray(fv)
    fv = np.power(fv, 1/2)
    fv = torch.from_numpy(fv)

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return fv


#TODO 9
def get_SIFT_descriptors(
    image_bw: torch.tensor,
    X: torch.tensor,
    Y: torch.tensor,
    feature_width: int = 16
) -> torch.tensor:
    """
    This function returns the 128-d SIFT features computed at each of the input points
    Implement the more effective SIFT descriptor
    (See Szeliski 4.1.2 or the original publications at
    http://www.cs.ubc.ca/~lowe/keypoints/)

    Args:
        image: A torch tensor of shape (m,n), the image
        X: A torch tensor of shape (k,), the x-coordinates of interest points
        Y: A torch tensor of shape (k,), the y-coordinates of interest points
        feature_width: integer representing the local feature width in pixels.
            You can assume that feature_width will be a multiple of 4 (i.e. every
                cell of your local SIFT-like feature will have an integer width
                and height). This is the initial window size we examine around
                each keypoint.
    Returns:
        fvs: A torch tensor of shape (k, feat_dim) representing all feature vectors.
            "feat_dim" is the feature_dimensionality (e.g. 128 for standard SIFT).
            These are the computed features.
    """
    assert image_bw.ndim == 2, 'Image must be grayscale'
    #############################################################################
    # TODO: YOUR CODE HERE                                                      #                                          #
    #############################################################################

    #raise NotImplementedError('`get_features` function in ' +
    #    '`student_sift.py` needs to be implemented')
    
    fvs = np.empty((0,0))
    X_grad, Y_grad = compute_image_gradients(image_bw)
    
    # obtain magnitude and orientation of each gradient pixel
    mag, orient = get_magnitudes_and_orientations(X_grad, Y_grad)
    
    for index in range(len(X)): 
        fv = get_feat_vec(X[index], Y[index], mag, orient, feature_width)
        # print(fv)
        fvs = np.append(fvs, fv)
    fvs = fvs.reshape(len(X), feature_width*8)
    fvs = torch.from_numpy(fvs)
    
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return fvs

#TODO 10
def compute_feature_distances(
    features1: torch.tensor,
    features2: torch.tensor
) -> torch.tensor:
    """
    This function computes a list of distances from every feature in one array
    to every feature in another.

    Using Numpy broadcasting is required to keep memory requirements low.

    Note: Using a double for-loop is going to be too slow.
    One for-loop is the maximum possible. Vectorization is needed.
    See numpy broadcasting details here:
        https://cs231n.github.io/python-numpy-tutorial/#broadcasting

    Args:
        features1: A numpy array of shape (n1,feat_dim) representing one set of
            features, where feat_dim denotes the feature dimensionality
        features2: A numpy array of shape (n2,feat_dim) representing a second set
            features (n1 not necessarily equal to n2)

    Returns:
        dists: A numpy array of shape (n1,n2) which holds the distances 
            (in feature space) from each feature in features1 to each feature 
            in features2
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    #raise NotImplementedError('`match_features` function in ' +
    #    '`student_feature_matching.py` needs to be implemented')
    #feature_1 = features1.shape[0]
    #feature_2 = features2.shape[0]

    #distance = np.zeros((feature_1,feature_2), dtype = float)
    #for index in range(feature_1):
        #for jndex in range(feature_2):
            ## print(np.linalg.norm(features1[index,:] - features2[jndex,:]))
            #distance[index, jndex] = np.linalg.norm(features1[index,:] - features2[jndex,:])
    #distance = torch.from_numpy(distance)
    
    #distance=np.zeros((features1.shape[0],features2.shape[0]))
    #for index in range(len(features1)):
        #num1 = features1[index]
        #for jndex in range(len(features2)):
            #num2 = features2[jndex]
            #dist = np.linalg.norm(num1-num2)
            #dists[index][jndex] = distance

    #distance = torch.from_numpy(dists)
    x_1 = features1.shape[0]
    x_2 = features2.shape[0]
    distance = torch.zeros((x_1,x_2))
    for index in range(0,x_1):
        distance[index, :] = torch.norm(features2 - features1[index, :], dim = -1)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return distance


#TODO 11
def match_features_ratio_test(
    features1: torch.tensor,
    features2: torch.tensor
) -> Tuple[torch.tensor, torch.tensor]:
    """ Nearest-neighbor distance ratio feature matching.

    This function does not need to be symmetric (e.g. it can produce
    different numbers of matches depending on the order of the arguments).

    To start with, simply implement the "ratio test", equation 7.18 in
    section 7.1.3 of Szeliski. There are a lot of repetitive features in
    these images, and all of their descriptors will look similar. The
    ratio test helps us resolve this issue (also see Figure 11 of David
    Lowe's IJCV paper).

    You should call `compute_feature_distances()` in this function, and then
    process the output.

    Args:
        features1: A torch tensor of shape (n1,feat_dim) representing one set of
            features, where feat_dim denotes the feature dimensionality
        features2: A torch tensor of shape (n2,feat_dim) representing a second
            set of features (n1 not necessarily equal to n2)

    Returns:
        matches: A torch tensor of shape (k,2), where k is the number of matches.
            The first column is an index in features1, and the second column is an
            index in features2
        confidences: A torch tensor of shape (k,) with the real valued confidence
            for every match

    'matches' and 'confidences' can be empty e.g. (0x2) and (0x1)
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    #raise NotImplementedError('`match_features` function in ' +
    #    '`student_feature_matching.py` needs to be implemented')
   
    distance = compute_feature_distances(features1, features2)
    
    arranged_distance_index = np.argsort(distance, axis=1)
    #print(arranged_distance_index.shape)
    arranged_distance = np.sort(distance, axis=1)
    fraction = arranged_distance[:, 0]/arranged_distance[:, 1]
    
    threshold = 0.81
    masking = fraction < threshold

    matches = np.concatenate((np.arange(arranged_distance.shape[0]).reshape((-1,1)),arranged_distance_index[:, 0].reshape((-1,1))), axis=1)
    matches = torch.from_numpy(matches)
    fraction = torch.from_numpy(fraction)
    matches = matches[masking]
    confidences = fraction[masking]
    #------------------------------------------------------------------------

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return matches, confidences
