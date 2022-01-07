#!/usr/bin/python3

import copy
import pdb
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import sys
import os

sys.path.append(os.getcwd())

from student_code import (
    get_harris_interest_points,
    match_features_ratio_test,
    get_magnitudes_and_orientations,
    get_gradient_histogram_vec_from_patch,
    get_SIFT_descriptors,
    get_feat_vec,
)
from utils import load_image, evaluate_correspondence, rgb2gray, PIL_resize

ROOT = Path(__file__).resolve().parent.parent  # ../..

def test_get_magnitudes_and_orientations():
    """ Verify gradient magnitudes and orientations are computed correctly"""
    Ix = torch.from_numpy(np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])).float()
    Iy = torch.from_numpy(np.array([[0, 0, 0], [1, 1, 1], [1, 1, 1]])).float()
    magnitudes, orientations = get_magnitudes_and_orientations(Ix, Iy)
    magnitudes = magnitudes.numpy()
    orientations = orientations.numpy()
    # there are 3 vectors -- (1,0) at 0 deg, (0,1) at 90 deg, and (-1,1) and 135 deg
    expected_magnitudes = np.array([[1, 1, 1], [1, 1, 1], [np.sqrt(2), np.sqrt(2), np.sqrt(2)]])
    expected_orientations = np.array(
        [[0, 0, 0], [np.pi / 2, np.pi / 2, np.pi / 2], [3 * np.pi / 4, 3 * np.pi / 4, 3 * np.pi / 4]]
    )

    assert np.allclose(magnitudes, expected_magnitudes)
    assert np.allclose(orientations, expected_orientations)

def test_get_gradient_histogram_vec_from_patch():
    """ Check if weighted gradient histogram is computed correctly """
    window_magnitudes = np.array(
        [
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ]
    )

    A = 1/8 * np.pi # squarely in bin [0, pi/4]
    B = 3/8 * np.pi # squarely in bin [pi/4, pi/2]

    window_orientations = np.array(
        [
            [ A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A],
            [ A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A],
            [ A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A],
            [ A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A],
            [ B, B, B, B, B, B, B, B, B, B, B, B, B, B, B, B],
            [ B, B, B, B, B, B, B, B, B, B, B, B, B, B, B, B],
            [ B, B, B, B, B, B, B, B, B, B, B, B, B, B, B, B],
            [ B, B, B, B, B, B, B, B, B, B, B, B, B, B, B, B],
            [ A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A],
            [ A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A],
            [ A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A],
            [ A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A],
            [ B, B, B, B, B, B, B, B, B, B, B, B, B, B, B, B],
            [ B, B, B, B, B, B, B, B, B, B, B, B, B, B, B, B],
            [ B, B, B, B, B, B, B, B, B, B, B, B, B, B, B, B],
            [ B, B, B, B, B, B, B, B, B, B, B, B, B, B, B, B]
        ]
    )

    wgh = get_gradient_histogram_vec_from_patch(torch.from_numpy(window_magnitudes), torch.from_numpy(window_orientations))

    expected_wgh = np.array(
        [
            [ 0.,  0.,  0.,  0., 16.,  0.,  0.,  0.], # bin 4, magnitude 1
            [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.], # bin 4, magnitude 0
            [ 0.,  0.,  0.,  0., 16.,  0.,  0.,  0.], # bin 4
            [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.], # bin 4
            [ 0.,  0.,  0.,  0.,  0., 32.,  0.,  0.], 
            [ 0.,  0.,  0.,  0.,  0., 16.,  0.,  0.], # bin 5
            [ 0.,  0.,  0.,  0.,  0., 32.,  0.,  0.], # bin 5, magnitude 2
            [ 0.,  0.,  0.,  0.,  0., 16.,  0.,  0.], # bin 5
            [ 0.,  0.,  0.,  0., 16.,  0.,  0.,  0.],
            [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
            [ 0.,  0.,  0.,  0., 16.,  0.,  0.,  0.],
            [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
            [ 0.,  0.,  0.,  0.,  0., 16.,  0.,  0.],
            [ 0.,  0.,  0.,  0.,  0., 16.,  0.,  0.],
            [ 0.,  0.,  0.,  0.,  0., 16.,  0.,  0.],
            [ 0.,  0.,  0.,  0.,  0., 16.,  0.,  0.]
        ]
    ).reshape(128, 1)

    assert np.allclose(wgh.numpy(), expected_wgh, atol=1e-1)

def test_get_feat_vec():
    """ Check if feature vector for a specific interest point is returned correctly """
    window_magnitudes = np.array(
        [
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ]
    )

    A = 1/8 * np.pi # squarely in bin [0, pi/4]
    B = 3/8 * np.pi # squarely in bin [pi/4, pi/2]
    C = 5/8 * np.pi # squarely in bin [pi/2, 3pi/4]

    window_orientations = np.array(
        [
            [ B, B, B, B, B, B, B, B, B, B, B, B, B, B, B, B ],
            [ B, B, B, B, B, B, B, B, B, B, B, B, B, B, B, B ],
            [ B, B, B, B, B, B, B, B, B, B, B, B, B, B, B, B ],
            [ B, B, B, B, B, B, B, B, B, B, B, B, B, B, B, B ],
            [ C, C, C, C, C, C, C, C, C, C, C, C, C, C, C, C ],
            [ C, C, C, C, C, C, C, C, C, C, C, C, C, C, C, C ],
            [ C, C, C, C, C, C, C, C, C, C, C, C, C, C, C, C ],
            [ C, C, C, C, C, C, C, C, C, C, C, C, C, C, C, C ],
            [ B, B, B, B, B, B, B, B, B, B, B, B, B, B, B, B ],
            [ B, B, B, B, B, B, B, B, B, B, B, B, B, B, B, B ],
            [ B, B, B, B, B, B, B, B, B, B, B, B, B, B, B, B ],
            [ B, B, B, B, B, B, B, B, B, B, B, B, B, B, B, B ],
            [ C, C, C, C, C, C, C, C, C, C, C, C, C, C, C, C ],
            [ C, C, C, C, C, C, C, C, C, C, C, C, C, C, C, C ],
            [ C, C, C, C, C, C, C, C, C, C, C, C, C, C, C, C ],
            [ C, C, C, C, C, C, C, C, C, C, C, C, C, C, C, C ]
        ]
    )

    feature_width = 16

    x, y = 7, 7

    fv = get_feat_vec(x, y, torch.from_numpy(window_magnitudes), torch.from_numpy(window_orientations), feature_width)

    expected_fv = np.array(
        [
            [0. , 0. , 0. , 0. , 0. , 0.485, 0.   , 0. ],
            [0. , 0. , 0. , 0. , 0. , 0.   , 0.   , 0. ],
            [0. , 0. , 0. , 0. , 0. , 0.485, 0.   , 0. ],
            [0. , 0. , 0. , 0. , 0. , 0.   , 0.   , 0. ],
            [0. , 0. , 0. , 0. , 0. , 0.   , 0.687, 0. ],
            [0. , 0. , 0. , 0. , 0. , 0.   , 0.485, 0. ],
            [0. , 0. , 0. , 0. , 0. , 0.   , 0.687, 0. ],
            [0. , 0. , 0. , 0. , 0. , 0.   , 0.485, 0. ],
            [0. , 0. , 0. , 0. , 0. , 0.485, 0.   , 0. ],
            [0. , 0. , 0. , 0. , 0. , 0.   , 0.   , 0. ],
            [0. , 0. , 0. , 0. , 0. , 0.485, 0.   , 0. ],
            [0. , 0. , 0. , 0. , 0. , 0.   , 0.   , 0. ],
            [0. , 0. , 0. , 0. , 0. , 0.   , 0.485, 0. ],
            [0. , 0. , 0. , 0. , 0. , 0.   , 0.485, 0. ],
            [0. , 0. , 0. , 0. , 0. , 0.   , 0.485, 0. ],
            [0. , 0. , 0. , 0. , 0. , 0.   , 0.485, 0. ]
        ]
    ).reshape(128, 1)

    assert np.allclose(fv.numpy(), expected_fv, atol=1e-2)

def test_get_SIFT_descriptors():
    """ Check if the 128-d SIFT feature vector computed at each of the input points is returned correctly """

    image1 = torch.from_numpy(np.array(
        [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
            [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
            [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
            [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
            [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
            [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
            [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25],
            [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26],
            [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27],
            [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28],
            [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
            [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
            [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
            [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32],
            [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33],
            [17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34],
        ]
    ).astype(np.float32))

    X1, Y1 = np.array([8, 9]).astype(np.int32), np.array([8, 9]).astype(np.int32)

    SIFT_descriptors = get_SIFT_descriptors(image1, torch.from_numpy(X1), torch.from_numpy(Y1))

    expected_SIFT_descriptors_1 = np.array(
        [
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.379, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.379, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.379, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.329, 0.0, 0.499],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.379, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.379, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.379, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.329, 0.0, 0.547],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.379, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.379, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.379, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.329, 0.0, 0.592],
                [0.0, 0.0, 0.499, 0.0, 0.0, 0.329, 0.0, 0.0],
                [0.0, 0.0, 0.547, 0.0, 0.0, 0.329, 0.0, 0.0],
                [0.0, 0.0, 0.592, 0.0, 0.0, 0.329, 0.0, 0.0],
                [0.0, 0.332, 0.544, 0.0, 0.0, 0.285, 0.0, 0.544],
            ],
        ]
    ).reshape(2, 128)
    expected_SIFT_descriptors_2 = np.array(
        [
            [
                [0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0, 0.0, 0.379, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.379, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.379, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.329, 0.0, 0.0, 0.499],
                [0.0, 0.0, 0.0, 0.0, 0.379, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.379, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.379, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.329, 0.0, 0.0, 0.547],
                [0.0, 0.0, 0.0, 0.0, 0.379, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.379, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.379, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.329, 0.0, 0.0, 0.592],
                [0.0, 0.0, 0.499, 0.0, 0.329, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.547, 0.0, 0.329, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.592, 0.0, 0.329, 0.0, 0.0, 0.0],
                [0.0, 0.332, 0.544, 0.0, 0.285, 0.0, 0.0, 0.544],
            ],
        ]
    ).reshape(2, 128)
    assert np.allclose(SIFT_descriptors.numpy(), expected_SIFT_descriptors_1, atol=1e-1) or np.allclose(SIFT_descriptors.numpy(), expected_SIFT_descriptors_2, atol=1e-1)

def test_feature_matching_speed():
    """
    Test how long feature matching takes to execute on the Notre Dame pair.
    This unit test must run in under 90 seconds.
    """
    start = time.time()
    image1 = load_image("../data/1a_notredame.jpg")
    image2 = load_image("../data/1b_notredame.jpg")
    eval_file = "../ground_truth/notredame.pkl"
    scale_factor = 0.5
    image1 = PIL_resize(image1, (int(image1.shape[1] * scale_factor), int(image1.shape[0] * scale_factor)))
    image2 = PIL_resize(image2, (int(image2.shape[1] * scale_factor), int(image2.shape[0] * scale_factor)))
    image1_bw = rgb2gray(image1)
    image2_bw = rgb2gray(image2)

    X1, Y1, _ = get_harris_interest_points(torch.from_numpy(copy.deepcopy(image1_bw)))
    X2, Y2, _ = get_harris_interest_points(torch.from_numpy(copy.deepcopy(image2_bw)))

    image1_features = get_SIFT_descriptors(torch.from_numpy(image1_bw), X1, Y1)
    image2_features = get_SIFT_descriptors(torch.from_numpy(image2_bw), X2, Y2)

    matches, confidences = match_features_ratio_test(image1_features, image2_features)
    print("{:d} matches from {:d} corners".format(len(matches), len(X1)))

    end = time.time()
    duration = end - start
    print(f"Your Feature matching pipeline takes {duration:.2f} seconds to run on Notre Dame")

    MAX_ALLOWED_TIME = 90  # sec
    assert duration < MAX_ALLOWED_TIME

def test_feature_matching_accuracy():
    """
    Test how long feature matching takes to execute on the Notre Dame pair.
    This unit test must achieve at least 80% accuracy.
    """
    image1 = load_image("../data/1a_notredame.jpg")
    image2 = load_image("../data/1b_notredame.jpg")
    eval_file = "../ground_truth/notredame.pkl"
    scale_factor = 0.5
    image1 = PIL_resize(image1, (int(image1.shape[1] * scale_factor), int(image1.shape[0] * scale_factor)))
    image2 = PIL_resize(image2, (int(image2.shape[1] * scale_factor), int(image2.shape[0] * scale_factor)))
    image1_bw = rgb2gray(image1)
    image2_bw = rgb2gray(image2)

    X1, Y1, _ = get_harris_interest_points(torch.from_numpy(copy.deepcopy(image1_bw)))
    X2, Y2, _ = get_harris_interest_points(torch.from_numpy(copy.deepcopy(image2_bw)))

    image1_features = get_SIFT_descriptors(torch.from_numpy(image1_bw), X1, Y1)
    image2_features = get_SIFT_descriptors(torch.from_numpy(image2_bw), X2, Y2)

    matches, confidences = match_features_ratio_test(image1_features, image2_features)

    X1 = X1.numpy()
    Y1 = Y1.numpy()
    X2 = X2.numpy()
    Y2 = Y2.numpy()

    acc, _ = evaluate_correspondence(
        image1,
        image2,
        eval_file,
        scale_factor,
        X1[matches[:, 0]],
        Y1[matches[:, 0]],
        X2[matches[:, 1]],
        Y2[matches[:, 1]],
    )

    print(f"Your Feature matching pipeline achieved {100 * acc:.2f}% accuracy to run on Notre Dame")

    MIN_ALLOWED_ACC = 0.80  # 80 percent
    assert acc > MIN_ALLOWED_ACC