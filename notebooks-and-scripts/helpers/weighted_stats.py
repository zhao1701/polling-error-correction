#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 13:10:35 2017

@author: derekzhao
"""

import numpy as np

def weighted_mean(x, w):
    if len(x) != len(w):
        raise AssertionError('Dimensions do not match.')
    return np.dot(x,w) / sum(w)

def weighted_variance(x, w):
    if len(x) != len(w):
        raise AssertionError('Dimensions do not match.')
    mean = weighted_mean(x,w)
    return np.dot(w, (x - mean)**2) / sum(w)

def weighted_covariance(x, y, w):
    if len(x) != len(w) or len(x) != len(y) or len(y) != len(w):
        raise AssertionError('Dimensions do not match.')
    x_mean = weighted_mean(x,w)
    y_mean = weighted_mean(y,w)
    return sum(w * (x - x_mean) * (y - y_mean)) / sum(w)

def weighted_correlation(x, y, w):
    if len(x) != len(w) or len(x) != len(y) or len(y) != len(w):
        raise AssertionError('Dimensions do not match.')
    return weighted_covariance(x, y, w) / np.sqrt(weighted_variance(x, w) * weighted_variance(y,w))
    
def weighted_correlation_matrix(data, w):
    data = np.array(data)
    num_cols = data.shape[1]
    corr = np.zeros([num_cols, num_cols])
    for i in range(0, num_cols):
        for j in range(0, num_cols):
            x = data[:,i]
            y = data[:,j]
            corr[i][j] = weighted_correlation(x, y, w)
            
    return corr
