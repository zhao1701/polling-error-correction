#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 20:56:50 2017

@author: derekzhao
"""
import numpy as np

def thermometer_to_intensity(x):
    if x == np.nan:
        return np.nan
    else:
        return abs(x - 50) ** 2