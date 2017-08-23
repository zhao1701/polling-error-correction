#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 13:10:35 2017

@author: derekzhao
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

def binary_vs_target(firstFeatureName, lastFeatureName, targetName, featureKey, df):
    firstIndex = df.columns.get_loc(firstFeatureName)
    lastIndex = df.columns.get_loc(lastFeatureName)
    targetIndex = df.columns.get_loc(targetName)
    
    num_plots = lastIndex - firstIndex + 1
    fig, axes = plt.subplots(num_plots // 4 + 1, 4, figsize = (10, .7 * num_plots))
    
    p_list = []
    
    temp_df = df
    for axesIndex, index in zip(range(0, num_plots), range(firstIndex, lastIndex + 1)):
        n1 = sum(temp_df.iloc[:,index] == 0)
        n2 = sum(temp_df.iloc[:,index] == 1)
        x1 = sum((temp_df.iloc[:,index] == 0) & (temp_df.iloc[:,targetIndex] == 0))
        x2 = sum((temp_df.iloc[:,index] == 1) & (temp_df.iloc[:,targetIndex] == 0))
        p1 = x1 / n1
        p2 = x2 / n2
        g = sns.barplot([0,1],[p1,p2], ax = axes[axesIndex // 4][axesIndex % 4])
        if axesIndex % 4 == 0:
            g.set_ylabel('Proportion NV\nper response')
        
        z = np.abs((p1 - p2) / np.sqrt(p1 * (1-p1) / n1 + p2 * (1-p2) / n2))
        p_value = 2 * (1 - norm.cdf(z))
        feature_name = wrap_text(featureKey[df.columns[index]])
        title = feature_name + '\n' + '('+ df.columns[index] + ')' + '\np = ' + str(p_value)
        g.set_title(title)
        g.set_xticklabels(['No','Yes'])
        #g.set_xlabel(feature)
        
        p_list.append([abs(p1-p2), p_value, df.columns[index]])
    fig.tight_layout()
    return p_list