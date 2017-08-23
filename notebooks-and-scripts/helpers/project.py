#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 13:10:35 2017

@author: derekzhao
"""

import csv
import numpy as np
import pandas as pd

with open('helpers/feature_key.csv') as csvfile:
    reader = csv.reader(csvfile)
    feature_key = dict()
    for row in reader:
        feature_key[row[0]] = row[1]
        
def wrap_text(in_string, num = 15):
    out_string = str()
    counter = 0
    for char in in_string:
        if char == ' ' and counter > num:
            out_string = out_string + '\n'
            counter = 0
        else:
            out_string = out_string + char
            counter = counter + 1
    return out_string

def add_thermometer_intensity(df):
    
    def thermometer_to_intensity(x):
        if x == np.nan:
            return np.nan
        else:
            return abs(x - 50) ** 2
    
    columns_to_convert = (df.max() > 50) & (df.min() <= 0)
    columns_to_convert[[df.columns.get_loc('VCF0114_r1'),df.columns.get_loc('VCF1015')]] = False
    thermometer_df = df.loc[:,columns_to_convert]
    thermometer_df = thermometer_df.applymap(thermometer_to_intensity)
    thermometer_df.columns = thermometer_df.columns + '_int'
    thermometer_df['int_sum_therm'] = thermometer_df.sum(axis = 1) ** 1.2
    return pd.concat([df, thermometer_df], axis = 1)

def add_ordinal_intensity(df):
    columns_to_convert = ['VCF0803','VCF0806','VCF0830','VCF0851','VCF9014','VCF9015','VCF9039','VCF9042','VCF0301',
                     'VCF0303','VCF0502','VCF0604','VCF0605','VCF0880a','VCF9009','VCF9045']
    intensity_df = df.loc[:,columns_to_convert]
    intensity_df = abs(intensity_df - (intensity_df.max() + intensity_df.min()) / 2)
    intensity_df.columns = intensity_df.columns + '_int'
    intensity_df['int_sum_ord'] = intensity_df.sum(axis = 1) ** 2
    return pd.concat([df, intensity_df], axis = 1)

def add_dk_sum(df):
    df = df[:]
    dk_column_index = df.columns.str.contains('dk')
    dk_df = df.loc[:,dk_column_index]
    df['dk_sum'] = dk_df.sum(axis = 1) ** 2
    return df