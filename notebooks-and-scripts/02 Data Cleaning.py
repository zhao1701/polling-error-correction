#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 14:21:33 2017

INPUT
-   anes_cdf_abridged.csv: The result of filtering responents and features from the 
    full ANES Cumulative Time Series Survey dataset.

OUTPUT:
-   anes_cdf_converted.csv: A CSV with all features encoded as numerical values. Additionally,
    all feature are grouped by type (numerical, binary, ordinal, and categorical) for ease of
    mass data analysis later.

This module will build the output CSV by appending onto a new DataFrame rather than converting
one in place.

@author: derekzhao
"""

import pandas as pd
import numpy as np
import re
import helpers.cleaning as cln

# Turn off verification plots for faster performance
cln.print_verification = True

# Read file. Due to unknown bug, '01 Data Filtering.py' may fail to remove column 'Unnamed: 0'.
clean_df = pd.read_csv('../data/anes_cdf_abridged.csv')
if clean_df.columns[0] == 'Unnamed: 0':
    clean_df = clean_df.drop('Unnamed: 0', axis = 1)
    
# Initialize new DataFrame for processed feature accumulation.
final_df = pd.DataFrame()

##################
### BASIC INFO ###
##################

# Add basic information with new features names
final_df['year'] = clean_df.VCF0004
final_df['age'] = clean_df.VCF0101
final_df.age = final_df.age.replace(0, final_df.age.mean()) # Replace ages of 0 with mean age
final_df['congressional_district'] = clean_df.VCF0900c
final_df['state'] = clean_df.VCF0901b      
final_df['gender'] = clean_df.VCF0104.apply(lambda x: int(x[0])-1)
final_df['weight'] = clean_df.VCF0011z
final_df['final_vote'] = clean_df.VCF0704.apply(lambda x: int(x[0]))

#######################
### BINARY FEATURES ###
#######################

cln.convert_binary(final_df, clean_df, 'VCF0108', [2, 8, 9], [0, np.nan, np.nan])
cln.convert_binary(final_df, clean_df, 'VCF0113', [2],[0])
cln.convert_binary(final_df, clean_df, 'VCF0127', [2],[0])
cln.convert_binary(final_df, clean_df, 'VCF0143', [8,9,5],[np.nan, np.nan, 0], fillna = '9')
cln.convert_binary(final_df, clean_df, 'VCF0146', [8,9,2],[np.nan, np.nan, 0], 0, '9')
cln.convert_binary(final_df, clean_df, 'VCF0311', [-1],[np.nan], -1, '0')
cln.convert_binary_batch(final_df, clean_df, 'VCF0346', 'VCF0373', [0,2,8,9],
                         [np.nan, 0, np.nan, np.nan], offset = 0, fillna = '9')
cln.convert_binary_batch(final_df, clean_df, 'VCF0374', 'VCF0392', [0,5,8,9],
                         [np.nan, 0, np.nan, np.nan], offset = 0, fillna = '9')
cln.convert_binary_batch(final_df, clean_df, 'VCF0475', 'VCF0493', [0,5,8,9],
                         [np.nan, 0, np.nan, np.nan], offset = 0, fillna = '9')
cln.convert_binary(final_df, clean_df, 'VCF0450', [0,8,2],[np.nan,np.nan,0], 0, '8')
cln.convert_binary(final_df, clean_df, 'VCF0501', [0,8,9,1,2],[np.nan,np.nan,np.nan,0,1], 0, '9')
cln.convert_binary(final_df, clean_df, 'VCF0702', [0,1,2], [np.nan, 0, 1], 0, '0')
cln.convert_binary(final_df, clean_df, 'VCF0704a', [2,0], [0,np.nan], 0, '0')
cln.convert_binary(final_df, clean_df, 'VCF0707', [2,0], [0,np.nan], 0, '0')
cln.convert_binary(final_df, clean_df, 'VCF0708', [2,0], [0,np.nan], 0, '0')
cln.convert_binary_batch(final_df, clean_df, 'VCF0717', 'VCF0721', [-1],
                         [np.nan], offset = -1, fillna = '0')
cln.convert_binary(final_df, clean_df, 'VCF0724', [-1], [np.nan], -1, '0')
cln.convert_binary(final_df, clean_df, 'VCF0729', [-1], [np.nan], -1, '0')
cln.convert_binary(final_df, clean_df, 'VCF0731', [8,9,5],[np.nan, np.nan, 0], 0, '9')
cln.convert_binary_batch(final_df, clean_df, 'VCF0740', 'VCF0742', [0,2], [np.nan,0],
                         offset = 0, fillna = '0')
cln.convert_binary(final_df, clean_df, 'VCF0846', [8,0,2],[np.nan, np.nan, 0], 0, '0')
cln.convert_binary(final_df, clean_df, 'VCF0878', [0,8,9,5],[np.nan, np.nan, np.nan, 0], 0, '9')
cln.convert_binary(final_df, clean_df, 'VCF0905', [1,2,9],[0,1,np.nan], 0, '9')
cln.convert_binary(final_df, clean_df, 'VCF9021', [8,9,5],[np.nan, np.nan, 0], 0, '9')
cln.convert_binary(final_df, clean_df, 'VCF9029', [8,9,3],[np.nan, np.nan, 0], 0, '9')
cln.convert_binary_batch(final_df, clean_df, 'VCF9030a', 'VCF9031', [8,9,2,5],
                         [np.nan,np.nan,0,0], offset = 0, fillna = '9')

##########################
### NUMERICAL FEATURES ###
##########################

cln.convert_numerical_batch(final_df, clean_df, 'VCF0204', 'VCF0291', scale = 10, fillna = 99)
cln.convert_numerical_batch(final_df, clean_df, 'VCF0412', 'VCF0429', scale = 10, fillna = 99,
                            correction = True)
final_df.VCF0414 = final_df.VCF0414.replace(995.0, np.nan)
cln.convert_numerical(final_df, clean_df, 'VCF0471', scale = 10, fillna = 99)
cln.convert_numerical_batch(final_df, clean_df, 'VCF0906', 'VCF0909', scale = 10, fillna = 99)
cln.convert_numerical(final_df, clean_df, 'VCF9005', scale = 10, fillna = 99)
cln.convert_numerical_batch(final_df, clean_df, 'VCF9056', 'VCF9057', scale = 10, fillna = 99)
cln.convert_numerical_batch(final_df, clean_df, 'VCF0648', 'VCF0656', scale = 10, fillna = 99)
cln.convert_numerical(final_df, clean_df, 'VCF0801', scale = 10, fillna = 99)

clean_df.VCF0138 = clean_df.VCF0138.fillna('0').apply(lambda x: int(x[0]))
cln.convert_numerical(final_df, clean_df, 'VCF0138', scale = 1, fillna = 99)
cln.convert_numerical(final_df, clean_df, 'VCF0723', scale = 1, fillna = 99)

clean_df.VCF0733 = clean_df.VCF0733.fillna('0').apply(lambda x: int(x[0]))
cln.convert_numerical(final_df, clean_df, 'VCF0733', scale = 1, fillna = 99)
cln.convert_numerical(final_df, clean_df, 'VCF1015', scale = 1, fillna = 99)
cln.convert_numerical(final_df, clean_df, 'VCF1016', scale = 1, fillna = 99)

cln.add_range(final_df, clean_df.VCF0114, 'VCF0114')

########################
### ORDINAL FEATURES ###
########################

cln.convert_ordinal(final_df, clean_df, 'VCF0503', [8,0], np.nan, fillna = '0', dk = 8,
                    offset = 0, parse_string = True)
cln.convert_ordinal(final_df, clean_df, 'VCF0504', [8,0], np.nan, fillna = '0', dk = 8,
                    offset = 0, parse_string = True)
cln.convert_ordinal_batch(final_df, clean_df, 'VCF0513','VCF0550', [0,9,8], np.nan, fillna = 0,
                          dk = 8, offset = 0, parse_string = False)
cln.convert_ordinal(final_df, clean_df, 'VCF0803', [9,0], [4,np.nan], fillna = 0, dk = 9,
                    offset = 0, parse_string = True)
cln.convert_ordinal(final_df, clean_df, 'VCF0804', [8,9,0], np.nan, fillna = 0, dk = 9,
                    offset = 0, parse_string = True)
cln.convert_ordinal(final_df, clean_df, 'VCF0806', [8,9,0], np.nan, fillna = 0, dk = 9,
                    offset = 0, parse_string = False)
cln.convert_ordinal(final_df, clean_df, 'VCF0830', [8,9,0], np.nan, fillna = 0, dk = 9,
                    offset = 0, parse_string = False)
cln.convert_ordinal_batch(final_df, clean_df, 'VCF9081','VCF9087', [0,9,8], np.nan, fillna = 9,
                          dk = 8, offset = 0, parse_string = False)
cln.convert_ordinal_batch(final_df, clean_df, 'VCF9089','VCF9095', [0,9,8], np.nan, fillna = 9,
                          dk = 8, offset = 0, parse_string = False)
cln.convert_ordinal_batch(final_df, clean_df, 'VCF0886','VCF0894', [0,9,8], np.nan, fillna = '9',
                          dk = 8, offset = 0, parse_string = True)
cln.convert_ordinal_batch(final_df, clean_df, 'VCF9047','VCF9049', [0,9,8,7],
                          [np.nan,np.nan,np.nan,3], fillna = '9', dk = 8, offset = 0,
                          parse_string = True)
cln.convert_ordinal_batch(final_df, clean_df, 'VCF0851','VCF0854', [8,9],
                          np.nan, fillna = '9', dk = 8, offset = 0, parse_string = True)
cln.convert_ordinal_batch(final_df, clean_df, 'VCF9013','VCF9018', [8,9],
                          np.nan, fillna = '9', dk = 8, offset = 0, parse_string = True)
cln.convert_ordinal_batch(final_df, clean_df, 'VCF9039','VCF9042', [8,9],
                          np.nan, fillna = '9', dk = 8, offset = 0, parse_string = True)
cln.convert_ordinal(final_df, clean_df, 'VCF0110', 0, np.nan, fillna = 0, dk = 8,
                    offset = 0, parse_string = True)
cln.convert_ordinal(final_df, clean_df, 'VCF0140a', [8,9], np.nan, fillna = 0, dk = 8,
                    offset = 0, parse_string = True)

column = clean_df.VCF0127a.apply(lambda x: int(re.search(r'(\d+)',x).group(1)))
column = column.replace([0, 11, 12, 13, 21, 99, 22, 23, 31], [0, 3, 2, 1, 6, 0, 5, 4, 7])
final_df['VCF0127a'] = column

cln.convert_ordinal(final_df, clean_df, 'VCF0130', [8,9,1,2,4,5], [np.nan,np.nan,5,4,2,1],
                    fillna = 0, dk = 8, offset = 0, parse_string = True)
cln.convert_ordinal(final_df, clean_df, 'VCF0148', 10, np.nan, fillna = '9', dk = 8,
                    offset = 1, parse_string = True)
cln.convert_ordinal(final_df, clean_df, 'VCF0301', 0, np.nan, fillna = 0, dk = 8,
                    offset = 0, parse_string = True)
cln.convert_ordinal(final_df, clean_df, 'VCF0303', 0, np.nan, fillna = 0, dk = 8,
                    offset = 0, parse_string = True)
cln.convert_ordinal(final_df, clean_df, 'VCF0305', 0, np.nan, fillna = 0, dk = 8,
                    offset = 0, parse_string = True)
cln.convert_ordinal(final_df, clean_df, 'VCF0310', [0,9], np.nan, fillna = 0, dk = 8,
                    offset = 0, parse_string = True)
cln.convert_ordinal(final_df, clean_df, 'VCF0451', [0,8,1,2,3,4], [np.nan,np.nan,4,3,2,1], 
                    fillna = '8', dk = 8, offset = 0, parse_string = True)
cln.convert_ordinal(final_df, clean_df, 'VCF0502', [2,9,0],[3,2,np.nan], 
                    fillna = '0', dk = 8, offset = 0, parse_string = True)
cln.convert_ordinal(final_df, clean_df, 'VCF0604', [0,9], np.nan, 
                    fillna = '0', dk = 8, offset = 0, parse_string = True)
cln.convert_ordinal(final_df, clean_df, 'VCF0605', [0,9,2], [np.nan,2,3], 
                    fillna = '0', dk = 8, offset = 0, parse_string = True)
cln.convert_ordinal(final_df, clean_df, 'VCF0606', [0,9], np.nan, 
                    fillna = '0', dk = 9, offset = 0, parse_string = True)
cln.convert_ordinal(final_df, clean_df, 'VCF0624', [0,9], np.nan, 
                    fillna = '0', dk = 9, offset = 0, parse_string = True)
cln.convert_ordinal(final_df, clean_df, 'VCF0703', 0, np.nan, 
                    fillna = '0', dk = 9, offset = 0, parse_string = True)
cln.convert_ordinal(final_df, clean_df, 'VCF0838', [0,9], np.nan, 
                    fillna = '0', dk = 9, offset = 0, parse_string = True)
cln.convert_ordinal(final_df, clean_df, 'VCF0847', [8,0,5], [np.nan,np.nan,0], 
                    fillna = '0', dk = 8, offset = 0, parse_string = True)
cln.convert_ordinal(final_df, clean_df, 'VCF0850', [0,9], np.nan, 
                    fillna = '0', dk = 9, offset = 0, parse_string = True)
cln.convert_ordinal(final_df, clean_df, 'VCF0867a', [4,5,7,8,9], [3,4,np.nan,np.nan,np.nan], 
                    fillna = '9', dk = 7, offset = 0, parse_string = True)
cln.convert_ordinal(final_df, clean_df, 'VCF0870', [1,3,5,8,9,0], [1,2,3,np.nan,np.nan,np.nan], 
                    fillna = '9', dk = 8, offset = 0, parse_string = True)
cln.convert_ordinal(final_df, clean_df, 'VCF0871', [8,9,0], np.nan, 
                    fillna = '9', dk = 0, offset = 0, parse_string = True)
cln.convert_ordinal(final_df, clean_df, 'VCF0872', [1,3,5,8,9,0], [1,2,3,np.nan,np.nan,np.nan], 
                    fillna = '9', dk = 8, offset = 0, parse_string = True)
cln.convert_ordinal(final_df, clean_df, 'VCF0879', [8,9,0], np.nan, 
                    fillna = '9', dk = 8, offset = 0, parse_string = True)
cln.convert_ordinal(final_df, clean_df, 'VCF0880a', [8,9,0], np.nan, 
                    fillna = '9', dk = 8, offset = 0, parse_string = True)
cln.convert_ordinal(final_df, clean_df, 'VCF0881', [8,9,0], np.nan, 
                    fillna = '9', dk = 9, offset = 0, parse_string = True)
cln.convert_ordinal(final_df, clean_df, 'VCF9009', [9,8], [np.nan,3], 
                    fillna = '9', dk = 8, offset = 0, parse_string = True)
cln.convert_ordinal(final_df, clean_df, 'VCF9032', [9,8,0,1,3,5], [np.nan,np.nan,1,4,3,2], 
                    fillna = '9', dk = 8, offset = 0, parse_string = True)
cln.convert_ordinal(final_df, clean_df, 'VCF9045', [9,8,3,5], [np.nan,np.nan,2,3], 
                    fillna = '9', dk = 8, offset = 0, parse_string = True)

############################
### CATEGORICAL FEATURES ###
############################

cln.add_onehot(final_df, clean_df, 'VCF0105a', 9, 0, fillna = '0')
cln.add_onehot(final_df, clean_df, 'VCF0107', [7,8,9], [5,6,0], fillna = '0')
cln.add_onehot(final_df, clean_df, 'VCF0112', None, None, fillna = '0')
cln.add_onehot(final_df, clean_df, 'VCF0116', [4,5,6,7,8,9], [3,4,5,6,7,0], fillna = '0')
cln.add_onehot(final_df, clean_df, 'VCF0128', None, None, fillna = '0')
cln.add_onehot(final_df, clean_df, 'VCF0147', [9,8], [0,1], fillna = '0')
cln.add_onehot(final_df, clean_df, 'VCF0149', [5,9,6], 0, fillna = '0')
cln.add_onehot(final_df, clean_df, 'VCF0302', [8,9], 0, fillna = '0')
cln.add_onehot(final_df, clean_df, 'VCF0704', None, None, fillna = '0')
cln.add_onehot(final_df, clean_df, 'VCF0709', None, None, fillna = '0')
cln.add_onehot(final_df, clean_df, 'VCF0710', None, None, fillna = '0')
cln.add_onehot(final_df, clean_df, 'VCF0713', 9, 5, fillna = '0')
cln.add_onehot(final_df, clean_df, 'VCF0714', 9, 0, fillna = '0')
cln.add_onehot(final_df, clean_df, 'VCF0734', None, None, fillna = '0')
cln.add_onehot(final_df, clean_df, 'VCF0736', [5,7], [2,3], fillna = '0')
cln.add_onehot(final_df, clean_df, 'VCF0804', 9, 4, fillna = '0')
cln.add_onehot(final_df, clean_df, 'VCF0823', 9, 3, fillna = '0')
cln.add_onehot(final_df, clean_df, 'VCF0904', [9,0], [0,3], fillna = '9')
cln.add_onehot(final_df, clean_df, 'VCF1004', 9, 0, fillna = '9')
cln.add_onehot(final_df, clean_df, 'VCF9030', [6,8], 7, fillna = '9')
cln.add_onehot(final_df, clean_df, 'VCF9131', [8,9], [3,0], fillna = '9')
cln.add_onehot(final_df, clean_df, 'VCF9132', [8,9], [3,0], fillna = '9')
cln.add_onehot(final_df, clean_df, 'VCF9133', [8,9], [3,0], fillna = '9')

final_df.to_csv('../data/anes_cdf_converted.csv')
