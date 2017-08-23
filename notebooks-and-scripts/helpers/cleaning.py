#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 13:10:35 2017

These helper functions assist in encoding various features into a format
suitable for a machine learning algorithm.

@author: derekzhao
"""

import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt

print_verification = True
               
#######################       
### BINARY ENCODING ###
#######################

def convert_binary(out_df, in_df, column_name, replace_value, replace_with, offset = 0,
                   fillna = '9'):
    """
    Performs binary encoding on a column of in_df, appends the result to out_df, 
    and generates console output for accuracy verification.
    
    INPUT:
    - out_df: DataFrame that the processed column will be appended to
    - in_df: DataFrame from which the column will be processed
    - column_name: identifies the column to be processed
    - replace_value: list of values to be replaced in the processed column
    - replace_with: list of values with which to substitute
    - offset: subtracts a uniform value from all entries in the column
    - fillna: value to replace missing data
    """
    new_column = in_df[column_name].fillna(fillna).apply(lambda x: int(x[0]) + offset)
    new_column = new_column.replace(replace_value, replace_with)
    out_df[column_name] = new_column
          
    print('IN_DF:')
    print(in_df[column_name].value_counts().sort_index())
    print('OUT_DF:')
    print(out_df[column_name].value_counts().sort_index())
    print()

def convert_binary_batch(out_df, in_df, start_column, end_column, replace_value, 
                         replace_with, offset = 0, fillna = '9'):
    """
    Performs the same function as convert_binary but on a group of consecutive columns.
    
    INPUT:
    - see convert_binary
    - start_column / end_column: denote the names of the first and last columns of the group
    """
    startIndex = in_df.columns.get_loc(start_column)
    endIndex = in_df.columns.get_loc(end_column)
    
    for index in range(startIndex, endIndex + 1):
        convert_binary(out_df, in_df, in_df.columns[index], replace_value, replace_with,
                       offset, fillna)

##########################
### NUMERICAL ENCODING ###
##########################

def convert_numerical(out_df, in_df, column_name, scale = 1, fillna = 99, correction = False):
    """
    Performs numerical encoding on a column of in_df and appends the result to out_df.
    If a column has greater than 10 'Don't know' responses, an additional binary encoded column
    is generated. Plots are generated for accuracy verification.
    
    INPUT:
    - out_df: DataFrame that the processed column will be appended to
    - in_df: DataFrame from which the column will be processed
    - column_name: identifies the column to be processed
    - scale: specifies how to round each value
    - fillna: value to replace missing data
    """
    new_column = in_df[column_name].fillna(fillna)
    
    if max(new_column.dropna()) < 900 or correction:
        if sum(new_column == 98) >= 10:
            out_df[column_name+'_dk'] = new_column.apply(lambda x: x == 98)
            print('# DK\'s:', sum(out_df[column_name+'_dk']))
        new_column = new_column.replace([98,99,98.0,99.0],np.nan)
    
    if max(new_column.dropna()) >= 900:
        if sum(new_column == 997) >= 10:
            out_df[column_name+'_dk'] = new_column.apply(lambda x: x == 997)
            print('# DK\'s:', sum(out_df[column_name+'_dk']))
        new_column = new_column.replace(range(900,1000), np.nan)
        
    new_column = round(new_column / scale)
    out_df[column_name] = new_column
    
    if print_verification:
        fig, axes = plt.subplots(1, 2, figsize = (10, 3))
        in_df[column_name].value_counts().sort_index().plot(kind = 'bar', ax = axes[0])
        out_df[column_name].value_counts().sort_index().plot(kind = 'bar', ax = axes[1])
        axes[0].set(title = 'IN_DF: ' + column_name)
        axes[1].set(title = 'OUT_DF: ' + column_name)
    
def convert_numerical_batch(out_df, in_df, start_column, end_column, scale = 1, fillna = 99,
                            correction = False):
    """
    Performs the same function as convert_numerical but on a group of consecutive columns.
    
    INPUT:
    - see convert_numerical
    - start_column / end_column: denote the names of the first and last columns of the group
    """
    startIndex = in_df.columns.get_loc(start_column)
    endIndex = in_df.columns.get_loc(end_column)
    
    for index in range(startIndex, endIndex + 1):
        convert_numerical(out_df, in_df, in_df.columns[index], scale, fillna, correction)

########################
### ORDINAL ENCODING ###
########################
    
def convert_ordinal(out_df, in_df, column_name, replace_value, replace_with, fillna = 0, dk = 8,
                    offset = 0, parse_string = False):
    """
    Performs ordinal encoding on a column of in_df, appends the result to out_df, 
    and generates console output for accuracy verification.
    
    INPUT:
    - out_df: DataFrame that the processed column will be appended to
    - in_df: DataFrame from which the column will be processed
    - column_name: identifies the column to be processed
    - replace_value: list of values to be replaced in the processed column
    - replace_with: list of values with which to substitute
    - offset: subtracts a uniform value from all entries in the column
    - fillna: value to replace missing data
    - dk: value respresenting 'Don't Know' responses; if column contains greater than 10 dk's
      a new binary column is created and appended to out_df
    - parse_string: set to True if specified column contains string values
    """
    
    new_column = in_df[column_name].fillna(fillna)
    if parse_string == True:
        new_column = new_column.apply(lambda x: int(x[0]) + offset)
    dk_column = new_column == dk
    if sum(dk_column) >= 10:
        out_df[column_name + '_dk'] = dk_column
        print('# DK\'s:', sum(out_df[column_name+'_dk']))
    new_column = new_column.replace(replace_value, replace_with)
    out_df[column_name] = new_column
    
    if print_verification:
        fig, axes = plt.subplots(1, 2, figsize = (10, 3))
        in_df[column_name].value_counts().sort_index().plot(kind = 'bar', ax = axes[0])
        out_df[column_name].value_counts().sort_index().plot(kind = 'bar', ax = axes[1])
        axes[0].set(title = 'IN_DF: ' + column_name)
        axes[1].set(title = 'OUT_DF: ' + column_name)
    
def convert_ordinal_batch(out_df, in_df, start_column, end_column, replace_value, replace_with, fillna = 0, dk = 8,
                    offset = 0, parse_string = False):
    """
    Performs the same function as convert_ordinal but on a group of consecutive columns.
    
    INPUT:
    - see convert_ordinal
    - start_column / end_column: denote the names of the first and last columns of the group
    """
    startIndex = in_df.columns.get_loc(start_column)
    endIndex = in_df.columns.get_loc(end_column)
    
    for index in range(startIndex, endIndex + 1):
        convert_ordinal(out_df, in_df, in_df.columns[index], replace_value, replace_with, fillna, 
                        dk, offset, parse_string)
        
############################
### CATEGORICAL ENCODING ###
############################

def add_binary(out_df, in_df, column_name):  
    """
    Performs binary encoding on specified column and adds the resulting dataframe to out_df.
    Binary encoding converts a categorical variable to an ordinal variable, then converts that
    ordinal value to a binary string. Each digit of the string represents a new column feature.
    
    INPUT:
    - out_df: DataFrame to which binary encoded columns are appended
    - in_df: DataFrame containing the column to be encoded
    - column_name: specifies the column to encode
    """
    new_column = in_df[column_name]
    maxValue = max(new_column)
    numDigits = int(np.log2(maxValue) + 1)
    for n, i in zip(range(numDigits-1, -1, -1), range(1, numDigits+1)):
        col1 = new_column // 2 ** n
        new_column = new_column % 2 ** n
        column_name = column_name + '_b' + str(i)
        out_df[column_name] = col1
                   
def add_onehot(out_df, in_df, column_name, replace_value, replace_with, fillna = '0'):
    """
    Performs one-hot encoding on specified column and adds the resulting dataframe to out_df.
    
    INPUT:
    - out_df: DataFrame that the processed dummy columns will be appended to
    - in_df: DataFrame from which the column will be processed
    - column_name: identifies the column to be processed
    - replace_value: list of values to be replaced in the processed column
    - replace_with: list of values with which to substitute
    - fillna: value to replace missing data
    """
    new_column = in_df[column_name]
    
    print('IN_DF:')
    print(new_column.value_counts().sort_index())
    
    new_column = new_column.fillna('0').apply(lambda x: int(x[0]))
    new_column = new_column.replace(replace_value, replace_with)
    numColumns = len(new_column.value_counts())
    
    dummies = pd.get_dummies(new_column)
    dummies.columns = [column_name + '_oh' + str(i) for i in range(0, numColumns)]
    
    print('OUT_DF:')
    for each in dummies.columns:
        out_df[each] = dummies[each]
        print(each + ':', sum(out_df[each]))
    

def add_range(inDataFrame, inColumn, inColumnName):
    """
    Creates two columns that represent the lower and upper bounds of inColumn.
    
    INPUT:
    - inDataFrame: DataFrame to which newly encoded binary features will be added
    - inColumn: DataFrame column of range variables
    - inColumnName: column name onto which postfixes will be appended
    """
    def get_range(x, n):
        if x == 'NA':
            return np.nan
        else:
            return float(re.search(r'\d+\.\s*(\d+)\sto\s(\d+)', x).group(n))
    
    inColumn = inColumn.fillna('NA')
    inColumn[inColumn.str.contains('NA')] = 'NA'
    column1Name = inColumnName + '_r1'
    column2Name = inColumnName + '_r2'
    newColumn1 = inColumn.apply(lambda x: get_range(x, 1))
    newColumn2 = inColumn.apply(lambda x: get_range(x, 2))
    inDataFrame[column1Name] = newColumn1
    inDataFrame[column2Name] = newColumn2
        
    
    