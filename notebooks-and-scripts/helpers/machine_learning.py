#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 20:56:50 2017

@author: derekzhao
"""
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin

class Normalizer(TransformerMixin):
    
    standard_deviations = np.array
    means = np.array
    numeric_indices = np.array
    binary = bool
    
    def __init__(self, binary = False):
        self.binary = binary
        
    def fit(self, df, y = None):
        df = pd.DataFrame(df)
        num_unique = np.array([df.iloc[:,i].nunique() for i in range(0, df.shape[1])]) 
        self.numeric_indices = num_unique > 2
        self.means = df.iloc[:,self.numeric_indices].mean()
        #self.standard_deviations = np.std(df, axis = 0)
        self.standard_deviations = df.iloc[:,self.numeric_indices].std(ddof = 0)   
        return self
                      
        
    def transform(self, df):
        df = pd.DataFrame(df)
        if self.binary:
            df = (df - self.means) / self.standard_deviations
        else:
            df.iloc[:,self.numeric_indices] = \
            (df.iloc[:,self.numeric_indices] - self.means) / self.standard_deviations
        return np.array(df)
    
    #def fit_transform(self, df, *_):
        #self.fit(df)
        #df = self.transform(df)
        #return df
        
        
class SoftVotingClassifier:
    def __init__(self, estimators, weights = None, print_cv = True):
        self.estimators = estimators
        self.n = len(estimators)
        if weights == None:
            self.weights = np.ones(self.n)
        else:
            self.weights = weights
        self.blending = LinearRegression()
        self.print_cv = print_cv
        
    def fit(self, X_train_list, y_train):
        assert(self.n == len(X_train_list))
        predict_probas = np.zeros([len(y_train), self.n])
        for i in range(0, self.n):
            self.estimators[i].fit(X_train_list[i], y_train)
            predict_probas[:,i] = self.estimators[i].predict_proba(X_train_list[i])[:,1]
            
        y_pred = np.dot(predict_probas, self.weights) / self.weights.sum()
        
        return y_pred, predict_probas
    
    def predict_proba(self, X_test_list, y_test):
        assert(self.n == len(X_train_list))
        predict_probas = np.zeros([len(y_test), self.n])
        for i in range(0, self.n):
            predict_probas[:,i] = self.estimators[i].predict_proba(X_test_list[i])[:,1]
        
        y_pred = np.dot(predict_probas, self.weights) / self.weights.sum()
        
        return y_pred