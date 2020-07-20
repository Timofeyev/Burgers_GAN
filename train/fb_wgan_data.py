# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 14:07:33 2019

@author: Jeric
"""

import numpy as np
import math


#=============================================
#Sampling for stochatic gradient descent
#=============================================
#---------------
def sample_data_wor(iter_sample, n=200):
    
    global x_train_shuffle
    
    iter_per_epoch = int((Train_dim-1)/n)
    xm1x = np.zeros((2,4,n), np.float64)
    temp_xm1x = np.zeros((2, 4,Train_dim-1), np.float64)
    
    if iter_sample % iter_per_epoch == 0:
        j = 0
        for ii in range(Train_dim-1):
            temp_xm1x[0].T[j] = x_train[ii]
            temp_xm1x[1].T[j] = x_train[ii+1]
            j +=1
        temp_xm1x_T = temp_xm1x.T
        np.random.shuffle(temp_xm1x_T)
        x_train_shuffle = temp_xm1x_T.T
    indx = range(np.mod(iter_sample*n, Train_dim-1), np.mod(iter_sample*n + n  - 1 , Train_dim-1))
    j = 0
    for ii in indx:
        xm1x[0].T[j] = x_train_shuffle[0].T[ii]
        xm1x[1].T[j] = x_train_shuffle[1].T[ii]
        j +=1

    return xm1x[0][:2],xm1x[0][2:]

#-------------------
def sample_all_train():
    xm1x = np.zeros((2, 4,Train_dim-1), np.float64)
    j = 0
    for ii in range(Train_dim-1):
        xm1x[0].T[j] = x_train[ii]
        xm1x[1].T[j] = x_train[ii+1]
        j +=1

    return xm1x[0][:2],xm1x[0][2:]


#-------------------
def sample_all_valid():
    xm1x = np.zeros((2, 4,Valid_dim-1), np.float64)
    j = 0
    for ii in range(Valid_dim-1):
        xm1x[0].T[j] = x_valid[ii]
        xm1x[1].T[j] = x_valid[ii+1]
        j +=1

    return xm1x[0][:2],xm1x[0][2:]

#=============================================
#Load data for local averages U_0 and U_1 and subgrid tendencies G1 and G2
#=============================================
    
U_all = np.genfromtxt('u1.dat',dtype=np.float64, delimiter=' ')
G1_all = np.genfromtxt('Bu1.dat',dtype=np.float64, delimiter=' ')
G2_all = np.genfromtxt('Bu2.dat',dtype=np.float64, delimiter=' ')
subs = 1


x_all = np.vstack([U_all, G1_all, G2_all]).T
x = x_all[::subs]
lenx = len(x)

#=============================================
#Dividing data into training and 
#=============================================
Train_dim = 100000 
Valid_dim = lenx - Train_dim

x_train = x[:Train_dim]
x_valid = x[Train_dim:]
xm1x = np.zeros((2, 4,Train_dim-1), np.float64)


#=============================================
#First shuffle for stochastic gradient descent
#=============================================
j = 0
for ii in range(Train_dim-1):
    xm1x[0].T[j] = x_train[ii]
    xm1x[1].T[j] = x_train[ii+1]
    j +=1
xm1x_T = xm1x.T
np.random.shuffle(xm1x_T)
x_train_shuffle = xm1x_T.T


 





