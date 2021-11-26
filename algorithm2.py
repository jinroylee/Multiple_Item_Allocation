import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import math
from scipy import stats
import cvxpy as cp
from scipy import stats

def RANCE(x):
    # num_agents : Number of agents
    # num_items  : Number of items
    # x          : Solution matrix of the convex problem
    # p          : Matrix representing possibilities that theta_i will be greater than x[i,j] for each component of matrix x

    num_agents = x.shape[0]
    num_items = x.shape[1]

    p = 1-x

    #1. algorithm first stage 
    # m : Number of items
    # n : Number of agents
    # theta : numpy array storing the value of theta_i for each row i of the matrix x

    m = num_items
    n = num_agents
    theta = np.zeros(shape = m)

    x_theta = np.copy(x.T) # copy the transposed version of matrix x

    # store theta_i for each for of the matrix x
    for i in np.arange(m):
        theta[i] = float(np.random.uniform(0,1,1))

    # store binary values into matrix x_theta depend on the value of theta and x
    for i in np.arange(m):
        x_theta[i][x_theta[i] >= theta[i]] = 1
        x_theta[i][x_theta[i] < theta[i]] = 0

    x_theta = x_theta.T # transpose x_theta again

    
    #2. algrithm second stage
    # A : numpy array storing the cardinality of set A_i for each row i as an integer
    # r : output probability matrix of the fair contention resolution

    A = np.sum(x_theta, axis=1)
    r = np.zeros(shape = x_theta.shape)

    for i in range(len(r)):
        # if the cardinality of A_i is 1 or 0, the i th row of r is equal to the i th row of x_theta
        if A[i] <= 1:
            r[i] = x_theta[i]
        else:
            for j in range(len(r[i])):
                sigma = np.sum(p[i,x_theta[i]==1])/(A[i]-1) + np.sum(p[i,x_theta[i]==0])/(A[i])
                if x_theta[i,j] == 1 : #if agent j is included in set A, subtract p[i,j]/(A[i] - 1) from sigma
                    sigma -= p[i,j]/(A[i] - 1)
                else:
                    sigma = 0
                r[i,j]=(1/np.sum(p[i]))*sigma
        
    #3. assign probabilities
    for i in range(r.shape[0]):
        xk = np.arange(len(r[i]))
        if np.sum(r[i]) == 0:
            r[i] = 1/len(r[i])
        pk = r[i]
        dist = stats.rv_discrete(values=(xk,pk))
        r[i] = 0
        r[i][dist.rvs(size=1)[0]] = 1

    return r
