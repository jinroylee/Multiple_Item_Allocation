import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import math
from scipy import stats
import cvxpy as cp
from scipy import stats

def IntProg(a_input):
    n = a_input.shape[1]
    m = a_input.shape[0]

    A = np.zeros(shape=(n, n*m))

    beg = 0
    end = m
    for i in np.arange(n):
        idx = np.linspace(beg, end, m, endpoint = False)
        idx = idx.astype(int)
        A[i,idx] = 1
        beg += m
        end += m
    
    # x : The solution variable
    # a : Parameter a
    # A : Parameter A
    # b : Parameter b, which is equal to the vector of size n including ones.

    x = cp.Variable((n*m), integer = True)

    A_param = cp.Parameter((n, n*m))
    b = cp.Parameter(n)

    A_param.value = A
    b.value = np.ones(shape=(n))
    a = np.array(a_input) #assign input value of the function
    a = a.reshape(m, n*n)
    
    constraints = [0 <= x, A_param @ x == b]

    def obj(a,x):
        out = 0
        idx = np.arange(x.shape[0]) 

        for i in np.arange(a.shape[0]): # loop over each item i
            # extract the indices of vector x that correspond to the item i
            # size of x_idcs must be m
            x_idcs = idx[idx % m == i]
            for j in np.arange(a.shape[1]): #loop over each element of a^{i}
                # a_row corresponds to the index ji and a_col corresponds to the index ki
                a_row, a_col = (math.floor(j / n), j % n)
                if a_row != a_col:
                    out += a[i][j] * cp.minimum(x[x_idcs[a_row]], x[x_idcs[a_col]])
        return out

    objective = cp.Maximize(obj(a,x))

    prob = cp.Problem(objective, constraints)

    result = prob.solve()


    x = x.value.reshape(n,m)

    return x