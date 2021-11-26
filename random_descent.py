import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import random
import math
from scipy import stats
import cvxpy as cp
from scipy import stats

def RandomDescentStep(a,x,tracker):
    n = a.shape[1]
    m = a.shape[0]
    
    obj = ObjVal(a,x)
    i = np.random.choice(tracker)
    updated = False
    
    for j in np.arange(m):
        new_x = np.copy(x)
        col = np.zeros(m)
        col[j] = 1
        new_x[i] = col
        new_obj = ObjVal(a,new_x)
        if obj < new_obj:
            obj = new_obj
            x = new_x
            updated = True
            
    return x, updated, i

def RandomDescent(a_input):
    n = a_input.shape[1]
    m = a_input.shape[0]
    
    #create initial random output matrix x
    x = np.zeros(shape=(n,m))
    for i in range(n):
        row = np.zeros(m)
        xk = np.arange(m)
        pk = np.ones(m)/m
        dist = stats.rv_discrete(values=(xk,pk))
        x[i][dist.rvs(size=1)[0]] = 1

    tracker = np.arange(n)
    
    proceed = True
    
    obj_vals = []
    
    iterN = 0
    while proceed:
        iterN += 1
        obj = ObjVal(a_input, x)
        obj_vals.append(obj)
        x, updated, i = RandomDescentStep(a,x,tracker)
        if updated == False:
            tracker=np.delete(tracker,np.argwhere(x==i))
            
            if len(tracker) == 0:
                proceed = False
    print(iterN)    
    return x, obj_vals
            
            
            
            
    