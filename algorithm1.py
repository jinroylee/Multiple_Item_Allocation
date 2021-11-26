import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import math
from scipy import stats
import cvxpy as cp
from scipy import stats

def KTRounding(x):
    # num_agents : Number of agents
    # num_items  : Number of items
    # x          : Solution matrix of the convex problem

    num_agents = x.shape[0]
    num_items = x.shape[1]


    x = pd.DataFrame(x)
    
    # index the Dataframe
    #n : Set of agents that are represented by incrementing alphabetical characters
    #m : Set of items that are represented by incrementing integer numbers

    # initialize both n and m as empty sets
    n = set({}) 
    m = set({})

    # add elements to set n and m
    n_count, m_count = x.shape 

    n_elem = 'a'
    m_elem = '1'

    while (len(n) < n_count):
        n.add(n_elem)
        n_elem = chr(ord(n_elem) + 1)

    while (len(m) < m_count):
        m.add(m_elem)
        m_elem = str(int(m_elem)+1)

    # convert sets to lists and use them as index of the DataFrame x
    index_names = sorted(list(n))
    column_names = sorted(list(m))

    x.index = index_names
    x.columns = column_names

    # apply algorithm
    # S   : Set S initialized as an empty set
    # S_i : Dictionary of set where keys are item i, and values are set of agents that are allocated item i. All sets are 
    #       initialized as empty set. 

    S = set({})
    S_i = {}

    # initialize an empty set for each item i
    for i in m:
        S_i[i] = set({})

    while (S != n):
        m = list(m)
        i = random.choice(m)
        theta = float(np.random.uniform(0,1,1)) # select random number theta from uniform distribution [0,1]
        S_theta = set({})
        # within row i, add element to S_theta iif its value is equal or greater than theta
        col_i = x[i]
        for idx in col_i.index:
            if idx not in S:
                if col_i[idx] >= theta:
                    S_theta.add(idx)
        # merge S_theta to S and S_i        
        S = S.union(S_theta)
        S_i[i] = S_i[i].union(S_theta)
        
        S_i = dict(sorted(S_i.items()))
    
    #change the dictionary of sets to matrix
    item_list = sorted(list(S_i.keys()))
    agent_list = sorted(list(S))
    output = pd.DataFrame(columns = item_list, index = agent_list)

    for k in S_i.keys():
        for v in list(S_i[k]):
            output[k][v] = 1
            
    output = output.fillna(0)
    output = np.array(output)
    
    return output

    