import numpy as np
from math import *

def laplacians(A):
    n = A.shape[0]
    m = A.shape[1]
    A1 = A
    D = np.sum(A1, axis=1)
    A_L = np.zeros(A.shape)
    for i in range(n):
        for j in range(m):
            if i == j and D[i] != 0:
                A_L[i, j] = 1
            elif i != j and A1[i, j] != 0:
                A_L[i, j] = (-1) / sqrt(D[i] * D[j])
            else:
                A_L[i, j] = 0
    return A_L


def max_min_normalize(a):                              #矩阵归一化
    sum_of_line = np.sum(a, axis=1)
    line = a.shape[0]
    row = a.shape[1]
    i = 0
    while i < line:
        j = 0
        while j < row:
            if sum_of_line[i] != 0:
                max = np.max(a[i])
                min = np.min(a[i])
                a[i, j] = (a[i, j]-min) / (max-min)
            j = j + 1
        i = i + 1
    return a
