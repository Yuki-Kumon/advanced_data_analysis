# -*- coding: utf-8 -*-

"""
テスト。
Author :
    Yuki Kumon
Last Update :
    2019-04-18
"""

import numpy as np


A = np.array([[8, 6, 4, 8, 5, 3, 8, 8, 5, 4, 6, 3],
       [2, 6, 4, 7, 1, 3, 9, 1, 3, 7, 7, 1],
       [2, 3, 5, 7, 6, 3, 9, 7, 5, 5, 9, 5],
       [7, 5, 7, 7, 5, 7, 9, 8, 6, 8, 7, 7]])

B = [A[:, 0:4], A[:, 4:8], A[:, 8:12]]
B = np.vsplit(A, 4)

print(B)

A = np.array([8, 3, 3, 8, 4, 1])
B = np.sort(A)
# print(B)

A = np.array([[6, 5, 2, 6, 5, 4, 4, 6],
       [4, 7, 5, 6, 7, 9, 7, 1],
       [1, 8, 5, 6, 1, 5, 3, 4],
       [1, 7, 9, 6, 9, 6, 4, 2]])
B = np.array([A[0:2, :], A[2:4, :]])

B = A.reshape(2, 2, 8)
# print(B)

A = np.array([7, 2, 8, 2, 6, 2, 7, 5, 3, 4, 5, 3, 7, 0, 5, 3, 8, 7, 4, 3])
B = np.reshape(A, (5, 4))
# print(B)

A = np.array([5, 4, 2, 6, 7, 2])
B = np.sort(A)
# print(B)

# print(np.eye(7))
print(np.eye(7))

A = np.array([7, 3, 1, 4, 7, 0, 4, 5, 7, 4, 8, 2, 2, 5, 7, 7, 0, 1, 5, 4, 5, 2,
       0, 7, 8])
B = A.reshape(5, 5)
print(B)

A = np.array([[4],
       [5],
       [4],
       [8],
       [7],
       [3]])
B = A.reshape(6)
# print(B)


A = np.array([[9, 4, 4, 8, 8, 9, 1, 9, 9],
       [4, 7, 5, 7, 3, 6, 7, 6, 4],
       [3, 9, 9, 9, 8, 9, 2, 6, 6]])
B = A.reshape(3, 3, 3)
print(B)

A = np.array([[7, 7, 3, 8],
       [7, 3, 1, 4],
       [3, 2, 8, 7],
       [8, 4, 7, 2]])
B = np.array([[6, 3, 2, 1],
       [6, 8, 8, 1],
       [7, 8, 2, 6],
       [1, 7, 5, 3]])
# print(A + B)

A = np.array([[3, 0, 5, 3, 0, 3, 1, 2],
       [7, 0, 4, 6, 5, 2, 5, 7],
       [7, 5, 4, 6, 4, 7, 1, 7],
       [1, 2, 2, 1, 4, 2, 4, 8],
       [0, 8, 5, 3, 5, 8, 5, 0],
       [3, 8, 8, 1, 3, 8, 4, 7],
       [3, 6, 3, 6, 7, 8, 4, 2],
       [8, 4, 6, 4, 3, 7, 4, 1]])
B = np.diag(A[::-1])
C = np.diag(B)[::-1]
# print(C)

A = np.array([[2, 1, 8, 5],
       [4, 8, 0, 3],
       [3, 0, 6, 5]])
B = np.array([[7, 8, 5, 6],
       [5, 4, 7, 1],
       [2, 6, 5, 1]])
C = np.c_[A, B]
# print(C)


A = np.array([[40, 40, 70, 80, 40],
       [10, 20, 30, 90, 30],
       [60, 70, 10, 50, 90],
       [90, 50, 10, 70, 90],
       [20, 40, 80, 70, 50],
       [10, 30, 20, 60, 50],
       [30, 60, 40, 40, 30]])
B = np.array([[9, 5, 3, 6, 5]])
# print(A * B)

A = np.array([[4],
       [5],
       [4],
       [8],
       [7],
       [3]])
B = A.T[0]
B = np.r_[A, A].T[0, :8]
print(B)

A = np.array([8, 9, 5, 4])
B = A[[2,3,0,1]]
# print(B)
