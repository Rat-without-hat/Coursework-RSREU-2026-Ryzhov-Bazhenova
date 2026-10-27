import numpy as np
import pandas as pd
import matplotlib as mt
import heapq
from typing import Callable

def euclidean_dist(dot_x : np.ndarray, dot_y : np.ndarray):
    return np.sqrt(((dot_x - dot_y) ** 2).sum())

def manhattan_dist(dot_x : np.ndarray, dot_y : np.ndarray):
    return np.abs((dot_x - dot_y)).sum()

def cos_dist(dot_x : np.ndarray, dot_y : np.ndarray):
    return 1 - dot_x.dot(dot_y) / (np.linalg.norm(dot_x) * np.linalg.norm(dot_y))

k_near_dots = 11
obj_param = pd.read_csv("4-classes_1000x2/original.csv", sep = "," ,encoding = "utf-8")
obj_classes = pd.read_csv("4-classes_1000x2/class.csv", sep = "," ,encoding = "utf-8")

test_dot = np.array([1, 2], dtype = "float64")
# test_dot = np.random.uniform(-6, 6, size = obj_param.shape[1]) # Способ задания тестовой точки случайным образом

"""
Brute Force
"""

def k_nn_brute_force(search_dot : np.ndarray, k : int, parameters : pd.DataFrame,
                     classes : pd.DataFrame, func_dist : Callable):

    heap_nearest_dots = []
    dict_cnt_classes_near = {i : 0 for i in range(classes.nunique().iloc[0])}

    for i in range(parameters.shape[0]):
        cur_dot = np.array(parameters.loc[i])
        heap_nearest_dots.append((func_dist(search_dot, cur_dot), classes.loc[i].item()))

    heapq.heapify(heap_nearest_dots)

    for _ in range(k):
        cur_near_dot = heapq.heappop(heap_nearest_dots)
        dict_cnt_classes_near[cur_near_dot[1]] += 1

    return max(dict_cnt_classes_near, key = lambda i: dict_cnt_classes_near[i])

test_dot = pd.DataFrame(test_dot.reshape((1, 2)))
print(test_dot)