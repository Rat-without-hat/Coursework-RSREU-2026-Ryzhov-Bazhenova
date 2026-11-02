import numpy as np
import pandas as pd
import heapq
from typing import Callable

"""
Функции для расчета разных расстояний между точками
"""
def euclidean_dist(dot_x : np.ndarray, dot_y : np.ndarray):
    """
    Возвращает Евклидово расстояние между точками dot_x и dot_y
    """
    return np.sqrt(((dot_x - dot_y) ** 2).sum())

def manhattan_dist(dot_x : np.ndarray, dot_y : np.ndarray):
    """
    Возвращает Манхэттенское расстояние между точками dot_x и dot_y
    """
    return np.abs((dot_x - dot_y)).sum()

def cos_dist(dot_x : np.ndarray, dot_y : np.ndarray):
    """
    Возвращает Косинусное расстояние между точками dot_x и dot_y
    """
    return 1 - dot_x.dot(dot_y) / (np.linalg.norm(dot_x) * np.linalg.norm(dot_y))

k_near_dots = 11
obj_param = pd.read_csv("4-classes_1000x2/original.csv", sep = "," ,encoding = "utf-8")
obj_classes = pd.read_csv("4-classes_1000x2/class.csv", sep = "," ,encoding = "utf-8")

"""
Задание тестовой точки вручную
"""
test_dot = np.array([1, 2], dtype = "float64")

""" 
Задание тестовой точки случайным образом
"""
# test_dot = np.random.uniform(-6, 6, size = obj_param.shape[1])

"""
Brute Force
"""

def k_nn_brute_force(search_dot : np.ndarray, k : int, parameters : pd.DataFrame,
                     classes : pd.DataFrame, func_dist : Callable):

    heap_nearest_dots = []
    dict_cnt_classes_near = {i : 0 for i in range(classes.nunique().iloc[0])}

    for i in range(parameters.shape[0]):
        heap_nearest_dots.append((func_dist(search_dot, parameters.loc[i]), classes.loc[i].item()))

    heapq.heapify(heap_nearest_dots)

    for _ in range(k):
        cur_near_dot = heapq.heappop(heap_nearest_dots)
        dict_cnt_classes_near[cur_near_dot[1]] += 1

    return max(dict_cnt_classes_near, key = lambda i: dict_cnt_classes_near[i])

print(k_nn_brute_force(test_dot, k_near_dots, obj_param, obj_classes, cos_dist))