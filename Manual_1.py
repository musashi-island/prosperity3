#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 14:43:40 2025

@author: iyad
"""
import math
from itertools import product
objects = ["SNOWBALL", "PIZZA", "SILICON", "SEASHELL"]

rates = {
        "SNOWBALL": {"SNOWBALL": 1, "PIZZA": 1.45, "SILICON": 0.52, "SEASHELL": 0.72},
        "PIZZA": {"SNOWBALL": 0.7, "PIZZA": 1, "SILICON": 0.31, "SEASHELL": 0.48},
        "SILICON": {"SNOWBALL": 1.95, "PIZZA": 3.1, "SILICON": 1, "SEASHELL": 1.49},
        "SEASHELL": {"SNOWBALL": 1.34, "PIZZA": 1.98, "SILICON": 0.64, "SEASHELL": 1}
        }   

def optimal_path():
    max_return = 1
    max_path = ("SEASHELL","SEASHELL","SEASHELL","SEASHELL","SEASHELL","SEASHELL")
    paths = [("SEASHELL", *p, "SEASHELL") for p in product(objects, repeat=4)]
    
    for path in paths:
        path_return = math.prod(rates[path[i]][path[i+1]] for i in range(5))
        if path_return > max_return:
            max_return = path_return
            max_path = path
    return max_path, max_return
        
    