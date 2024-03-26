# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 01:00:18 2024

@author: hadaw
"""
from functools import lru_cache
import numpy as np
# lets create some basic functions for later use
# I need to add zip to the cycle function
# but i want to do it in an elegant way


#@lru_cache(maxsize=777)
def cycle(inputs):
    if isinstance(inputs, tuple):
        inputs = zip(inputs)
    try:
        while True:
            yield next(inputs)
    except:
        yield np.array(inputs)

    
def equal_to(data_one, data_two, boolean=False):
    # Turn both data points into np arrays
    if not isinstance(data_one, np.ndarray):
        data_one = np.array(data_one)
        
    if not isinstance(data_two, np.ndarray):
        data_two = np.array(data_two)
    
    
    # return boolean values if True else return 1,0
    if boolean:
        return data_one == data_two
    else:    
        return np.where(data_one == data_two, 1,0)
    
inputs = [1,2,3,4]
inputs_two = [2,3,4,4]

for inputs in cycle(inputs):
    print(inputs)

equal_to(inputs,inputs_two)    