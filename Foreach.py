# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 13:16:11 2024

@author: hadaw
"""

import numpy as np

def double_array(arr):
    return arr * 2

def foreach(iterable, action=None, size=None):
    # Check if action is a function
    if not callable(action):
        # If iterable is already a NumPy array, perform the operation directly
        if isinstance(iterable, np.ndarray):
            return iterable * action
        else:
            # Otherwise, convert to NumPy array and perform the operation
            return np.array(iterable) * action
    else:
        # Check if the result has been cached
        if size is not None and "cached_results" in foreach.__dict__:
            cached_results = foreach.__dict__["cached_results"]
            for cached_iterable, cached_action, cached_output in cached_results:
                if np.array_equal(np.array(iterable), cached_iterable) and action == cached_action:
                    return cached_output
        
        # Convert iterable to a NumPy array if not already
        iter_array = np.array(iterable)
        
        # Apply the action function to the entire array
        result = action(iter_array)
        
        # Cache the result for future use
        if size is not None:
            foreach.__dict__.setdefault("cached_results", []).append((iter_array, action, result))
            if len(foreach.__dict__["cached_results"]) > size:
                foreach.__dict__["cached_results"].pop(0)
        else:
            foreach.__dict__["cached_results"] = [(iter_array, action, result)]
        
    del iter_array, action, iterable
        
    return result
    
# Test the foreach function
input_array = [i for i in range(0,10000000)]
output_array = foreach(input_array, double_array, size=2)

print(output_array)
