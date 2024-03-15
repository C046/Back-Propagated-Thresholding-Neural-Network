# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 13:16:11 2024

@author: hadaw
"""
# Helped with AI, kind of just instructed it to do stuff for me. It was like me and AI together, and we made unison and hopefully butter
# Coding along with AI is the future.
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
        # Initialize or retrieve the cached results dictionary
        cached_results = foreach.__dict__.setdefault("cached_results", {})
        
        # Generate a unique key based on iterable and action
        key = (tuple(iterable), action)
        
        # Check if the result has been cached
        if size is not None and key in cached_results:
            return cached_results[key]
        
        # Convert iterable to a NumPy array if not already
        iter_array = np.array(iterable)
        
        # Apply the action function to the entire array
        result = action(iter_array)
        
        # Cache the result for future use
        if size is not None:
            cached_results[key] = result
            if len(cached_results) > size:
                # Remove the oldest entry if cache size exceeds the limit
                cached_results.pop(next(iter(cached_results)))
        
    del iter_array, action, iterable
        
    return result
    


    
# Test the foreach function
input_array = [i for i in range(0,20000000)]
output_array = foreach(input_array, double_array, size=2)

print(output_array)
