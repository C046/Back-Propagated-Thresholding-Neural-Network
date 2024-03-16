
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 13:16:11 2024

@author: hadaw
"""
# Helped with AI, kind of just instructed it to do stuff for me. It was like me and AI together, and we made unison and hopefully butter
# Coding along with AI is the future.

import numpy as np
import plotly.graph_objects as go

def subtract_value(arr, value):
    return np.subtract(arr, value)

def add_value(arr, value):
    return np.add(arr, value)

def divide_value(arr, value):
    return np.divide(arr, value)

def multiply_value(arr, value):
    return np.multiply(arr, value)


def foreach(value, other_values, action, size=None):
    
    if not isinstance(value, np.ndarray):
        value = np.array(value)
        
    if not isinstance(other_values, np.ndarray):
        other_values = np.array(other_values)
    
        
    # Check if value is a matrix
    if len(value.shape) >= 1:
        shape = value.shape
        # Handle matrix operations
        if not callable(action):
            # If action is a scalar, perform element-wise multiplication
            result = np.reshape((value * action), newshape=shape)
        else:
            # Apply the action function to each element of the matrix
            result = action(value, other_values)
            # Reshape the result to match the original shape of the matrix
            result = np.reshape(result, newshape=shape)
    else:
        # Handle regular iterable operations
        if not callable(action):
            print(True)
            # If action is a scalar, perform element-wise multiplication
            result = value * action
        else:
            # Apply the action function to the entire array
            result = action(value, other_values)
    
    # Caching logic
    if size is not None:
        # Generate a unique key based on the relevant parameters
        key = (tuple(value.flatten()), action, tuple(other_values.flatten()))
        cached_results = foreach.__dict__.setdefault("cached_results", {})
        if key in cached_results:
            return cached_results[key]
        else:
            cached_results[key] = result
            if len(cached_results) > size:
                cached_results.pop(next(iter(cached_results)))
    
    return result

def MSE(values, predicted_values, size=None):
    if not isinstance(values, np.ndarray):
        if isinstance(values, (int, float)):
            values = [values]
        values = np.array(values)

    if not isinstance(predicted_values, np.ndarray):
        if isinstance(predicted_values, (int, float)):
            predicted_values = [predicted_values]
        predicted_values = np.array(predicted_values)
    
    n = 1/values.size
    
    return n*sum(foreach(values, predicted_values, action=subtract_value, size=size)**2)

def MAE(values, predicted_values, size=None):
    if not isinstance(values, np.ndarray):
        if isinstance(values, (int, float)):
            values = [values]
        values = np.array(values)

    if not isinstance(predicted_values, np.ndarray):
        if isinstance(predicted_values, (int, float)):
            predicted_values = [predicted_values]
        predicted_values = np.array(predicted_values)
    
    n = 1/values.size
    
    return n*sum(np.abs(foreach(values, predicted_values, action=subtract_value, size=size)))

def RMSE(values, predicted_values, size=None):
    if not isinstance(values, np.ndarray):
        if isinstance(values, (int, float)):
            values = [values]
        values = np.array(values)

    if not isinstance(predicted_values, np.ndarray):
        if isinstance(predicted_values, (int, float)):
            predicted_values = [predicted_values]
        predicted_values = np.array(predicted_values)   
        
    return np.sqrt(MSE(values,predicted_values, size=size))

def LinearRegression(values, predicted_values):
    """
    Perform simple linear regression.

    Parameters:
    values : array-like, shape (n_samples,)
        Independent variable values.
    predicted_values : array-like, shape (n_samples,)
        Dependent variable values (predictions).

    Returns:
    intercept : float
        Intercept of the regression line.
    slope : float
        Slope of the regression line.
    """
    if not isinstance(values, np.ndarray):
        if isinstance(values, (int, float)):
            values = [values]
        values = np.array(values)

    if not isinstance(predicted_values, np.ndarray):
        if isinstance(predicted_values, (int, float)):
            predicted_values = [predicted_values]
        predicted_values = np.array(predicted_values)

    # Calculate the means
    values_mean = np.mean(values)
    predicted_mean = np.mean(predicted_values)

    # Calculate the differences from the means
    diff_values = foreach(values, values_mean, action=subtract_value)
    diff_predicted = foreach(predicted_values, predicted_mean, action=subtract_value)

    # Calculate the slope
    numerator = foreach(diff_values, diff_predicted, action=multiply_value)
    denominator = foreach(diff_values, diff_values, action=multiply_value)
    slope = np.sum(numerator) / np.sum(denominator)

    # Calculate the intercept
    intercept = predicted_mean - slope * values_mean

    return intercept, slope

