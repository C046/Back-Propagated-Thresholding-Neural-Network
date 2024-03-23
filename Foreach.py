
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 13:16:11 2024

@author: hadaw
"""
# Helped with AI, kind of just instructed it to do stuff for me. It was like me and AI together, and we made unison and hopefully butter
# Coding along with AI is the future.

import numpy as np
import plotly.graph_objects as go
import mpmath as mp
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
            try:    
                result = np.reshape(result, newshape=shape)
            except Exception as E:
                return result
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


def foil(y_hat):
    """
    Perform the FOIL operation on the input array using the foreach function.

    Parameters:
    y_hat : array-like
        Array containing values of y_hat.

    Returns:
    result : array-like
        Array resulting from the FOIL operation.
    """
    one_minus_y_hat = foreach(y_hat, 1, action=lambda x, y: x - y)
    result = foreach(y_hat, one_minus_y_hat, action=lambda x, y: x * y)
    return result


# # Perform one iteration of gradient descent
# updated_parameters = gradient_descent(parameters, gradients, learning_rate)


def loss_partial_derivative(true_labels, predicted_values):
    """
    Compute the partial derivative of the loss function with respect to the predicted values.

    Parameters:
    true_labels : array-like
        True labels.
    predicted_values : array-like
        Predicted values.

    Returns:
    result : array-like
        Array containing the computed partial derivatives.
    """
    n = 1 / true_labels.size
    top = foreach(predicted_values, true_labels, action=lambda x, y: x - y)
    bottom = foil(predicted_values)
    result = foreach(top, bottom, action=lambda x, y: x / y) * n
    return result

def linear_rate_of_change(loss_derivative, predicted_outputs):
    return foreach(loss_derivative, predicted_outputs, action=lambda x, y: x / y)

def linear_partial_derivative(loss_derivative, linear_rate_of_change):
    return foreach(loss_derivative, linear_rate_of_change, action=lambda x, y: x * y)


def LinearRegression(x, y, epsilon=1e-15):
    """
    Perform simple linear regression.

    Parameters:
    x : array-like or scalar
        Independent variable values.
    y : array-like or scalar
        Dependent variable values (predictions).

    Returns:
    intercept : mp.mpf
        Intercept of the regression line.
    slope : mp.mpf
        Slope of the regression line.
    """
    # Convert inputs to numpy arrays if they're not already
    if not isinstance(x, np.ndarray):
        x = np.array(x)
        
    if not isinstance(y, np.ndarray):
        y = np.array(y)
        
    x_mean, y_mean = np.mean(x), np.mean(y)
    
    if x_mean == x:
        pass
    else:
        x = x-x_mean
    
    if y_mean == y:
        pass
    else:
        y = y-y_mean
    
    

            
    slope = np.sum(x*y)/(np.sum(x**2)+epsilon)
    intercept = y_mean - slope * x_mean
    
    return intercept, slope


def binary_entropy_gradient(y, y_hat, epsilon=1e+15):
    y = np.array(y)
    y_hat = np.array(y_hat)
    
    
    
    result = -(y/(y_hat+epsilon)) + (1-y)/((1-y_hat)+epsilon)
    return result 

def binary_cross_entropy(ground_truth, predicted, epsilon=1e-15):
    
    ground_truth = np.array(ground_truth)
    predicted = np.array(predicted)
    
    n = 1 / ground_truth.size
    #predicted = np.clip(predicted, epsilon, 1 - epsilon)  # Clip predicted probabilities to prevent log(0) or log(1)
    
    # Apply the binary cross entropy formula using foreach function
    loss = foreach(ground_truth, predicted, action=lambda x, y: -n * (x * np.log(y+epsilon) + (1 - x) * np.log(1 - y+epsilon)))
    
    return loss