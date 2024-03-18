
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

def calculate_gradient(true_labels, predicted_outputs, activations):
    """
    Compute the gradient of the loss function with respect to the parameters (weights and biases) of the neural network.

    Parameters:
    true_labels : array-like
        True labels.
    predicted_outputs : array-like
        Predicted output values.
    activations : list of array-like
        List containing the activations of each layer.

    Returns:
    gradients : list of dictionaries
        List containing dictionaries of gradients for each layer.
    """
    num_samples = true_labels.shape[0]
    num_layers = len(activations)
    gradients = []

    # Compute gradient for the output layer
    output_gradient = linear_derivative(true_labels, predicted_outputs)
    output_rate_of_change = linear_rate_of_change(output_gradient, activations[-2])
    output_partial_derivative = linear_partial_derivative(output_gradient, output_rate_of_change)
    output_gradients = {
        'weights': [],
        'biases': []
    }
    for i in range(len(output_partial_derivative)):
        # Compute gradient for weights
        weight_gradient = output_partial_derivative[i]
        output_gradients['weights'].append(weight_gradient)
        
        # Compute gradient for biases
        bias_gradient = output_partial_derivative[i]
        output_gradients['biases'].append(bias_gradient)
    gradients.append(output_gradients)

    # Compute gradient for hidden layers
    for layer in reversed(range(1, num_layers - 1)):
        hidden_gradient = None  # Compute gradient for hidden layer
        hidden_gradients = {
            'weights': [],
            'biases': []
        }
        for i in range(len(hidden_gradient)):
            # Compute gradient for weights
            weight_gradient = None
            hidden_gradients['weights'].append(weight_gradient)
            
            # Compute gradient for biases
            bias_gradient = None
            hidden_gradients['biases'].append(bias_gradient)
        gradients.append(hidden_gradients)

    return gradients

def gradient_descent(parameters, gradients, learning_rate):
    """
    Update the parameters (weights and biases) of the neural network using gradient descent.

    Parameters:
    parameters : list of dictionaries
        List containing dictionaries of parameters (weights and biases) for each layer.
    gradients : list of dictionaries
        List containing dictionaries of gradients for each layer.
    learning_rate : float
        Learning rate for gradient descent.

    Returns:
    updated_parameters : list of dictionaries
        List containing dictionaries of updated parameters (weights and biases) for each layer.
    """
    updated_parameters = []
    for layer_params, layer_gradients in zip(parameters, gradients):
        updated_layer_params = {
            'weights': [],
            'biases': []
        }
        for param, grad in zip(layer_params['weights'], layer_gradients['weights']):
            updated_param = foreach(param, grad, action=lambda x, y: x - learning_rate * y)
            updated_layer_params['weights'].append(updated_param)
        for param, grad in zip(layer_params['biases'], layer_gradients['biases']):
            updated_param = foreach(param, grad, action=lambda x, y: x - learning_rate * y)
            updated_layer_params['biases'].append(updated_param)
        updated_parameters.append(updated_layer_params)
    return updated_parameters


# # Example neural network parameters (weights and biases)
# parameters = [
#     {
#         'weights': [np.array([[0.1, 0.2], [0.3, 0.4]]), np.array([[0.5], [0.6]])],
#         'biases': [np.array([0.7, 0.8]), np.array([0.9])]
#     },
#     {
#         'weights': [np.array([[0.1, 0.2]]), np.array([[0.3]])],
#         'biases': [np.array([0.4]), np.array([0.5])]
#     }
# ]

# # Example gradients computed during backpropagation
# gradients = [
#     {
#         'weights': [np.array([[0.01, 0.02], [0.03, 0.04]]), np.array([[0.05], [0.06]])],
#         'biases': [np.array([0.07, 0.08]), np.array([0.09])]
#     },
#     {
#         'weights': [np.array([[0.01, 0.02]]), np.array([[0.03]])],
#         'biases': [np.array([0.04]), np.array([0.05])]
#     }
# ]

# # Learning rate for gradient descent
# learning_rate = 0.01

# # Perform one iteration of gradient descent
# updated_parameters = gradient_descent(parameters, gradients, learning_rate)

# # Print updated parameters
# print("Updated Parameters:")
for layer_params in updated_parameters:
    print(layer_params)
def linear_derivative(true_labels, predicted_values):
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

def linear_rate_of_change(linear_derivative, predicted_outputs):
    return foreach(linear_derivative, predicted_outputs, action=lambda x, y: x / y)

def linear_partial_derivative(linear_derivative, linear_rate_of_change):
    return foreach(linear_derivative, linear_rate_of_change, action=lambda x, y: x * y)

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


def binary_cross_entropy(ground_truth, predicted, epsilon=1e-15):
    n = 1 / ground_truth.size
    predicted = np.clip(predicted, epsilon, 1 - epsilon)  # Clip predicted probabilities to prevent log(0) or log(1)
    
    # Apply the binary cross entropy formula using foreach function
    loss = foreach(ground_truth, predicted, action=lambda x, y: -n * (x * np.log(y) + (1 - x) * np.log(1 - y)))
    
    return np.mean(loss)