import numpy as np

def subtract_value(arr, value):
    return arr - value

def foreach(iterable, action=None, size=None, value=None):
    # Check if iterable is a matrix
    if len(iterable.shape) >= 1:
        shape = iterable.shape
        # Handle matrix operations
        if not callable(action):
            # If action is a scalar, perform element-wise multiplication
            result = np.reshape((iterable * action), newshape=shape)
        else:
            # Apply the action function to each element of the matrix
            result = action(iterable, value)
            # Reshape the result to match the original shape of the matrix
            result = np.reshape(result, newshape=shape)
    else:
        # Handle regular iterable operations
        if not callable(action):
            # If action is a scalar, perform element-wise multiplication
            result = iterable * action
        else:
            # Apply the action function to the entire array
            result = action(iterable, value)
    
    # Caching logic
    if size is not None:
        # Generate a unique key based on iterable, action, value, and shape
        key = (tuple(iterable.flatten()), action, value, shape)
        cached_results = foreach.__dict__.setdefault("cached_results", {})
        if key in cached_results:
            return cached_results[key]
        else:
            cached_results[key] = result
            if len(cached_results) > size:
                cached_results.pop(next(iter(cached_results)))
    
    return result

def MAS(values, predicted_values):
    if not isinstance(values, np.ndarray):
        values = np.array(values)
    
    # Subtract the mean from each value using foreach function
    return np.mean(foreach(values, subtract_value, size=2, value=np.mean(predicted_values)) ** 2)
    
# Test the MAS function
input_array = np.random.uniform(0, 10000, size=(1, 3))
predicted = np.random.uniform(0,10000, size=(1,3))
mas_value = MAS(input_array, predicted)
print("Mean Absolute Square:", mas_value)