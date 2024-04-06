# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 07:29:52 2024

@author: hadaw
"""
import numpy as np
import inspect

class Iterate:
    def __init__(self, inputs=False):
        """
        Initialize Iterate object.
        
        Args:
            inputs (bool, list, int, float, optional): Initial input data. Defaults to False.
        """
        super().__init__()
        # Set inputs variable
        self.inputs = inputs
    
    def __add__(self, b, a=False):
        """
        Addition operation.

        Args:
            b (array-like or scalar): Second operand.
            a (array-like or scalar, optional): First operand. Defaults to False.
        
        Returns:
            numpy.ndarray: Sum of a and b, or a + b if they are scalars.
        """
        if a == False:
            a = self.inputs
            
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return a + b

        a_array = np.asarray(a) if isinstance(a, list) else (self.inputs if a is False else a)
        b_array = np.asarray(b) if isinstance(b, list) else b

        return np.add(a_array, b_array)
    
    
    def __subtract__(self, b, a=False):
        """
        Subtraction operation.

        Args:
            b (array-like or scalar): Second operand.
            a (array-like or scalar, optional): First operand. Defaults to False.
        
        Returns:
            numpy.ndarray: Difference of a and b, or a - b if they are scalars.
        """
        if a == False:
            a = self.inputs
            
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return a - b

        a_array = np.asarray(a) if isinstance(a, list) else (self.inputs if a is False else a)
        b_array = np.asarray(b) if isinstance(b, list) else b

        return np.subtract(a_array, b_array)
    
    
    def __multiply__(self, b, a=False):
        """
        Multiplication operation.

        Args:
            b (array-like or scalar): Second operand.
            a (array-like or scalar, optional): First operand. Defaults to False.
        
        Returns:
            numpy.ndarray: Product of a and b, or a * b if they are scalars.
        """
        if a == False:
            a = self.inputs
            
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return a * b

        a_array = np.asarray(a) if isinstance(a, list) else (self.inputs if a is False else a)
        b_array = np.asarray(b) if isinstance(b, list) else b

        return np.multiply(a_array, b_array)
    
    def __divide__(self, b, a=False):
        """
        Division operation.

        Args:
            b (array-like or scalar): Second operand.
            a (array-like or scalar, optional): First operand. Defaults to False.
        
        Returns:
            numpy.ndarray: Quotient of a and b, or a / b if they are scalars.
        """
        if a == False:
            a = self.inputs
            
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return a / b

        a_array = np.asarray(a) if isinstance(a, list) else (self.inputs if a is False else a)
        b_array = np.asarray(b) if isinstance(b, list) else b

        return np.divide(a_array, b_array)
    
    def __mod__(self, b, a=False):
        """
        Modulo operation.

        Args:
            b (array-like or scalar): Second operand.
            a (array-like or scalar, optional): First operand. Defaults to False.
        
        Returns:
            numpy.ndarray: Remainder of a divided by b, or a % b if they are scalars.
        """
        if a == False:
            a = self.inputs
            
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return a % b
        
        a_array = np.asarray(a) if isinstance(a, list) else (self.inputs if a is False else a)
        b_array = np.asarray(b) if isinstance(b, list) else b

        return np.mod(a_array, b_array)
    
    def __floor__(self, b, a=False):
        """
        Floor division operation.

        Args:
            b (array-like or scalar): Second operand.
            a (array-like or scalar, optional): First operand. Defaults to False.
        
        Returns:
            numpy.ndarray: Floor division of a by b, or a // b if they are scalars.
        """
        if a == False:
            a = self.inputs
            
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return a // b

        a_array = np.asarray(a) if isinstance(a, list) else (self.inputs if a is False else a)
        b_array = np.asarray(b) if isinstance(b, list) else b

        return np.floor_divide(a_array, b_array)
    
    def __equal__(self, b, a=False):
        """
        Equal comparison operation.

        Args:
            b (array-like or scalar): Second operand.
            a (array-like or scalar, optional): First operand. Defaults to False.
        
        Returns:
            numpy.ndarray: Boolean array indicating where elements of a and b are equal, or True if a and b are scalars and equal.
        """
        # Set a to self.inputs if not provided
        if a == False:
            a = self.inputs
            
        # Perform comparison for scalars
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return a == b

        # Convert a and b to numpy arrays if they are lists
        a_array = np.asarray(a) if isinstance(a, list) else (self.inputs if a is False else a)
        b_array = np.asarray(b) if isinstance(b, list) else b
        
        # Perform element-wise comparison
        return np.equal(a_array, b_array)
    
    def count_parameters(self, func):
        """
        Count the number of parameters in a function.

        Args:
            func (callable): The function to check.

        Returns:
            int: Number of parameters in the function.
        """
        signature = inspect.signature(func)
        return len(signature.parameters)
    
    def cycle(self, array, action=False):
        """
        Cycle through the given array applying an optional action.

        Args:
            array (iterable): Array to cycle through.
            action (callable, optional): Optional action to apply to each element. Defaults to False.

        Returns:
            list: List containing the results after cycling through the array.
        """
        # Initialize result cache
        res = []
        # Set array to an iterable
        iterable = iter(array)
        
        # Check for base inputs
        if self.inputs is not False:
            a = iter(self.inputs)
        else:
            if self.count_parameters(action) >= 2:
                a=iterable
            else:
                a=None
            
        
        try:
            # While iterable available 
            while iterable:
                # If conditional for action 
                if action is not False:
                    # yield action on iterable
                    yield action(next(iterable), a=next(a)) if a is not None else action(next(iterable))
                else:
                    # if that fails, append the original inputs to the cache
                    yield next(iterable)
        
        # raise an exception and return the result
        except StopIteration:
            return res
                
def sigmoid(x):
    threshold = np.random.uniform(np.random.uniform(0.45,0.49), np.random.uniform(0.49,np.random.uniform(0.55,0.59)))
    x = x + -.0000001e-2
    #return (1/50)* (1/(1+np.exp(x)**-x))
    result = (1/(1+(np.exp(x)**(-x))))
    
    return result, threshold
    

    
            
            
inputs = [-i+0.0001 for i in range(50)]
inputs_two = [4,3,2,1]
Iterator = Iterate()

for values in Iterator.cycle(inputs, action=sigmoid):
    print(values)