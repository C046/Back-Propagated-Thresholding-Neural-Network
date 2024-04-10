# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 07:29:52 2024

@author: hadaw
"""
import numpy as np
import inspect
import mpmath as mp
from scipy.stats import linregress
import matplotlib.pyplot as plt
import pandas as pd
import os

# set the working directory
os.chdir("D:/.WindowsAPI")

mp.dps = 2

# set data path
data_path = "breast-cancer.csv"
# load the data
data =  pd.read_csv(data_path)

# set the inputs
inputs = data.drop('diagnosis', axis=1)
# set the labels
actual_labels = (data['diagnosis'].values == 'M').astype(int)




def calculate_accuracy(array1, array2):
    """
    Calculate the accuracy based on the element-wise comparison of two arrays.

    Parameters:
    array1 : numpy.ndarray
        First array for comparison.
    array2 : numpy.ndarray
        Second array for comparison.

    Returns:
    accuracy : float
        Accuracy of the second array compared to the first array.
    """
    # Element-wise comparison
    element_wise_comparison = array1 == array2

    # Calculate accuracy
    accuracy = np.sum(element_wise_comparison) / len(array1)

    return accuracy


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
    
    def __log__(self, b):
        """
        Equal comparison operation.

        Args:
            b (array-like or scalar): Second operand.
            a (array-like or scalar, optional): First operand. Defaults to False.
        
        Returns:
            numpy.ndarray: Boolean array indicating where elements of a and b are equal, or True if a and b are scalars and equal.
        """
        # Perform comparison for scalars
        #b = b[0]
        
        if isinstance(b, (int, float,complex, mp.mpf)):
            return np.log(b)
        
        
        b_array = np.asarray(b) if isinstance(b, (list)) else b
        
        result = []
        # Perform element-wise comparison
        for value in b_array:
            result.append(np.log(value))
            
        return np.array(result)
    

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

     

class Network:
    def __init__(self, actual_labels, inputs, weights=None, bias=None, learning_rate=0.0001):
        super().__init__()
        self.plt = plt
        
        # create a blank figure
        slopes, intercepts = [0],[0]
        self.plt.figure(figsize=(8, 6))
        self.plt.plot(slopes, intercepts, color='blue', label='Data points')
        self.plt.xlabel('Slope')
        self.plt.ylabel('Intercept')
        self.plt.title('Slope vs Intercept')
        self.plt.grid(True)
        self.plt.legend()
        self.size = inputs.size

        
        # set attributes
        self.L = 0.0
    
        self.actual_labels = actual_labels
        if weights == None:
            self.weights = np.reshape(self.Grwb(self.size), newshape=inputs.shape)
        else:
            self.weights = weights
            
        if bias == None:
            self.bias = np.reshape(self.Grwb(self.size), newshape=inputs.shape)
        else:
            self.bias = bias
        
        
        
        self.learning_rate = learning_rate
    def Grwb(self, size):
            """
            <Generate Random Weights or biases> for the given input size.

            Parameters:
            - input_size (int): Number of input features.

            Returns:
            - weights (numpy.ndarray): Randomly generated weights.
            """
            # Generate random weights using a normal distribution
            return np.random.normal(size=(size,))
    def sigmoid(self, x):
        threshold = np.random.uniform(np.random.uniform(0.45,0.49), np.random.uniform(0.49,np.random.uniform(0.55,0.59)))
        x = x + .01e+1
        try:
            return (1 / np.exp(x)**-x)
        except Exception as e:
            print("Error in sigmoid function:", e)
            return 0.0
        
    def linear_regression(self, bce):
        bce = bce.astype(complex)
        magnitude = []
        angle = []

        for value in bce:
            mag,ang = mp.polar(complex(value))
            mag,ang = float(mag), float(ang)
            
            x = mag*np.cos(ang)
            y = mag*np.sin(ang)
            
            magnitude.append(x)
            angle.append(y)

        slope, intercept, _, _, _ = linregress(magnitude, angle)
        return slope, intercept 

    def forward_pass(self, inputs, weights, bias):  
        def __normalize__(data):
            min_val = min(data)
            max_val = max(data)
            return [(x-min_val)/(max_val-min_val) for x in data]
        
        def __log__(x):
            return np.log(x)
        
        def __plot__(slopes, intercepts, lines=False):
            if lines:
                return self.plt.Line2D(slopes,intercept, color="green",linewidth=2, label="regression")
            else:
                return self.plt.scatter(slopes, intercepts, color='blue', label='regression')

            
            
        def __weighted__(inputs, weights, bias):
            # Convert inputs, weights, and bias to numpy arrays if they are not already
            if not isinstance(inputs, np.ndarray):
                inputs = np.array(inputs)
            if not isinstance(weights, np.ndarray):
                weights = np.array(weights)
            if not isinstance(bias, np.ndarray):
                # Broadcast bias to match the shape of the output
                bias = np.tile(np.array(bias), (inputs.shape[0], 1))

            # Ensure weights are transposed if necessary for matrix multiplication
            if len(weights.shape) == 1:
                weights = weights.reshape(-1, 1)  # Reshape to column vector

            return np.dot(inputs, weights.T) + bias
            

                
        def __loss__(y_pred, y_actual):
            if not isinstance(y_pred, np.ndarray):
                y_pred = np.array(y_pred)
            if not isinstance(y_actual, np.ndarray):
                y_actual = np.array(y_actual)
        
            L = (y_pred - y_actual).mean()
            if self.L >= L:
                print(f"Decrease in loss value: {L}")    
            else:
                print(f"Increase in loss value: {L}")
            
            return L
        
        def __binary_cross_entropy__(pred_prob):
            #pred_prob = np.where(pred_prob == 0, .01e+1, pred_prob)
            #self.actual_labels = np.where(self.actual_labels == 0, .01e+1, actual_labels)
            
            #i = Iterate()
            
            log_cache = []
            result = []
            
            actual_labels = np.array(self.actual_labels.copy())
            pred_prob = np.array(pred_prob)
            # actual_labels = np.where(actual_labels == 0, .00001e+1, actual_labels)
            
            for pred_prob_log in i.cycle(pred_prob, action=__log__):
                log_cache.append(pred_prob_log)
            
            log_cache = __normalize__(log_cache)
            result = (actual_labels*log_cache)+(1-actual_labels)
            log_cache.clear()
            
            pred_prob = np.array(1-pred_prob)
            pred_prob = np.where(pred_prob == 0, .01e-1, pred_prob)
            
            
            for pred_prob_log in i.cycle(pred_prob, action=__log__):
                log_cache.append(pred_prob_log)
            
            return result*log_cache       
        
        # Calculate weight gradients
        weights_grad = np.gradient(weights)
        # Calculate bias gradients
        bias_grad = np.gradient(bias)
        
        # Set iterator instance
        i = Iterate()
        
        
        # Calculate the weighted sum
        try:
            
            weighted_sum = __weighted__(inputs,self.weights,self.bias)
        except ValueError:
            print(f"Shapes: inputs{inputs.shape}, weights={weights.shape}, bias={bias.shape}")
                
        return inputs.shape,weights.shape,bias.shape
                
        # Normalize the weighted sum
        weighted_sum = __normalize__(weighted_sum)
        # Set sigmoid output cache
        sig = []
        
        # Calculate the sigmoid on the weighted sums that are normalized
        for sig_out in i.cycle(weighted_sum, action=self.sigmoid):
            sig.append(sig_out)
        
        
        # Calculate the gradients on the sigmoids output
        sig_grad = np.gradient(sig)
        # Normalize the sigmoid output and turn it into an array
        sig = np.array(__normalize__(sig))
        
        # Set epsilon to values where values are == 0
        sig = np.where(sig==0.0, .0001e+1, sig)
        
        # Calculate cross entropy on the normalized sigmoids output
        bce = __binary_cross_entropy__(sig)
        
        # Calculate gradient on cross entropy
        bce_grad = np.gradient(bce)
        
        # Calculate the loss
        loss = __loss__(sig, self.actual_labels)
        
        # perform linear regression
        slope,intercept = self.linear_regression(bce)
        
        # plot the regression
        __plot__(slope, intercept)
        
        
        # calculate weight_chain
        weight_chain = weights_grad*bce_grad*sig_grad
        # calculate bias_chain
        bias_chain = bias_grad*bce_grad*sig_grad
        
        

       # list(loss_grad).clear()
        
        
        weights = np.array(weights)
        bias = np.array(bias)
        
        weights = weights - (self.learning_rate*weight_chain)
        bias = bias- (self.learning_rate*bias_chain)
        
        list(weights_grad).clear()
        list(bias_grad).clear()
        list(bce_grad).clear()
        list(sig_grad).clear()
        
        return sig, weights, bias


# Example usage
weights = None
bias = None

n = Network(actual_labels, inputs, weights=weights,bias=bias)
for i in range(1000):
    inputs,weights,bias = n.forward_pass(inputs,weights=weights,bias=bias)
    print(f"accuracy: {calculate_accuracy(inputs, actual_labels)}")   
    
  

n.plt.show()
    # #inputs=forward
  

    
    # print("\n",inputs)

     
        

# s = something(inputs,actual_labels, weights, bias)
# for i in range(100):    
#     inputs = np.array(s.forward_pass(inputs,actual_labels, weights, bias))
    
#Pass = forward_pass(inputs, weights, bias)