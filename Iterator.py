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
inputs = np.array(data.drop('diagnosis', axis=1))
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
        self.learning_rate = learning_rate
        self.inputs = inputs

        
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
        x = np.where((x == 0), .0001e+1, x)
  
        try:
            return (1 / np.exp(x)**-x)
        
        except Exception as e:
            print("Error in sigmoid function:", e)
            return 0.0
        
    def sigmoid_diff(self, x):
        sig_x = self.sigmoid(x)
        
        return sig_x*(1-sig_x)

        
    def mag_and_ang(self, weighted_sum):
        # Extract real and imaginary parts
        a = np.real(weighted_sum)
        b = np.imag(weighted_sum)
    
        # Compute magnitude
        magnitude = np.sqrt(a**2 + b**2)
    
        # Compute angle
        angle = np.arctan2(b, a)
    
        return magnitude, angle    
    def linear_regression(self, weighted_sum):
        # Perform linear regression on the weighted sum
        weighted_sum = np.reshape(weighted_sum, newshape=self.weights.shape)
        slope, intercept, _, _, _ = linregress(x=weighted_sum, y=self.actual_labels)
        return slope, intercept


    def forward_pass(self, inputs, weights, bias):
        # Set iterator instance
        i = Iterate()
        def __normalize__(data):
            min_val = np.min(data)
            max_val = np.max(data)
            
            if not isinstance(data, np.ndarray):
                data = np.array(data)
            else:
                pass
            
                
            return ((data-min_val)/(max_val-min_val)).tolist()
        
                
        def __log__(x):
            return np.log(x)
        
        def __plot__(slopes, intercepts, lines=False):
            if lines:
                return self.plt.Line2D(slopes,intercepts, color="green",linewidth=2, label="regression")
            else:
                return self.plt.scatter(slopes, intercepts, color='blue', label='regression')

            
            
        def __weighted__(inputs, weights, bias):
            # Convert inputs, weights, and bias to numpy arrays if they are not already
            if isinstance(inputs, pd.DataFrame):
                inputs = np.array(inputs)
            if not isinstance(weights, np.ndarray):
                weights = np.array(weights)
            if not isinstance(bias, np.ndarray):
                # Broadcast bias to match the shape of the output
                bias = np.tile(np.array(bias), (inputs.shape[0], 1))

            # Ensure weights are transposed if necessary for matrix multiplication
            if len(weights.shape) == 1:
                weights = weights.reshape(-1, 1)  # Reshape to column vector

            
            
            return np.dot(inputs, weights.T) + bias.T
            

                
        def __loss__(y_pred, y_actual):
            if not isinstance(y_pred, np.ndarray):
                y_pred = np.array(y_pred)
            if not isinstance(y_actual, np.ndarray):
                y_actual = np.array(y_actual)
            counter = 0
            L = (y_pred - y_actual).mean()
            if self.L >= L:
                counter+=1
                if counter == y_pred.size/2:    
                    print(f"Decrease in loss value: {L}")
                    counter=0
                else:
                    pass
            else:
                if counter == y_pred.size/2:
                    print(f"Increase in loss value: {L} ")
                    counter=0
                else:
                    pass
                
            
            return L
        
        def __loss_diff__(y_pred, y_actual):
            return y_pred*y_actual
        
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
            
            log_cache = __normalize__(log_cache)
            result = result*log_cache
            log_cache.clear()
            
            return result      
        
        def __binary_cross_entropy_differentiated__(prediction):
            labels = self.actual_labels.copy()
            labels = np.where(labels==0, .01e-1, labels)
            prediction = np.where(prediction==0, .01e-1, prediction)
            
        
            return (labels/prediction)-((1+labels)/(1+prediction))

      
        # Calculate bias gradients
        bias_grad = np.gradient(bias)

        # Calculate the weighted sum
        weighted_sum = __weighted__(inputs,weights,bias)
        # Normalize the weighted sum
        normalized_sum = np.array(__normalize__(weighted_sum))
        
        # Calculate the sigmoid on the weighted sums that are normalized
        self.sig = self.sigmoid(normalized_sum)
        # Normalize the sigmoids output
        self.sig = __normalize__(self.sig)
        # Initialize a random threshold to abide by, like a law for the brain, but laws change right?
        threshold = np.random.uniform(np.random.uniform(0.45,0.49), np.random.uniform(0.49,np.random.uniform(0.55,0.59)))
        # Calculate the gradients on the sigmoid output
        self.sig_grad = np.gradient(self.sig)
        # Propagate the sigmoids output
        self.sig = self.sig-(self.learning_rate*np.array(self.sig_grad))
        # Create a predicted output between a 1 or 0+epsilon
        self.predicted_output = np.where((np.array(self.sig) >= threshold), 1,.001e-1)
        # Calculate cross entropy on the normalized sigmoid output
        self.bce = __binary_cross_entropy__(self.predicted_output)
        self.bce = np.array(__normalize__(self.bce))
        self.bce_grad = __binary_cross_entropy_differentiated__(self.bce)
        
        # Propagate the bce with the gradient
        self.bce = np.array(self.bce)-(self.learning_rate*np.array(self.bce_grad))
        
        # Propagate the weights, not working at the moment but we will fix that later.
        self.weights = np.array(self.weights)-np.array((self.learning_rate*self.bce_grad))
        self.bias = np.array(self.bias)-np.array((self.learning_rate*self.bce_grad))
        
        
        return self.bce, self.weights, self.bias, self.sig, self.bce_grad

# Example usage
weights = None
bias = None
accu=[]

# Example usage
n = Network(actual_labels, inputs)
def train(inputs, epochs=1):
    for epoch in range(epochs):
        accu = []
        outputs = []  # Store outputs for each epoch
        for values in inputs:
            output, weights, bias, sig, loss = n.forward_pass(values, weights=n.weights, bias=n.bias)
            # Assuming the output of the network is a probability between 0 and 1
        
            accuracy = calculate_accuracy(actual_labels, output)
            accu.append(accuracy)
            outputs.append(output.mean())
        
         
        #return outputs[-1].T, inputs
        inputs = outputs[::-1]
        sig = sig[::-1]
        loss = loss[::-1]
        weights = weights[::-1]
        bias = bias[::-1]
        
        
        
        # Calculate mean accuracy for the epoch
        mean_accuracy = np.mean(accu)
        print(f"Epoch {epoch + 1}: Accuracy = {mean_accuracy}")
        accu = []
        
        #print(f"accuracy: {calculate_accuracy(inputs, actual_labels)}")   
        
  
t = train(inputs)
n.plt.show()
    # #inputs=forward
  

    
    # print("\n",inputs)

     
        

# s = something(inputs,actual_labels, weights, bias)
# for i in range(100):    
#     inputs = np.array(s.forward_pass(inputs,actual_labels, weights, bias))
    
#Pass = forward_pass(inputs, weights, bias)