import numpy as np

import plotly.express as px
neurons = np.array([1, 2, 3, 4])
import mpmath as mp
import plotly.io as pio

from Foreach import *
class Activations:
    def __init__(self, input_array, e=False):
        self.input_array = input_array
        self.input_size = self.input_array.size
        
        self.biases = self.Grwb(size=self.input_size)
        self.weights = self.Grwb(size=self.input_size)
        
        self.e = 2.71828
        self.epsilon = 1e-15
    
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

    def iter_neuron(self, inputs=False, bias=False, weights=False):
        if inputs != False:
            self.input_array = inputs
        
        if bias != False:
            self.biases = bias
        
        if weights != False:
            self.weights = weights
            
        try:    
            for element, bias, weights in zip(self.input_array, self.biases, self.weights):
                yield (element, bias, weights)
        
        except StopIteration:
            pass
    def mpwhere(self, condition, x, y):
        result = []
        for c, xi, yi in zip(condition, x, y):
            result.append(xi if c else yi)
        return result
    
    def Sigmoid(self, x, threshold=np.random.uniform(0.43,0.46), gradient=False):
        # If x is a single value, convert it to an array
        if not isinstance(x, np.ndarray):
            x = np.array([x])
        
        if x.size > 1:
            x = x/np.max(np.abs(x))
        else:
            x = x/np.abs(x)
            
        
        x_exp = np.exp(-x)
        bottom = 1+x_exp
        
        res = 1/bottom
        
        if gradient:
            pass
        else:    
            res = np.where(res >= threshold, 1,0)
        
        if res.size <= 1:
            return res[0], threshold
        
        return res,threshold
    
    def Sigmoid_Gradient(self, x ):
        if not isinstance(x, np.ndarray):
            x = np.array([x])+self.epsilon
        
        
        
        x_sig, thresh = self.Sigmoid(x, gradient=True)
        x_sig = x_sig*(1-x_sig)
        
        
            
        return x_sig
        
        # for val in x:
        #     x_exp = np.exp(val)
        #     bottom = 1+x_exp
        #     res = 1/bottom
        #     result.append(float(res))
        
        # return result
    
        # for val in x:
        #     # Adjust the threshold if necessary
        #     if val >= 2:
        #         val = 2  # Cap the value at 2 or scale it as needed
            
        #     # Compute the sigmoid for each value
        #     x_exp = mp.exp(val)
        #     bottom = 1 + x_exp
        #     res = 1 / bottom
        #     result.append(float(res))
    
        # # Return the result and threshold
        # return result, threshold
        
        # x_exp = mp.exp(-np.mean(-x))
        # bottom = (1+x_exp) + epsilon
        
        # return (1/bottom), threshold
        
        
        
        
        # # Scale and normalize the input values
        # #max_value = np.max(np.abs(x))
        # # normalization_factor = 10 ** np.ceil(np.log10(max_value))
        # # x_normalized = x / normalization_factor
    
        # # Compute sigmoid for each normalized value
        # for value in x:
        #     value = -value
        #     Value = float(mp.exp(value))
        #     bottom = (1 + Value) + epsilon
        #     res = 1 / bottom
        #     result.append(res)
        
        # return result, threshold    

        # res = []
        # if not isinstance(x, (list, tuple, np.ndarray)):
        #     # If x is not iterable, compute sigmoid directly
        #     x = mp.mpf(str(x))
        #     exp = mp.exp(-x)
        #     bottom = 1 + exp
        #     result = 1 / bottom+epsilon
            
        # else:
            
        #     for value in x:
        #         value = np.array(value, dtype=np.float64)  # Ensure value is a NumPy array
        #         value_mp = mp.mpf(str(value))  # Convert to mpmath's arbitrary precision float
        #         exp = mp.exp(-value_mp)
        #         bottom = 1 + exp
        #         result = 1 / bottom
        #         res.append(result)

        # return res, threshold
    #    return res, threshold  # Return the result and threshold value
        #     try:
                
        #         exp = mp.exp(x)
        #     except Exception as E:
        #         print(E)
        #     bottom = foreach(1, exp, action=add_value)
        #     result = foreach(1, bottom, action=divide_value)

        # # Apply threshold if necessary
        # result = np.where(result >= threshold, 1.0, 0.0)
        # return result, threshold



    
    
    def Sigmoid_Derivative(self, sigmoid_output):
        return np.dot(sigmoid_output, (1-sigmoid_output)) + self.epsilon 
   
    
            
    def update_weights(self, gradients, learning_rate):
        # Perform the weight updates here based on your optimization algorithm
        # Example: Simple gradient descent update
        mean_gradient = np.mean(gradients, axis=0)
        self.weights -= learning_rate * mean_gradient.reshape(self.weights.shape)


    def plot_sigmoid_derivative(self, inputs, derivative_values):
        # Adjust the range accordingly
        x_values = inputs
    
        # Create a scatter plot
        fig = px.line(x=x_values, y=derivative_values, labels={'x': 'Input', 'y': 'Derivative Value'},
                         title='Derivative of Sigmoid Function', template='plotly')
                        
        # Show the plot
        fig.show()

        # Save the HTML representation of the plot to a file
        with open("plot.html", "w", encoding="utf-8") as file:
            plot_html = str(pio.to_html(fig, full_html=False))
            file.write(plot_html)
        