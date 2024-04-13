# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 07:29:52 2024

@author: hadaw
"""
import numpy as np
import inspect
import json
from scipy.stats import linregress
import matplotlib.pyplot as plt
import pandas as pd
import os

# set the working directory
os.chdir("D:/.WindowsAPI")

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

        if isinstance(b, (int, float,complex)):
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
    def __init__(self, actual_labels, inputs, weights=None, bias=None, learning_rate=0.00001):
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

    def __plot__(self, slopes, intercepts, lines=False):
        if lines:
            return self.plt.Line2D(slopes,intercepts, color="green",linewidth=2, label="regression")
        else:
            return self.plt.scatter(slopes, intercepts, color='blue', label='regression')
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


    def forward_pass(self, inputs, weights, bias, threshold=None, learn_rate=None):
        # Set iterator instance
        i = Iterate()

        def __sigmoid__(x):
            x_norm = __normalize__(x)
            x_exp = np.exp(x_norm)
            try:
                return 1/x_exp
            except Exception as E:
                print(E)

        def __sigmoid_diff__(x):
            sig_x = __sigmoid__(x)

            return sig_x*(1-sig_x)
        def __normalize__(data):
            min_val = np.min(data)
            max_val = np.max(data)

            if not isinstance(data, np.ndarray):
                data = np.array(data)
            else:
                pass

            bottom = (max_val-min_val)

            if bottom == 0:
                bottom = bottom+1e+1

            return ((data-min_val)/bottom).tolist()

        def __log__(x):
            x = np.array(__normalize__(x))
            x = np.where(x == 0, .001e+1, x)


            return np.log(x)



        def __weighted__(inputs, weights, bias):
            # normalize the inputs
            inputs = __normalize__(inputs)

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
            result = np.dot(inputs, weights.T) + bias.T
            return np.array(__normalize__(result))


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
            log_cache = []
            #result = []

            actual_labels = self.actual_labels
            if not isinstance(pred_prob, np.ndarray):
                pred_prob = np.array(pred_prob)

            for pred_prob_log in i.cycle(pred_prob, action=__log__):
                log_cache.append(pred_prob_log)


            result = (actual_labels*log_cache)+(1-actual_labels)
            log_cache.clear()

            pred_prob = 1-pred_prob

            for pred_prob_log in i.cycle(pred_prob, action=__log__):
                log_cache.append(pred_prob_log)

            result = result*log_cache
            log_cache.clear()

            return np.array(result)

        def __binary_cross_entropy_differentiated__(prediction):
            labels = self.actual_labels
            labels = np.where(labels==0, .01e-1, labels)
            prediction = np.where(prediction==0, .01e-1, prediction)

            return (labels/prediction)-((1+labels)/(1+prediction))



        # Calculate the weighted sum
        weighted_sum = __weighted__(inputs,weights,bias)


        # Calculate the sigmoid on the weighted sums that are normalized
        self.sig = __sigmoid__(weighted_sum)
        # Calculate the gradients on the sigmoid output
        self.sig_grad = __sigmoid_diff__(self.sig)

        # Propagate the sigmoids output
        self.sig -= (self.learning_rate*np.array(self.sig_grad))

        if threshold is None:

            # Initialize a random threshold to abide by, like a law for the brain, but laws change right?
            threshold = np.random.uniform(np.random.uniform(0.490,0.499), np.random.uniform(0.500,0.509))
        else:
            pass





        # Create a predicted output between a 1 or 0+epsilon
        self.predicted_output = np.where((np.array(self.sig) >= threshold), 1,0)
        # Calculate cross entropy on the normalized sigmoid output
        self.bce = __binary_cross_entropy__(self.sig)
        self.bce_grad = __binary_cross_entropy_differentiated__(self.bce)
        self.bce -= (self.learning_rate*np.array(self.bce_grad))

        threshold -= (self.learning_rate*np.array(self.bce))




        # if self.learning_rate <= 0.0001:
        #     print(True)
        #     self.learning_rate = learn_rate

        # self.learning_rate -= (np.array(self.learning_rate).mean()*np.array(self.bce))
        # self.learning_rate = self.learning_rate.mean()



        # Propagate the bce with the gradient
        #self.bce = (np.array(self.bce)+(self.learning_rate*np.array(self.bce_grad))).astype(np.int64)



        # return everything because, ... yeah bitch
        return self.sig, self.weights, self.bias, self.bce, self.bce_grad, self.predicted_output.T[:, 0], threshold, self.learning_rate


# Set weights to none to start with random weight values
weights = None
# set bias to none to start with random bias values
bias = None


# Set accuracy cache
accu=[]

# Create a network instance
n = Network(actual_labels, inputs, learning_rate=0.001)
class Model:
    def __init__(self, data=None):
        super().__init__()
        self.data = data

    def __store_data__(self, data, filepath="/Models", filename="default.json"):
        """
        Save data to a JSON file with the given filename in the specified filepath,
        overwriting it if it already exists.

        Parameters:
            - data: Dictionary containing the data to be saved.
            - filepath: Path to the directory where the file will be saved.
            - filename: Name of the JSON file to be saved.
        """
        # Ensure the directory exists, create it if not
        directory = os.path.join(filepath)
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Construct the full file path
        file_path = os.path.join(directory, filename)

        # Write data to a JSON file (overwriting if it already exists)
        with open(file_path, "w") as json_file:
            json.dump(data, json_file)

    def __load_data__(self, filepath="/Models", filename="default.json"):
        directory = os.path.join(filepath)
        if not os.path.exists(directory):
            os.makedirs(directory)

        filepath = os.path.join(directory, filename)

        with open(filepath, "r") as json_file:
            data = json.load(json_file)

        json_file.close()

        return data

# This really did not start out as a training function, but here it is.
    def train(self, inputs, epochs=1, num_hidden=1):

        # def __plot__(slopes, intercepts, lines=False):
        #     if lines:
        #         return self.plt.Line2D(slopes,intercepts, color="green",linewidth=2, label="regression")
        #     else:
        #         return self.plt.scatter(slopes, intercepts, color='blue', label='regression')

        threshold = None
        learn_rate=None
        # Iterate through number of epochs
        for epoch in range(epochs):
            # Set accuracy cache
            accu = []
            # Set output cache
            outputs = []

            # Iterate through the hidden layers.
            for hidden in range(num_hidden):
                # During the iteration of hidden layers
                # Perform the forward pass on the inputs
                for values in inputs:
                    # Unzip the output, weights, bias, bce, and bce grad
                    sig_out, weights, bias, bce, bce_grad, predicted_output, threshold, learn_rate = n.forward_pass(values, weights=n.weights, bias=n.bias, threshold=threshold, learn_rate=learn_rate)

                    # Transpose the outputs
                    sig_out = sig_out.T
                    # Slice the bce(binary cross entropy) output
                    bce = bce.T[:, 0]
                    # Transpose bce(binary cross entropy) gradients
                    bce_grad = bce_grad.T

                    # Calculate the accuracy with the actual labels and binary cross entropy
                    accuracy = calculate_accuracy(actual_labels, predicted_output)
                    # Append the accuracy to a list
                    accu.append(accuracy)

                # Set the inputs to the sigmoids output
                inputs=sig_out

            # Check if the weights instance are numpy arrays
            if not isinstance(weights, np.ndarray):
                weights = np.array(weights)

            # Check if the bias instance are numpy arrays
            if not isinstance(bias, np.ndarray):
                bias = np.array(bias)


            # Average out the accuracy
            accu = np.array(accu).mean()

            # Update the weights and biases via backpropagation
            weights-=(n.learning_rate*bce_grad)
            n.weights = weights

            bias -= (n.learning_rate*bce_grad)
            n.bias = bias


            # Clear the gradients
            bce_grad = list(bce_grad)
            bce_grad.clear()
            bce_grad = np.array(bce_grad)


            #print(inputs)
            # Shuffle the inputs
            np.random.shuffle(inputs)

            slopes = []
            intercepts = []

            slopes.append(epoch)
            intercepts.append(accu)
            # Print the accuracy after each epoch
            print(f"Epoch-{epoch}-\nAccuracy: {accu}")
            if accu == 1.0:
                np.random.shuffle(actual_labels)

            n.__plot__(slopes=slopes,intercepts=intercepts)
            # Set the accu var back to a list for re-use
            accu = []

        self.data = json.dumps({
            "Sigmoid_out": sig_out.tolist(),
            "Weights": weights.tolist(),
            "Bias": bias.tolist(),
            "Binary_Cross_Entropy": bce.tolist(),
            "Binary_Cross_Entropy": bce_grad.tolist(),
            "Actual_Labels": actual_labels.tolist(),
            "Predicted_Output": predicted_output.tolist(),
            "Threshold": threshold.tolist(),
            "Learn_Rate": [learn_rate],
        })

        #pd.DataFrame(self.data)
        return self.__store_data__(self.data)

# start model instance
M = Model()
# train model
T = M.train(inputs,epochs=2, num_hidden=2)
data = M.__load_data__()
# plot the model
n.plt.show()


# when this model is done, train another one and lets see what we can do with two using chain rules of calculus.

# Experiment with backpropagating the learn rate.
# You can only do this via chain rule, if threshold has already been backpropagated


# learn_rate -= learn_rate*threshold_grad
