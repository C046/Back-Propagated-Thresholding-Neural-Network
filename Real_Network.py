# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 23:42:41 2024

@author: hadaw
"""
import os
import pandas as pd
import numpy as np
from Input import InputLayer

# Change the working directory to data directory
os.chdir("D:/.WindowsAPI")



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




class Network:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.features = self.data.drop('diagnosis', axis=1)
        self.labels = (self.data['diagnosis'].values == 'M').astype(int)
        self.max_batch_size = self.labels.size
    
      
    
    def train(self, hidden_layers=1, epochs=1, learning_rate=0.001):
        lab = self.labels.copy()
        
        
        for epoch in range(epochs):
            for hidden_layer in range(hidden_layers):      
                if hidden_layer == 0:   
                    
                    batch = InputLayer(lab, max_batch_size=self.max_batch_size)
                    forward_pass_data = batch.forward_pass()
                else:
                    forward_pass_data = batch.forward_pass(inputs=forward_pass_data["output"], bias=forward_pass_data["bias"], weights=forward_pass_data["weights"])
                    np.random.shuffle(self.labels)
                    forward_pass_data["weights"] = np.array(forward_pass_data["weights"])-learning_rate
                    forward_pass_data["chain_grad"] = np.array(forward_pass_data["chain_grad"])
                    forward_pass_data["weights"] = list(forward_pass_data["weights"]*forward_pass_data["chain_grad"])
                    
                    
                    forward_pass_data["bias"] = np.array(forward_pass_data["bias"])-learning_rate
                    forward_pass_data["bias"] = list(forward_pass_data["bias"] * forward_pass_data["chain_grad"])
                    
                    
                    batch.clear_gradients()
                    
                    # np.random.shuffle(forward_pass_data["weights"])
                    # np.random.shuffle(forward_pass_data["bias"])
                
                
                lab = forward_pass_data["output"]
                
        
            accuracy = calculate_accuracy(self.labels, lab)
            print(f"Accuracy: {accuracy}")
            
        return lab, self.labels
    
network = Network("breast-cancer.csv")
batch = network.train(hidden_layers=100, epochs=1000) 