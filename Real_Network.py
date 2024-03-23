# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 23:42:41 2024

@author: hadaw
"""
import os
import pandas as pd
import numpy as np

# Change the working directory to data directory
os.chdir("D:/.WindowsAPI")


from Input import InputLayer
from Foreach import *





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
        print(self.features.size)
        self.labels = (self.data['diagnosis'].values == 'M').astype(int)
        self.max_batch_size = self.labels.size
    
      
    
    def train(self, hidden_layers=1, epochs=1, learning_rate=0.001):
        lab = self.labels.copy()
        batches = []
        
        
        for epoch in range(epochs):
            np.random.shuffle(self.labels)
            for hidden_layer in range(hidden_layers):
                #while len()
                if hidden_layer <= hidden_layers:   
                    
                    batch = InputLayer(lab, max_batch_size=self.max_batch_size)
                    forward_pass_data = batch.forward_pass()
                    batches.append(forward_pass_data["output"])
                else:
                    pass
                if hidden_layer >= hidden_layers:
                    for batch in batches:    
                        forward_pass_data = batch.forward_pass(inputs=batch, bias=forward_pass_data["bias"], weights=forward_pass_data["weights"])
                        accuracy = calculate_accuracy(self.labels, forward_pass_data["output"])
                        print(f"Accuracy: {accuracy}")
                forward_pass_data["weights"] = np.array(forward_pass_data["weights"])
                forward_pass_data["avg_chain_grad"] = np.array([forward_pass_data["avg_chain_grad"]])
                forward_pass_data["bias"] = np.array(forward_pass_data["bias"])
                    
                    
                    
                forward_pass_data["weights"] -= learning_rate * forward_pass_data["avg_chain_grad"]
                forward_pass_data["bias"] -= learning_rate * forward_pass_data["avg_chain_grad"]
                    
                forward_pass_data["weights"] = list(forward_pass_data["weights"])
                forward_pass_data["chain_grad"] = list(forward_pass_data["chain_grad"])
                forward_pass_data["bias"] = list(forward_pass_data["bias"])
            batch.clear_gradients()
            #np.random.shuffle(self.labels)
                    
                    
                    
                    
                    
                    # np.random.shuffle(forward_pass_data["weights"])
                    # np.random.shuffle(forward_pass_data["bias"])
                
                
            lab = forward_pass_data["output"]
                
            
            
            
            
            
        return self.labels, forward_pass_data
    
network = Network("breast-cancer.csv")
batch = network.train(hidden_layers=31, epochs=10000) 