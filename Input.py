# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 15:05:40 2024

@author: hadaw
"""
import numpy as np
from Acts import Activations
from Foreach import binary_cross_entropy, binary_entropy_gradient, loss_partial_derivative
from itertools import cycle 


class InputLayer(Activations):
    def __init__(self, input_array, max_batch_size=5):
        
        if isinstance(input_array, list):
            self.input_array = np.array(input_array)
        else:
            self.input_array=input_array
            
        self.input_size = self.input_array.size
        self.batch_size = self.calculate_batch_size(self.input_size, max_batch_size)

        
    def batch_inputs(self):
        # Check if the batch size is not evenly divisible
        if self.input_size % self.batch_size != 0:
            raise ValueError("Batch size must be evenly divisible by the input size.")
        
        # Iterate over batches and yield them
        for i in range(0, self.input_size, self.batch_size):
            batch_elements = self.input_array[i:i + self.batch_size]
            yield batch_elements

    def calculate_batch_size(self, total_samples, max_batch_size):
        """
            Calculate the batch size based on the total number of samples and a maximum batch size.
            Parameters:
                total_samples : int
                Total number of samples in the dataset.
                max_batch_size : int
                Maximum batch size allowed.
        
            Returns:
                batch_size : int
                Calculated batch size.
        """
        
        return min(total_samples, min(total_samples, max_batch_size))
    
    def clear_gradients(self):
        if not isinstance(self.data["sig_gradients"], np.ndarray):
            self.data["sig_gradients"].clear()
        else:
            self.data["sig_gradients"] = []
            
        if not isinstance(self.data["loss_gradients"], np.ndarray):
            self.data["loss_gradients"].clear()
        else:
            self.data["loss_gradients"] = []
            
        if not isinstance(self.data["chain_grad"], np.ndarray):
            self.data["chain_grad"].clear()
        else:
            self.data["chain_grad"] = []   
            
        if not isinstance(self.data["avg_chain_grad"], np.ndarray):
            self.data["avg_chain_grad"] = 0.0
        else:
            self.data["avg_chain_grad"] = 0.0   
    
    def forward_pass(self, inputs=False, bias=False, weights=False, threshold=None):
        # Set the threshold for the sigmoid function
        thresh = threshold if threshold is not None else np.random.uniform(0.43, 0.46)
        
        self.data = {
            "output": [],
            "sig_gradients": [],
            "loss": [],
            "loss_gradients": [],
            "loss_partial": [],
            "chain_grad": [],
            "weights": [],
            "bias": [],
            "avg_chain_grad": 0.0
        }

        # Iterate over batches
        for batched_inputs in self.batch_inputs():
            # Assign activation class to inputs
            activators = Activations(batched_inputs)
            
            # Iterate over the inputs in the batch
            for neuron, bias, weights in activators.iter_neuron(inputs=inputs,bias=bias,weights=weights):
                # Calculate the weighted sum
                neuron_weighted_sum = np.dot(neuron, weights) + bias
                # Calculate the sigmoid output and gradient
                sig_out, thresh = activators.Sigmoid(neuron_weighted_sum, threshold=thresh)
                # Append the output and gradient for this neuron to the batch lists
                self.data["output"].append(sig_out)
                self.data["sig_gradients"].append(activators.Sigmoid_Gradient(neuron_weighted_sum))
                self.data["weights"].append(weights)
                self.data["bias"].append(bias)
            
            for input_neuron, output_sig in zip(batched_inputs, cycle(self.data["output"])):
                self.data["loss"].append(binary_cross_entropy(input_neuron, output_sig))
                self.data["loss_gradients"].append(binary_entropy_gradient(input_neuron, output_sig))
                self.data["loss_partial"].append(loss_partial_derivative(input_neuron, output_sig))
                
            #for sig_grad, loss_grad in zip()
        for sig_grad, loss_grad, loss_partial in zip(self.data["sig_gradients"], self.data["loss_gradients"], self.data["loss_partial"]):
            self.data["chain_grad"].append(sig_grad*loss_grad*loss_partial)
        
        self.data["avg_chain_grad"]+=np.mean(self.data["chain_grad"])
        
        bias_weight=False
        
        
        
        return self.data
    
    

    
# true_labels = np.array([np.random.randint(0,2) for i in range(100)]) 
# batch = InputLayer(true_labels, max_batch_size=true_labels.size)
# data = batch.forward_pass()
   
