# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 20:39:57 2024

@author: hadaw
"""

import os
os.chdir("D:/.WindowsAPI")
from Foreach import *
from Input import *
from Acts import *
import pandas as pd


class NeuralNetwork:
    def __init__(self, data_path, batch_size):
        
        # Load dataset and initialize attributes
        self.data = pd.read_csv(data_path)
        self.features = self.data.drop('diagnosis', axis=1)
        self.labels = (self.data['diagnosis'].values == 'M').astype(int)
        self.batch_size = batch_size
        self.slope, self.intercept,self.weights,self.biases,self.sums,self.accu,self.loss,self.loss_der=([],[],[],[],[],[],[],[])

        self.neuron_data = {
            "Weights": self.weights,
            "Bias": self.biases,
            "Slope": self.slope,
            "Intercept": self.intercept,
            "Loss": self.loss,
            "Loss_Gradient": self.loss_der,
        }
        self.epsilon = 1e-15

    def train(self, hidden_layers=1, epochs=1, learning_rate=0.001):
        batches_finished=False
        hidden_finished = False
        # iterate over epochs
        for epoch in range(epochs):
            # randomize weights, biases, and labels beginning each epoch
            np.random.shuffle(self.weights)
            np.random.shuffle(self.biases)
            np.random.shuffle(self.labels)
            
            # iterate over hidden layers
            for hidden_layer in range(hidden_layers):
                # iterate over neurons
                for _, neurons in self.features.items():
                    # Configure the batch size for the input layer
                    input_layer = InputLayer(neurons, batch_size=self.batch_size)
                    # iterate over batches
                    for batched_inputs in input_layer.batch_inputs():
                        # assign activation functions to inputs
                        activators = Activations(batched_inputs)
                        # store sigmoid outputs
                        sig_out_batch = []
                        
                        # within each batch iterate over the batch of neurons
                        for neuron, bias, weights in activators.Iter_neuron():
                            if batches_finished == True:
                                if not isinstance(self.weights, np.ndarray):
                                    self.weights = np.array(self.weights)
                                    self.biases = np.array(self.biases)
                                
                                self.weights = self.weights.flatten()
                                self.biases = self.biases.flatten()
                                
                                self.sums = np.dot(self.sums, self.weights) + self.biases
                                
                                sig_out_batch, thresh = activators.Sigmoid(self.sums)
                   
                                #sig_out_batch = np.reshape(sig_out_batch,newshape=(31,569))
                                for batches in sig_out_batch:
                                    intercept,slope = LinearRegression(self.labels, batches)
                                    bin_cross_entropy = binary_cross_entropy(self.labels, batches)
                                    
                                    self.loss_der = np.array([])
                                    self.loss_der = CrossEntropy_Gradient(self.labels, batches)
                                    
                                    self.weights = np.reshape(np.array(self.weights), newshape=(31,569))
                                    self.biases = np.reshape(np.array(self.biases), newshape=(31,569))
                                    for batched_weights in self.weights:
                                        batched_weights -= learning_rate * np.array(self.loss_der)
                                        
                                    for batched_biases in self.biases:
                                        batched_biases -= learning_rate * np.array(self.loss_der)
                                
                                accu = np.mean(np.array(sig_out_batch) == np.array(self.labels))
                                print(accu)
                                         
                            else:
                                weighted_sum = np.dot(neuron, weights) + bias
                                sig_out, thresh = activators.Sigmoid(weighted_sum)
                                sig_out_batch.append(sig_out)
                                self.sums.append(weighted_sum)
                                self.weights.append(weights)
                                self.biases.append(bias)
                                
                
                batches_finished = True
                
                
            
            
            #print(bin_cross_entropy)
                
            # Drop columns except 'diagnosis'
            psuedo = self.data.drop(columns=self.data.columns.difference(['diagnosis']))
            
            # Convert sums list to DataFrame
            self.sums = pd.DataFrame(np.reshape(self.sums, (-1, len(self.features.keys()))))

            # Concatenate psuedo and sums DataFrames, drop 'diagnosis' column
            self.sums = pd.concat([psuedo, self.sums], axis=1).drop('diagnosis', axis=1)

            # Assign sums DataFrame to features.items
            self.features.items = self.sums.items 
            
            
            
            hidden_finished = True
            
       

            # Plotting 3D graph
            fig = go.Figure(data=[go.Scatter3d(
                x=self.intercept,
                y=self.slope,
                z=np.linspace(0, 1, len(sig_out_batch)),  # Assuming sig_out_batch has same length
                mode="lines",
                line=dict(
                    color='red',
                    width=2
                ),
                name="lines"
            )])
                    
            fig.update_layout(scene=dict(
                xaxis_title='slope',
                yaxis_title='intercept',
                zaxis_title="Length_Of_Sigmoid"
            ))
                        
            fig.write_html(f'plot_LinearRegression_{epoch}.html')                        

            
if __name__ == "__main__":
    neural_network = NeuralNetwork("breast-cancer.csv", 569)
    output = neural_network.train(hidden_layers=5,epochs=5)                            
                        
        