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
from itertools import cycle
import plotly.graph_objects as go

class NeuralNetwork:
    def __init__(self, data_path, batch_size):
        
        # Load dataset and initialize attributes
        self.data = pd.read_csv(data_path)
        self.features = self.data.drop('diagnosis', axis=1)
        self.labels = (self.data['diagnosis'].values == 'M').astype(int)
        self.batch_size = batch_size
        self.neurons,self.slope, self.intercept,self.weights,self.biases,self.sums,self.sig_out, self.accu,self.loss,self.loss_grad, self.sig_grad=([],[],[], [],[],[],[],[],[],[],[])
        self.accu = 0.0
        self.neuron_data = {
            "Neurons": self.neurons,
            "True_Labels": self.labels,
            "weights": self.weights,
            "bias": self.biases,
            "Slope": self.slope,
            "Intercept": self.intercept,
            "Weighted_Sums": self.sums,
            "Sig_Out": self.sig_out,
            "Sig_Grad": self.sig_grad,
            "Loss": self.loss,
            "Loss_Gradient": self.loss_grad,
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
                                    self.weights = np.array(self.weights).flatten()
                                    self.biases = np.array(self.biases).flatten()
                                    self.sums = np.array(self.sums).flatten()
                                    self.sig_out = np.array(self.sig_out).flatten()
                                    self.sig_grad =np.array(self.sig_grad).flatten()
                                    self.intercept = np.array(self.intercept).flatten()
                                    self.slope = np.array(self.slope).flatten()
                                    self.labels = np.array(self.labels).flatten()
                                
                                
                                
                                
                                
                                
                                self.sums = np.dot(self.sums, self.weights) + self.biases
                                
                                
                                self.sig_out, thresh = activators.Sigmoid(self.sums)
                                
                                self.sig_grad = activators.Sigmoid_Gradient(self.sig_out)
                                
                                for label in self.labels:    
                                    self.intercept, self.slope = LinearRegression(label, self.sig_out)
                                    self.loss = binary_entropy_gradient(label, self.sig_out)
                                    self.loss_grad = binary_entropy_gradient(label, self.sig_out)
                                    if label == self.sig_out:
                                        self.accu+=1
                                
                                self.accu = self.accu / len(self.labels)
                                
                                
                                  
                                    
                                    
                                
                                
                        else:
                            if not isinstance(self.sums, list):
                                self.sums = list(self.sums)
                                self.sig_out = list([self.sig_out])
                                self.sig_grad = list([self.sig_grad])
                                #self.sig_grad = list(self.sig_grad)
                                self.weights = list(self.weights)
                                self.biases = list(self.biases)
                                self.loss = list([self.loss]) 
                                self.loss_grad = list([self.loss_grad])
                                self.intercept = list([self.intercept])
                                self.slope = list([self.slope])
                                
                            weighted_sums = np.dot(neuron, weights) + bias
                            sig_out, thresh = activators.Sigmoid(weighted_sums)
                            sigmoid_gradient = activators.Sigmoid_Gradient(sig_out)
                            
                            #intercept, slope = LinearRegression([i for i in range(len(self.labels))], self.labels)
                            
                            self.neurons.append(neuron)
                            self.sums.append(weighted_sums)
                            self.sig_out.append(sig_out)
                            self.sig_grad.append(sigmoid_gradient)
                            # self.intercept.append(intercept)
                            # self.slope.append(slope)
                            self.weights.append(weights)
                            self.biases.append(bias)
                            
                            
                            for label in self.labels:
                                
                                loss = binary_cross_entropy(label, sig_out)
                                loss_gradient = binary_entropy_gradient(label, sig_out)
                                intercept, slope = LinearRegression(label, sig_out)
                                
                                self.loss.append(loss)
                                self.loss_grad.append(loss_gradient)
                                self.intercept.append(intercept)
                                
                                self.slope.append(slope)

                batches_finished = True
                print(f"Batches finished: {batches_finished}")
                print(self.accu)
                self.loss.clear()
                self.loss_grad.clear()
                self.sig_grad.clear()
                self.slope.clear()
                self.neurons.clear()
                self.intercept.clear()
                self.slope.clear()
                
                
                
                # fig = go.Figure(data=[go.Scatter3d(
                #     x=self.loss_grad,
                #     y=self.loss,
                #     z=[i for i in range(len(self.loss))],
                #     mode="lines",
                #     line=dict(
                #         color="red",
                #         width=2,
                #     ),
                #     name="lines",
                # )])
                # fig.update_layout(scene=dict(
                #     xaxis_title='Intercept',
                #     yaxis_title="Slope",
                #     zaxis_title="Loss Gradient",
                # ))
                # fig.write_html(f'plot_slope_intercept_gradient.html')
                #return self.neuron_data
            
            
            #print(bin_cross_entropy)
            #print(accu)
            # Drop columns except 'diagnosis'
            psuedo = self.data.drop(columns=self.data.columns.difference(['diagnosis']))
            
            # Convert sums list to DataFrame
            self.sig_out = pd.DataFrame(np.reshape(self.sig_out, (-1, len(self.features.keys()))))

            # Concatenate psuedo and sums DataFrames, drop 'diagnosis' column
            self.sig_out = pd.concat([psuedo, self.sums], axis=1).drop('diagnosis', axis=1)

            # Assign sums DataFrame to features.items
            self.features.items = self.sig_out.items
            
            
            
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
                        
            fig.write_html(f'plot_LinearRegression_e{epoch}.html')                        

            
if __name__ == "__main__":
    neural_network = NeuralNetwork("breast-cancer.csv", 569)
    output = neural_network.train(hidden_layers=2,epochs=1)                            
                        
        