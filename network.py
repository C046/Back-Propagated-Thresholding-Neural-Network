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


import torch as tt
#from NN.OrganizedNeuralNetwork import *
class NeuralNetwork:
    def __init__(self, data_path, batch_size):
        
        # Load dataset and initialize attributes
        self.data = pd.read_csv(data_path)
        self.features = self.data.drop('diagnosis', axis=1)
        self.labels = (self.data['diagnosis'].values == 'M').astype(int)
        self.batch_size = batch_size
        self.slope, self.intercept, self.neurons,self.weights,self.biases,self.sums,self.sig_out,self.sig_der,self.pred_labels,self.accu,self.loss,self.loss_der=([],[],[],[],[],[],[],[],[],[],[],[])

        self.neuron_data = {
            "Weights": self.weights,
            "Bias": self.biases,
        }
        
        
        self.activation_data = {
            "Sigmoid_Output":self.sig_out,
            "Sigmoid_Derivative":self.sig_der,
            "Predicted_Labels": self.pred_labels,
        }
        
        self.normal_data = {
            "Accuracy":self.accu,
            "Normal":self.loss,
            "Normal_Derivative":self.loss_der
        }
        


        self.epsilon = 1e-15
    # def propagation(self, output_layer_gradient, hidden_layers=1, epochs=1, learning_rate=0.001):
        
    #     for epoch in range(epochs):
    #         for hidden_layers in range(hidden_layers):
                
    

    
    def train(self, hidden_layers=25, epochs=3, learning_rate=0.001):
        counter=0
        batch_finished = False
        for epoch in range(epochs):
            Thresh = 0.0
            neuron_weighted_sum = None  # Reset at the start of each epoch
            
            # Iterate over each feature
            for feature_name, neurons in self.features.items():
                input_layer = InputLayer(neurons, batch_size=self.batch_size)
                
                for _ in range(hidden_layers):
                    for batched_inputs in input_layer.batch_inputs():
                        activators = Activations(batched_inputs)
                    
                        for neuron, bias, weights in activators.Iter_neuron():
                            if not batch_finished:
                                neuron_weighted_sum = np.dot(neuron, weights) + bias 
                                sig_out, thresh = activators.Sigmoid(neuron_weighted_sum, threshold=np.random.uniform(0.40, 0.50))
                                Thresh += thresh
                                self.neurons.append(neuron)
                                self.weights.append(weights)
                                self.biases.append(bias)
                                self.sig_out.append(sig_out)
                            else:
                                np.random.shuffle(self.weights)
                                np.random.shuffle(self.biases)
                                
                                self.neurons, thresh = activators.Sigmoid(self.sig_out, threshold=np.random.uniform(0.40, 0.50))
                                self.sig_out = self.neurons
                                
                    batch_finished = True  # Update batch_finished flag after the first batch
                    self.sig_out = np.array(self.sig_out).astype(np.int64)
                    self.labels = np.array(self.labels).astype(np.int64)
                    reg = LinearRegression(self.labels, self.sig_out)
                    
                    np.random.shuffle(self.labels)
                
                    # # Shuffle weights and biases after each epoch
                    # pseudo_weight_gradients = np.ones((len(self.weights), len(self.weights)))
                    # pseudo_bias_gradients = np.zeros(len(self.weights))
                    # self.weights=update_params(self.weights, pseudo_weight_gradients, 0.001)
                    # self.biases=update_params(self.biases, pseudo_bias_gradients, 0.001)
                    # np.random.shuffle(self.biases)
                    
                                
                            
                                
                            
                                
                                
                    # batch_finished = True
                    # print(len(self.neurons))
                       
                               
                    
                    Regression_Intercept, Slope = LinearRegression(self.labels, self.sig_out)
                    self.intercept.append(Regression_Intercept)
                    self.slope.append(Slope)
                        
                    
                    print(f"""\n
                              \n Intercept: {Regression_Intercept}
                              \n Slope: {Slope}
                        """)
                    # pseudo_weight_gradients = np.ones((len(self.weights), len(self.weights)))
                    # pseudo_bias_gradients = np.zeros(len(self.weights))
                    # self.weights=update_params(self.weights, pseudo_weight_gradients, 0.001)
                    # self.biases=update_params(self.biases, pseudo_bias_gradients, 0.001)
                    loss = binary_cross_entropy(self.labels, self.sig_out)
                    print("Loss:", loss)
                    # self.loss.append(loss)
                    # # sig = self.sig_out.copy()
                    # # neur = self.neurons.copy()
                    # # weight = self.weights.copy()
                    # # bias = self.biases.copy()
                    accu= np.mean(np.array(self.sig_out) == np.array(self.labels))
                    # # self.accu.append(accu)
                    print(f"accu: {accu}")
                    # # self.sig_out.clear()
                    # # self.neurons.clear()
                    # # self.biases.clear()
                    # # self.weights.clear()
            
            
            
            
            # fig = go.Figure(data=[go.Scatter3d(
            #     x= [i for i in range(len(self.accu))],
            #     y = self.accu,
            #     z = np.linspace(0,1, len(self.accu)),
            #     mode="lines",
            #     line=dict(
            #         color='red',
            #         width=2
            #     ),
            #     name="lines"
            # )])
                    
            # fig.update_layout(scene=dict(
            #     xaxis_title='index',
            #     yaxis_title='loss'
            # ))
                    
            # fig.write_html(f"loss_plot{epoch}.html")
            
                    
            # print(F"\n Epoch: {epoch}")
 
            # # Create a 3D scatter plot
            # fig = go.Figure(data=[go.Scatter3d(
            #     x=self.slope,
            #     y=self.intercept,
            #     z=np.linspace(0, 1, len(sig)),
            #     mode='markers',
            #     marker=dict(
            #         size=5,
            #         color='blue',                # set color to an array/list of desired values
            #         opacity=0.8
            #     ),
            #     name='Markers'
            # )])
        
            # # Add lines representing the data
            # fig.add_trace(go.Scatter3d(
            #     x=self.slope,
            #     y=self.intercept,
            #     z=np.linspace(0, 1, len(sig)),
            #     mode='lines',
            #     line=dict(
            #         color='red',               # set color to an array/list of desired values
            #         width=2
            #     ),
            #     name='Lines'
            # ))
        
            # # Update layout
            # fig.update_layout(scene=dict(
            #     xaxis_title='Intercept',
            #     yaxis_title='Slope',
            #     zaxis_title='Epochs'
            # ))
        
            # # Save plot as HTML file
            # fig.write_html(f'plot_epoch_{epoch}.html')
                        
                
if __name__ == "__main__":
    neural_network = NeuralNetwork("breast-cancer.csv", 569)
    output = neural_network.train()                            
                        
        