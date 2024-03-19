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
                
    
    def train(self, hidden_layers=1, epochs=1, learning_rate=0.001):
        for epoch in range(epochs):
            for hidden_layer in range(hidden_layers):
                neurons = self.neurons
                np.random.shuffle(self.labels)
                np.random.shuffle(self.weights)
                np.random.shuffle(self.biases)
                
                for feature_name, neurons in self.features.items():
                    input_layer = InputLayer(neurons, batch_size=self.batch_size)
                    for batched_inputs in input_layer.batch_inputs():
                        activators = Activations(batched_inputs)
                        sig_out_batch = []  # Store sigmoid outputs for the batch
                        for neuron, bias, weights in activators.Iter_neuron():
                            weighted_sum = np.dot(neuron, weights) + bias
                            sig_out, thresh = activators.Sigmoid(weighted_sum)
                            sig_out_batch.append(sig_out)
                    
                        intercept, slope = LinearRegression(self.labels, sig_out_batch)
                        loss = binary_cross_entropy(self.labels, sig_out_batch)
                        accu = np.mean(np.array(sig_out_batch) == np.array(self.labels))
                        self.intercept.append(intercept)
                        self.slope.append(slope)
                        self.neurons.append(sig_out_batch)
                        self.loss.append(loss)
                        self.accu.append(accu)
                        
                        print(f"""\n
                            \n_________________________\n
                              \nIntercept: {intercept}
                              \nSlope: {slope}
                              \nLoss: {loss}
                              \nAccuracy: {accu}\n
                            \n________________________\n
                              """)
                        
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
                        
                fig.write_html(f'plot_LinearRegression_{hidden_layer}.html')
    # def train(self, hidden_layers=25, epochs=3, learning_rate=0.001):
    #     batch_finished = False
    #     neuron_weighted_sums = []
    #     Thresh = 0.0
    
    #     for feature_name, neurons in self.features.items():
    #         input_layer = InputLayer(neurons, batch_size=self.batch_size)
    #         for batched_inputs in input_layer.batch_inputs():
                
    #             activators = Activations(batched_inputs)
    #             for neuron, bias, weights in activators.Iter_neuron():
                    
    #                 weighted_sum = np.dot(neuron, weights) + bias
                        
    #                 sig_out, thresh = activators.Sigmoid(weighted_sum, threshold=np.random.uniform(0.44,0.46))
    #                 Thresh += thresh
    #                 neuron_weighted_sums.append(weighted_sum)
                        
    #                 self.neurons.append(neuron)
    #                 self.weights.append(weights)
    #                 self.biases.append(bias)
    #                 self.sig_out.append(sig_out)
        
    #     print(self.sig_out)
        # Thresh /= len(self.neurons)
        
        # for epoch in range(epochs):
        #     for _ in range(hidden_layers):
        #         self.neurons = self.sig_out  # Update neurons with the last sig_out values
        #         activators = Activations(np.array(self.neurons))
        #         if not batch_finished:
        #             for neuron, bias, weights in activators.Iter_neuron():
        #                 weighted_sum = np.dot(neuron, weights) + bias            
        #                 sig_out, thresh = activators.Sigmoid(weighted_sum, threshold=np.random.uniform(0.44,0.46))
        #                 Thresh += thresh
                    
        #                 self.neurons.append(neuron)
        #                 self.weights.append(weights)
        #                 self.biases.append(bias)
        #                 neuron_weighted_sums.append(weighted_sum)
        #             batch_finished=True
        #         else:
        #             np.random.shuffle(self.weights)
        #             np.random.shuffle(self.biases)
                
        #             self.neurons.clear()  # Clear neurons from the previous iteration
        #             self.sig_out.clear()  # Clear sig_out from the previous iteration

        #             for neuron_weighted_sum in neuron_weighted_sums:
        #                 sig_out, thresh = activators.Sigmoid(neuron_weighted_sum, threshold=Thresh)
        #                 self.sig_out.append(sig_out)
        #                 self.neurons.append(neuron)  # Assuming activation is calculated here
        #             reg = LinearRegression(self.labels, self.neurons)
        #             print(reg)
        #             # Perform weight updates and other operations as needed

        #             # Reset neuron_weighted_sums for the next iteration
        #             neuron_weighted_sums.clear()

        #     batch_finished = True 
                
        
    # def train(self, hidden_layers=25, epochs=3, learning_rate=0.001):
    #     counter=0
    #     batch_finished = False
    #     for epoch in range(epochs):
    #         Thresh = 0.0
    #         neuron_weighted_sum = None  # Reset at the start of each epoch
            
    #         # Iterate over each feature
    #         for feature_name, neurons in self.features.items():
    #             input_layer = InputLayer(neurons, batch_size=self.batch_size)
                
    #             for _ in range(hidden_layers):
    #                 for batched_inputs in input_layer.batch_inputs():
    #                     activators = Activations(batched_inputs)
                    
    #                     for neuron, bias, weights in activators.Iter_neuron():
    #                         if not batch_finished:
    #                             neuron_weighted_sum = np.dot(neuron, weights) + bias 
    #                             sig_out, thresh = activators.Sigmoid(neuron_weighted_sum, threshold=np.random.uniform(0.40, 0.50))
    #                             Thresh += thresh
    #                             self.neurons.append(neuron)
    #                             self.weights.append(weights)
    #                             self.biases.append(bias)
    #                             self.sig_out.append(sig_out)
    #                         else:
    #                             pass
                                
    #                 self.sig_out, thresh = activators.Sigmoid(self.sig_out, threshold=np.random.uniform(0.44, 0.45))
                                
                                
    #                 batch_finished = True  # Update batch_finished flag after the first batch
    #                 self.sig_out = np.array(self.sig_out).astype(np.int64)
                    
    #                 self.labels = np.array(self.labels).astype(np.int64)
    #                 reg = LinearRegression(self.labels, self.sig_out)
                    
    #                 np.random.shuffle(self.labels)
                
    #                 # # Shuffle weights and biases after each epoch
    #                 # pseudo_weight_gradients = np.ones((len(self.weights), len(self.weights)))
    #                 # pseudo_bias_gradients = np.zeros(len(self.weights))
    #                 # self.weights=update_params(self.weights, pseudo_weight_gradients, 0.001)
    #                 # self.biases=update_params(self.biases, pseudo_bias_gradients, 0.001)
    #                 # np.random.shuffle(self.biases)
                    
                                
                            
                                
                            
                                
                                
    #                 # batch_finished = True
    #                 # print(len(self.neurons))
                       
                               
                    
    #                 Regression_Intercept, Slope = LinearRegression(self.labels, self.sig_out)
    #                 self.intercept.append(Regression_Intercept)
    #                 self.slope.append(Slope)
                        
                    
    #                 print(f"""\n
    #                           \n Intercept: {Regression_Intercept}
    #                           \n Slope: {Slope}
    #                     """)
    #                 # pseudo_weight_gradients = np.ones((len(self.weights), len(self.weights)))
    #                 # pseudo_bias_gradients = np.zeros(len(self.weights))
    #                 # self.weights=update_params(self.weights, pseudo_weight_gradients, 0.001)
    #                 # self.biases=update_params(self.biases, pseudo_bias_gradients, 0.001)
    #                 loss = binary_cross_entropy(self.labels, self.sig_out)
    #                 print("Loss:", loss)
    #                 # self.loss.append(loss)
    #                 # # sig = self.sig_out.copy()
    #                 # # neur = self.neurons.copy()
    #                 # # weight = self.weights.copy()
    #                 # # bias = self.biases.copy()
    #                 accu= np.mean(np.array(self.sig_out) == np.array(self.labels))
    #                 # # self.accu.append(accu)
    #                 print(f"accu: {accu}")
    #                 # # self.sig_out.clear()
    #                 # # self.neurons.clear()
    #                 # # self.biases.clear()
    #                 # # self.weights.clear()
            
            
            
            

                    
                        
                
if __name__ == "__main__":
    neural_network = NeuralNetwork("breast-cancer.csv", 569)
    output = neural_network.train(hidden_layers=5,epochs=5)                            
                        
        