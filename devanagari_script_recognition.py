from __future__ import division
import pyttsx3
engine = pyttsx3.init()

print("Phase - (1/3) : Importing the modules...")
engine.say("The program has started. Importing the required modules...")
engine.runAndWait()

import os, sys
import numpy
import scipy.special

TRAINING_DATA = 'train.csv'
VALIDATION_DATA = 'valid.csv'

if not os.path.isfile(TRAINING_DATA):
    print('Training Data is not present in the current folder.. Quitting Program..')
    engine.say("Training Data is not present in the current folder.. Quitting Program..")
    engine.runAndWait()
    sys.exit()

if not os.path.isfile(VALIDATION_DATA):
    print('Validation Data is not present in the current folder.. Quitting Program..')
    engine.say("Validation Data is not present in the current folder.. Quitting Program..")
    engine.runAndWait()
    sys.exit()

class neuralNetwork:
    
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        #number of input nodes
        self.input_nodes = inputnodes
        #number of hidden nodes
        self.hidden_nodes = hiddennodes
        #number of output nodes
        self.output_nodes = outputnodes
        #learning rate
        self.lr = learningrate
        
        #random weight of links between input and hidden layer (in range of -0.5 to +0.5)
        self.weight_input_hidden = (numpy.random.rand(self.hidden_nodes, self.input_nodes) - 0.5)
        #random weight of links between hidden and output layer (in range of -0.5 to +0.5)
        self.weight_hidden_output = (numpy.random.rand(self.output_nodes, self.hidden_nodes) - 0.5)
        
        #sigmoid function
        self.activation_function = lambda x: scipy.special.expit(x) 
    
    def train(self, inputs_list, targets_list):
        
        #convert inputs and targets list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        
        #inputs to hidden layer
        hidden_inputs = numpy.dot(self.weight_input_hidden, inputs)
        #outputs from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        
        #inputs to output layer
        final_inputs = numpy.dot(self.weight_hidden_output, hidden_outputs)
        #outputs from output layer
        final_outputs = self.activation_function(final_inputs)
        
        #error between target value and observed value of the output layer
        output_errors = targets - final_outputs
        
        #error for the hidden layer via backpropagation
        hidden_errors = numpy.dot(self.weight_hidden_output.T, output_errors)
        
        #gradient descent to update the weights
        #update the weights of the links between hidden and output layer
        self.weight_hidden_output += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))
        
        #update the weights of the links between the input and hidden layer
        self.weight_input_hidden += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))
    
    def predict(self, inputs_list):
        
        #convert input list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        
        #inputs to hidden layer
        hidden_inputs = numpy.dot(self.weight_input_hidden, inputs)
        
        #outputs from the hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        
        #inputs to output layer
        final_inputs = numpy.dot(self.weight_hidden_output, hidden_outputs)
        
        #outputs from the output layer
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs
        
input_nodes = 1024
hidden_nodes = 500
output_nodes = 46
learning_rate = 0.05
# Number of epochs ( Number of times the model is to be trained again to more robustly adjust the weights )
n = 5

print("Phase - (2/3) : Creating the Neural Network with given parameters...")
engine.say("Creating the Neural Network with the specified number of nodes in each layer")
engine.runAndWait()

neural_network = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

print("Phase - (3/3) : Training and testing the created model with training dataset for specified number of epochs...")
engine.say("Loading the training data to train the Neural Network.")
engine.runAndWait()

# load the train data CSV file
training_data = open("train.csv", 'r')
training_list = training_data.readlines()
training_data.close()

engine.say("Training and testing the model for specified number of epochs. This may take some time...")
engine.runAndWait()

for epoch in range(1, n+1):
    for record in training_list:
        all_values = record.split(',')
        #prepreocess the pixels in order to scale them in between 0.01 to 1.00
        inputs = (numpy.asfarray(all_values[1:]) * 0.99 ) + 0.01
        
        #target labels. all values are 0.01 except the correct label which has a value of 0.99
        targets = numpy.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        
        #begin the training
        neural_network.train(inputs, targets)
    
    
    engine.say("Loading test data and testing the model for corresponding epoch...")
    engine.runAndWait()
    
    # load the test data CSV file
    test_data = open("valid.csv", 'r')
    test_list = test_data.readlines()
    test_data.close()
        
    scores = []
    for record in test_list:
        all_values = record.split(',')
        
        #the first value is the label
        correct_label = int(all_values[0])
        
        #all the others are the pixels (i.e inputs)
        inputs = (numpy.asfarray(all_values[1:]) * 0.99 ) + 0.01
        
        #use the trained network to predict the output based on the given input
        outputs = neural_network.predict(inputs)
        
        #the index of the highest value in output is the predicted label
        label = numpy.argmax(outputs)

        #if the correct label, and the predicted label are same
        if (label == correct_label):
            #append 1 to scores.
            scores.append(1)
        else:
            #otherwise, wrong prediction, append 0 to scores.
            scores.append(0)
            pass
        pass
    
    scores_array = numpy.asarray(scores)
    print ("    " + "epoch(" + str(epoch) + "/" + str(n) + ") --------> ", (scores_array.sum() / scores_array.size) * 100, "%")

print ("Final accuracy of the Model Predictions = ", (scores_array.sum() / scores_array.size) * 100, "%")
    
engine.say("Model render complete. The accuracies of the model are shown in the terminal. The program has now completed.")
engine.runAndWait()