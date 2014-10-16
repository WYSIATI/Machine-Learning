from pylab import *
from numpy import *

data = loadtxt('numbers.dat',skiprows = 1)
#reshape inputs into #imagesx#pixels
inputs = reshape(data,(26,156))
targets = inputs

#IMPORTANT
#SET DIRECTORY TO WHERE ANN.py IS
import sys
sys.path.insert(0, '/Users/craig/Documents/Engineering/CSCI4155\ Machine\ Learning\ with\ Robotics/Alex\ Rudiuk\'s\ github\ \(HsoulT\)/')#MODIFY ME
#READ ABOVE
#PLZ

#use a constant to keep track of how many nodes we are using in the hidden layer
hidden_layer_nodes = 20
import ANN
#intialize
aa = ANN.ANN(inputs,targets, nhidden1 = hidden_layer_nodes,nlayers = 1,momentum = 0.9)
#train for n iterations
#first parameter is number of iterations
#second parameter is the learning rate
#third parameter is a boolean of whether or not you want to track and plot the error during training
aa.train_n_iterations(2000,0.1,plot_errors = False)
#get the ouputs using the inputs
#we are using all the inputs here because of no parameter is provided 
#forward pass defaults to all the data
results = aa.forward_pass()
#get rid of bias from weights
testWeight = aa.weights1[1:,:]
#transpose so it is positioned same as input
testWeight = transpose(testWeight)
#reshape 
testWeight = reshape(testWeight,(hidden_layer_nodes,12,13))
#plot weights
figure(0)
for i in range(20):
    #change the size of how how many suplots there are to fit 
    #how many weights there are
    #Ex: for 20 weights, 4x5 is good
    subplot(4,5,1+i); imshow(testWeight[i],'gray')
suptitle('Weights', fontsize = 18)

#plots output
results = reshape(results,(26,12,13))
figure(1)
for i in range(26):
    subplot(5,6,1+i); imshow(results[i],'gray')
suptitle('Outputs', fontsize = 18)


#plot inputs
inputs = reshape(inputs,(26,12,13))
figure(2)
for i in range(26):
    subplot(5,6,1+i); imshow(inputs[i],'gray')
suptitle('Inputs', fontsize = 18)

show()


######################

#reshape inputs for upcoming calculations
inputs = reshape(inputs,(26,156))
#check how the perceptron works on cleaning up noisy data
#make noisy data
noise_lvl = 0.2
flipped = random.random(inputs.shape) < noise_lvl
badinputs = inputs.copy()
badinputs[flipped] = 1 - badinputs[flipped]

#get output
#this time we need to add ones to the beginning since we are using a new data set that
#the perceptron has not already processed
results = aa.forward_pass(concatenate((ones((shape(badinputs)[0],1)),badinputs),axis=1))

#plot after reshaping
results = reshape(results,(26,12,13))
figure(0)
for i in range(26):
    subplot(5,6,1+i); imshow(results[i],'gray')
suptitle('Results', fontsize = 18)


badinputs = reshape(badinputs,(26,12,13))
figure(1)
for i in range(26):
    subplot(5,6,1+i); imshow(badinputs[i],'gray')
suptitle('Bad inputs', fontsize = 18)


show()
