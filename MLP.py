import numpy as np

#normalize data
#lots of code in this file is copied from the book
iris = np.loadtxt('iris_proc.dat',delimiter=',')
iris[:,:4] = iris[:,:4]-iris[:,:4].mean(axis=0)
imax = np.concatenate((iris.max(axis=0)*np.ones((1,5)),np.abs(iris.min(axis=0)*np.ones((1,5)))),axis=0).max(axis=0)
iris[:,:4] = iris[:,:4]/imax[:4]

# Set up target data as 1-of-N output encoding
target = np.zeros((np.shape(iris)[0],3));
indices = np.where(iris[:,4]==0) 
target[indices,0] = 1
indices = np.where(iris[:,4]==1)
target[indices,1] = 1
indices = np.where(iris[:,4]==2)
target[indices,2] = 1

# Randomly order the data so that we get a division of classes when we split the data
order = range(np.shape(iris)[0])
np.random.shuffle(order)
iris = iris[order,:]
target = target[order,:]

#IMPORTANT
#SET DIRECTORY TO WHERE ANN.py IS
import sys
sys.path.insert(0, 'C:\Users\Alexander\Documents\GitHub\Machine-Learning-Course')#MODIFY ME
#READ ABOVE
#PLZ

#impot ANN
import ANN
#initialize perceptron
net = ANN.ANN(iris[:,:4],target,nhidden1 = 5,nlayers = 1,momentum = 0.9 )
#split the data we randomized and encoded the output for earlier
net.split_50_25_25()
#train for n iterations
#first parameter is number of iterations
#second parameter is the learning rate
#third parameter is a boolean of whether or not you want to track and plot the error during training
net.train_n_iterations(1000,0.3,plot_errors = True)
#print confusion matrix
net.confmat()

#repeat for seq training
net = ANN.ANN(iris[:,:4],target,nhidden1 = 5,nlayers = 1,momentum = 0.9 )
net.split_50_25_25()
net.train_n_iterations_seq(1000,0.1,plot_errors = True)
net.confmat()

