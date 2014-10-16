import pylab as pl
import numpy as np


class ANN:

    #Initlialize
    def __init__(self, inputs, targets, nhidden1 = 0, nhidden2 = 0, nlayers = 1, momentum = 0, beta = 1):
        #use positive ones for bias node as opposed to book and position on the left of input
        #only works for input that is two dimensional because of shape(x)[1] looking for size of second dimension\
        #all variables are initialized, even ones that might not be used like the number of hidded nodes in a 
        #hidden layer
        self.inputs = np.concatenate((np.ones((np.shape(inputs)[0], 1)), inputs),axis=1)
        self.targets = targets
        #number of features plus bias
        self.feature_size = np.shape(self.inputs)[1]
        #number of ouputs
        self.output_size = np.shape(self.targets)[1]
        #hidden layer 1 size
        self.hidden_layer1_size = nhidden1
        #hidden layer 2 size
        self.hidden_layer2_size = nhidden2
        #number of hidden layers
        self.hidden_layer_count = nlayers
        #set momentum
        self.momentum = momentum
        #set beta term for logistic function
        self.beta = beta

        #initialize weight matrices
        if self.hidden_layer_count == 0:
            self.weights1 = (np.random.rand(self.feature_size,self.output_size)-0.5)*2/np.sqrt(self.feature_size)
            self.weights2 = []
            self.weights3 = []
        elif self.hidden_layer_count == 1:
            self.weights1 = (np.random.rand(self.feature_size,self.hidden_layer1_size)-0.5)*2/np.sqrt(self.feature_size)
            self.weights2 = (np.random.rand(self.hidden_layer1_size+1, self.output_size) - 0.5)* 2/np.sqrt(self.hidden_layer1_size)
            self.weights3 = []
        elif self.hidden_layer_count == 2:
            self.weights1 = (np.random.rand(self.feature_size, self.hidden_layer1_size)-0.5)*2/np.sqrt(self.feature_size)
            self.weights2 = (np.random.rand(self.hidden_layer1_size+1, self.hidden_layer2_size)-0.5)*2/np.sqrt(self.hidden_layer1_size)
            self.weights3 = (np.random.rand(self.hidden_layer2_size+1, self.output_size)-0.5)*2/np.sqrt(self.hidden_layer2_size)
        self.updatew1 = np.zeros((np.shape(self.weights1)))
        self.updatew2 = np.zeros((np.shape(self.weights2)))
        self.updatew3 = np.zeros((np.shape(self.weights3))) 

        self.train = []
        self.traint = []
        self.valid = []
        self.validt = []
        self.test = []
        self.testt = []


    #split data 50% train,25% valid/test
    def split_50_25_25(self):
        self.train = self.inputs[::2, :]
        self.traint = self.targets[::2]
        self.valid = self.inputs[1::4, :]
        self.validt = self.targets[1::4]
        self.test = self.inputs[3::4, :]
        self.testt = self.targets[3::4]

    #do a forward pass using the current weights using the provided data
    #if no data is provided then use the entire input data set
    #in the future it'd be nice to have more activation functions than logistic
    def forward_pass(self, input_data='none'):
        if input_data == 'none':
            input_data = self.inputs
        self.hidden1 = []
        self.hidden2 = []  
        if self.hidden_layer_count == 0:
            self.outputs = np.dot(input_data, self.weights1)
        elif self.hidden_layer_count == 1:
            self.hidden1 = np.dot(input_data, self.weights1)            
            self.hidden1 = 1.0/(1.0+np.exp(-self.beta*self.hidden1))
            self.hidden1 = np.concatenate((np.ones((np.shape(self.hidden1)[0],1)),self.hidden1),axis=1)            
            self.outputs = np.dot(self.hidden1, self.weights2)
        elif self.hidden_layer_count == 2:
            self.hidden1 = np.dot(input_data, self.weights1)
            self.hidden1 = 1.0/(1.0+np.exp(-self.beta*self.hidden1))
            self.hidden1 = np.concatenate((np.ones((np.shape(self.hidden1)[0],1)),self.hidden1),axis=1)
            self.hidden2 = np.dot(self.hidden1, self.weights2)
            self.hidden2 = 1.0/(1.0+np.exp(-self.beta*self.hidden2))
            self.hidden2 = np.concatenate((np.ones((np.shape(self.hidden2)[0],1)),self.hidden2),axis=1)
            self.outputs = np.dot(self.hidden2,self.weights3)
            self.output =  1.0/(1.0+np.exp(-self.beta*self.outputs))       
        return 1.0/(1.0+np.exp(-self.beta*self.outputs))

    #train for n iterations
    def train_n_iterations(self, iterations, learning_rate, plot_errors = False):

        #if no splitting was done then use entire input and target for training
        if self.train == []:
            self.train = self.inputs
            self.traint = self.targets
        #if plotting error over time initialize array
        if plot_errors == True:
            points = []
        for i in range(iterations):
            #if plottting error calculate error for validation set 
            #since we will calculate error for training anyways to perform training
            if plot_errors == True:
                self.outputs = self.forward_pass(self.valid)
                valid_error = 0.5*np.sum((self.outputs-self.validt)**2)
            self.outputs = self.forward_pass(self.train)
            train_error = 0.5*np.sum((self.outputs-self.traint)**2)

            #if plotting append errors to array for plotting later
            if plot_errors == True:
                points.append([train_error, valid_error])
            #print error every 100 iterations
            #make this user defined amount later
            if (np.mod(i,100)==0):
                print "Iteration: ",i, " Error: ",train_error    
            #calculate error based on logistic
            #add other activation functions later
            deltao = self.beta*(self.outputs-self.traint)*self.outputs*(1.0-self.outputs)
            #calculate errors depending on amount of hidden layers
            if self.hidden_layer_count == 0:
                self.updatew1 = learning_rate*(np.dot(np.transpose(self.train),deltao)) + self.momentum*self.updatew1
                self.weights1 -= self.updatew1
            if self.hidden_layer_count == 1:
                deltah1 = self.hidden1*self.beta*(1.0-self.hidden1)*(np.dot(deltao,np.transpose(self.weights2)))
                self.updatew1 = learning_rate*(np.dot(np.transpose(self.train),deltah1[:,1:])) + self.momentum*self.updatew1
                self.updatew2 = learning_rate*(np.dot(np.transpose(self.hidden1),deltao)) + self.momentum*self.updatew2
                self.weights1 -= self.updatew1
                self.weights2 -= self.updatew2
            elif self.hidden_layer_count == 2:
                deltah2 = self.hidden2*self.beta*(1.0-self.hidden2)*(np.dot(deltao,np.transpose(self.weights3)))
                deltah1 = self.hidden1*self.beta*(1.0-self.hidden1)*(np.dot(deltah2[:,1:],np.transpose(self.weights2)))
                self.updatew1 = learning_rate*(np.dot(np.transpose(self.train),deltah1[:,1:])) + self.momentum*self.updatew1
                self.updatew2 = learning_rate*(np.dot(np.transpose(self.hidden1),deltah2[:,1:])) + self.momentum*self.updatew2
                self.updatew3 = learning_rate*(np.dot(np.transpose(self.hidden2),deltao)) + self.momentum*self.updatew3
                self.weights1 -= self.updatew1
                self.weights2 -= self.updatew2
                self.weights3 -= self.updatew3    
        #if plotting then plot :)       
        if plot_errors == True:
            train_plot = [i[0] for i in points]
            valid_plot = [i[1] for i in points]
            pl.plot(train_plot, label = "train")
            pl.plot(valid_plot, label = "valid")
            pl.legend()
            pl.show()
    #same thing as train_n_iterations except we have an extra loop to make it sequential
    #add randomization for data set after each epoch
    def train_n_iterations_seq(self, iterations, learning_rate, plot_errors = False):

        #add case for no splitting
        if self.train == []:
            self.train = self.inputs

        if plot_errors == True:
            points = []

        for i in range(iterations):
            self.outputs = self.forward_pass(self.valid)
            valid_error = 0.5*np.sum((self.outputs-self.validt)**2)
            self.outputs = self.forward_pass(self.train)
            train_error = 0.5*np.sum((self.outputs-self.traint)**2)

            if plot_errors == True:
                    points.append([train_error, valid_error])
            if (np.mod(i,100)==0):
                print "Iteration: ",i, " Error: ",train_error    
            #sequential loop
            for j in range(np.shape(self.train)[0]):
                self.outputs = self.forward_pass(self.train[j,:]*np.ones((1,self.feature_size)))

                #use jth term
                deltao = self.beta*(self.outputs-self.traint[j])*self.outputs*(1.0-self.outputs)
                if self.hidden_layer_count == 0:
                    self.updatew1 = learning_rate*(np.dot(np.transpose(self.train),deltao)) + self.momentum*self.updatew1
                    self.weights1 -= self.updatew1
                if self.hidden_layer_count == 1:
                    #replace train with train[j]
                    deltah1 = self.hidden1*self.beta*(1.0-self.hidden1)*(np.dot(deltao,np.transpose(self.weights2)))
                    self.updatew1 = learning_rate*(np.dot(np.transpose((self.train[j,:]*np.ones((1,self.feature_size)))),deltah1[:,1:])) + self.momentum*self.updatew1
                    self.updatew2 = learning_rate*(np.dot(np.transpose(self.hidden1),deltao)) + self.momentum*self.updatew2
                    self.weights1 -= self.updatew1
                    self.weights2 -= self.updatew2
                elif self.hidden_layer_count == 2:
                    deltah2 = self.hidden2*self.beta*(1.0-self.hidden2)*(np.dot(deltao,np.transpose(self.weights3)))
                    deltah1 = self.hidden1*self.beta*(1.0-self.hidden1)*(np.dot(deltah2,np.transpose(self.weights2)))
                    self.updatew1 = learning_rate*(np.dot(np.transpose(self.train),deltah1[:,1:])) + self.momentum*self.updatew1
                    self.updatew2 = learning_rate*(np.dot(np.transpose(self.hidden1),deltah2[:,1:])) + self.momentum*self.updatew2
                    self.updatew3 = learning_rate*(np.dot(np.transpose(self.hidden2),deltao)) + self.momentum*self.updatew3
                    self.weights1 -= self.updatew1
                    self.weights2 -= self.updatew2
                    self.weights3 -= self.updatew3           
        if plot_errors == True:
            train_plot = [i[0] for i in points]
            valid_plot = [i[1] for i in points]
            pl.plot(train_plot, label = "train")
            pl.plot(valid_plot, label = "valid")
            pl.legend()
            pl.show()
    #this code is almost directly copied from book with a fix for the axes
    def confmat(self,inputs='none',targets='none', print_info = True):
        if inputs == 'none':
            inputs=self.valid
        if targets == 'none':
            targets=self.validt
        nclasses = self.output_size        
        outputs = self.forward_pass(inputs)
        if nclasses==1:
            nclasses = 2
            outputs = np.where(outputs>0.5,1,0)
        else:
            # 1-of-N encoding
            outputs = np.argmax(outputs,1)
            targets = np.argmax(targets,1)
        cm = np.zeros((nclasses,nclasses))
        for i in range(nclasses):
            for j in range(nclasses):
                cm[i,j] = np.sum(np.where(targets==i,1,0)*np.where(outputs==j,1,0))

        if print_info == True:
            print "Confusion matrix is:"
            print cm
            print "Percentage Correct: ",np.trace(cm)/np.sum(cm)*100
        return np.trace(cm)/np.sum(cm)*100

