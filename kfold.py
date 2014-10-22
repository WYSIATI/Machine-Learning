import numpy as np
import rbf

def kfold_xvalid(g_input, g_target, k):

	conf = np.zeros((3, 3))
	for fold in range(k):
#	for fold in range(1):
		print fold
		valid_input = g_input[fold]
		valid_target = g_target[fold]		
		#print valid_target	
		train_input = np.concatenate((g_input[:fold],g_input[fold+1:]), axis=0)
		train_target = np.concatenate((g_target[:fold],g_target[fold+1:]), axis=0)
		# print("Valid Inputs")
		# print(valid_input)
		# print("Valid Targets")
		# print(valid_target)

		# print("Train Inputs")
		# print(train_input)
		# print("Train Targets")
		# print(train_target)

		# Train & Test Perceptron network
		#net = rbf.rbf(train,traint,5,1,1)
		#net.rbftrain(train,traint,0.25,2000)
		#conf += net.confmat(test,testt)

		# Train & Test MultiLayer Perceptron network
		#net = rbf.rbf(train,traint,5,1,1)
		#net.rbftrain(train,traint,0.25,2000)
		#conf += net.confmat(test,testt)

		# Train & Test Radial Basis Function network
		net = rbf.rbf(train_input,train_target,5,1,1)
		net.rbftrain(train_input,train_target,0.25,1000)


		conf += net.confmat(valid_input,valid_target)

		# Train & Test Support Vector Machine
		#net = rbf.rbf(train,traint,5,1,1)
		#net.rbftrain(train,traint,0.25,2000)
		#conf += net.confmat(test,testt)

		# Train & Test Decision Tree
		#net = rbf.rbf(train,traint,5,1,1)
		#net.rbftrain(train,traint,0.25,2000)
		#conf += net.confmat(test,testt)


	# Get totals for each target
	total_tgt0 = sum(g_target==0)
	total_tgt1 = sum(g_target==1)
	total_tgt2 = sum(g_target==2)

	# Get percentages for confusion matrix
	# conf[0,:] /= total_tgt0
	# conf[1,:] /= total_tgt1
	# conf[2,:] /= total_tgt2
	print("Accuracy:")
	print np.trace(conf)/np.sum(conf)
	return conf