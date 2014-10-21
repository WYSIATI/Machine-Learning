


def kfold_xvalid(g_input, g_target, k):

	import rbf

	for fold in range(k):
		valid_input = g_input[fold]
		valid_target = g_target[fold]		
	
		train_input = np.conatenate((g_input[:fold],g_input[fold+1:]), axis=0)
		train_target = np.conatenate(g_target[:fold],g_target[fold+1:], axis=0)

		# Train & Test Perceptron network
		#net = rbf.rbf(train,traint,5,1,1)
		#net.rbftrain(train,traint,0.25,2000)
		#conf += net.confmat(test,testt)

		# Train & Test MultiLayer Perceptron network
		#net = rbf.rbf(train,traint,5,1,1)
		#net.rbftrain(train,traint,0.25,2000)
		#conf += net.confmat(test,testt)

		# Train & Test Radial Basis Function network
		net = rbf.rbf(train,traint,5,1,1)
		net.rbftrain(train,traint,0.25,2000)
		conf += net.confmat(test,testt)

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
	conf[0,:] /= total_tgt0
	conf[1,:] /= total_tgt1
	conf[2,:] /= total_tgt2

	return conf