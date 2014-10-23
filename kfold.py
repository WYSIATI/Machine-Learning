import numpy as np
import pcn as pcn
import mlp_ML as mlp
import rbf
import svm

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
		# net = pcn.pcn(train_input,train_target)
		# net.pcntrain(train_input,train_target,0.25,1000)
		# conf += net.confmat(valid_input,valid_target)

		# Train & Test MultiLayer Perceptron network

		# net = mlp.mlp(train_input,train_input,2)
		# # Not doing early stopping for now
		# #net.earlystopping(train_input,train_input, valid_input, valid_target, 0.1, 200)
		# net.mlptrain(train_input, train_input, 0.1, 200)
		# conf += net.confmat(valid_input,valid_target)
		# print "CONFUSION\n", conf

		# Train & Test Radial Basis Function network
		# net = rbf.rbf(train_input,train_target,5,1,1)
		# net.rbftrain(train_input,train_target,0.25,500)
		# conf += net.confmat(valid_input,valid_target)

		# Train & Test Support Vector Machine
		# Learn the full data
		output = np.zeros((2, 3))
		svm0 = svm.svm(kernel='linear')
		svm0 = svm.svm(kernel='poly',C=0.1,degree=3)
		svm0 = svm.svm(kernel='rbf')
		svm0.train_svm(train_input,np.reshape(train_target[:,0],(np.shape(train_input[:,:2])[0],1)))
		valid_input = np.vstack((valid_input, valid_input))
		output[:,0] = svm0.classifier(valid_input,soft=True).T

		#svm1 = svm.svm(kernel='linear')
		#svm1 = svm.svm(kernel='poly',C=0.1,degree=3)
		svm1 = svm.svm(kernel='rbf')
		svm1.train_svm(train_input,np.reshape(train_target[:,1],(np.shape(train_input[:,:2])[0],1)))
		output[:,1] = svm1.classifier(valid_input,soft=True).T

		#svm2 = svm.svm(kernel='linear')
		#svm2 = svm.svm(kernel='poly',C=0.1,degree=3)
		svm2 = svm.svm(kernel='rbf')
		svm2.train_svm(train_input,np.reshape(train_target[:,2],(np.shape(train_input[:,:2])[0],1)))
		output[:,2] = svm2.classifier(valid_input,soft=True).T

		# # Make a decision about which class
		# # Pick the one with the largest margin
		# bestclass = np.argmax(output,axis=1)
		# print bestclass
		# print iris[1::2,4]
		# err = np.where(bestclass!=iris[1::2,4])[0]
		# print err
		# print float(np.shape(testt)[0] - len(err))/ (np.shape(testt)[0]) , "test accuracy"

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