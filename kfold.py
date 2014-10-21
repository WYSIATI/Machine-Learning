def kfold_xvalid(g_input, g_target, k):
	
	for fold in range(k):
		valid_input = g_input[fold]
		valid_target = g_target[fold]		
	
		train_input = np.conatenate((g_input[:fold],g_input[fold+1:]) axis=0)
		train_target = np.conatenate(g_target[:fold],g_target[fold+1:] axis=0)

		




















	return outputs