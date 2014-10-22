import numpy as np
import kfold

g_input_data = np.loadtxt('iris_proc.data', delimiter = ',')
order = range(np.shape(g_input_data)[0])
np.random.shuffle(order)
g_input_data = g_input_data[order, :]

# make a new blank array of size (inputs x classes)

g_target = np.zeros((np.shape(g_input_data)[0],3));

#where the 3rd column = 0, make a 1 in the 0th column of target
indices = np.where(g_input_data[:,4]==0) 
g_target[indices,0] = 1

#where the 3rd column = 1, make a 1 in the 1th column of target
indices = np.where(g_input_data[:,4]==1)
g_target[indices,1] = 1

#where the 3rd column = 2, make a 1 in the 2th column of target
indices = np.where(g_input_data[:,4]==2)
g_target[indices,2] = 1

g_input = g_input_data[:, :4]
#g_target = g_input_data[:, 4:]

def normalize_data():
	for index in range(4):
		temp_list = []
		imax = 0
		for ele in g_input:
			temp_list.append(ele[index])
			if ele[index] > imax:
				imax = ele[index]
		for inner_idx in range(len(temp_list)):
			temp_list[inner_idx] /= imax
			g_input[inner_idx][index] = temp_list[inner_idx]

k = 150
normalize_data()
conf = kfold.kfold_xvalid(g_input, g_target, k)

print conf
