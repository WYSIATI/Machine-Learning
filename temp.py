import numpy as np
import kfold

g_input_data = np.loadtxt('iris_proc.data', delimiter = ',')
order = range(np.shape(g_input_data)[0])
np.random.shuffle(order)
g_input_data = g_input_data[order, :]

g_input = g_input_data[:, :4]
g_target = g_input_data[:, 4:]


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
