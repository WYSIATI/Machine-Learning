"""
This is for CSCI4155 Assignment 5
"""

import scipy.io
import math

# Get data from a matlab file NewsGroup.mat.
data = scipy.io.loadmat('NewsGroup.mat')
train_data = data.get('TRAIN_DATA')
train_label = data.get('TRAIN_LABEL')
test_data = data.get('TEST_DATA')
test_label = data.get('TEST_LABEL')

words_in_class1 = []
words_in_class2 = []
for element in train_data:
	if element[0] < 481:
		words_in_class1.append(element)
	else:
		words_in_class2.append(element)

def cal_prob(classn):
	# Calculate probability for each word in each given classn.
	# Take a list classn as input, return a dictionary word_probabilities.
	word_probabilities = dict()
	for element in classn:
		word_id = element[1]
		if word_id in word_probabilities:
			word_probabilities[word_id] += element[2]
		else:
			word_probabilities[word_id] = element[2]
	length = len(classn)
	for element in word_probabilities:
		word_probabilities[element] = float(word_probabilities[element]) / length
		#word_probabilities[element] = float(word_probabilities[element])
	return word_probabilities

# Probabilities of class 1 and class 2.
#len_train_data = len(train_data)
#class1_prob = 480.0 / len_train_data
#class2_prob = float(len_train_data - 480) / len_train_data

# Calculate probability for each word in each given class.
# word_prob_class1 is a dictionary with each word in class 1 and its probability,
# word_prob_class2 is a dictionary with each word in class 2 and its probability.
word_prob_class1 = cal_prob(words_in_class1)
word_prob_class2 = cal_prob(words_in_class2)

def prob_of_class(classn):
	# Calculate probability of document for given classn,
	# take a list classn as input, use formula
	# P(x1, x2,...., xn | c) = P(x1|c)*P(x2|c)*.....*P(xn|c)
	# to calculate the probability. Return a dictionary probability.
	# It will use global variable test_data as test document.
	probability = 1
	document = []
	probabilities = []
	for element in test_data:
		if len(document) == 0:
			document.append(tuple(element))
		if element[0] == document[0][0]:
			document.append(tuple(element))
		else:
			temp_prob = prob_of_class_helper(document, classn)
			probabilities.append(temp_prob)
			document = []
	if len(document) > 0:
		probabilities.append(prob_of_class_helper(document, classn))
	return probabilities
	
def prob_of_class_helper(document, classn):
	# Helper function of prob_of_class.
	probability = 1.0
	for word in document:
		word_id = word[1]
		word_count = word[2]
		if word_id in classn:
			probability *= (classn[word_id]**word_count)
			#probability += math.log1[(classn[word_id]**word_count)
		else:
			probability *= ((1.0/len(classn))**word_count)
	return probability

# prob1 and prob2 are the probabilities of documents with given
# classes class 1 and class 2.
prob1 = prob_of_class(word_prob_class1)
prob2 = prob_of_class(word_prob_class2)

# Compare prob1 and prob2 to get predicted_label
predicted_label = []
for index in range(len(prob1)):
	if prob1[index] < prob2[index]:
		predicted_label.append([2])
	else:
		predicted_label.append([1])

accuracy = 0
for index in range(len(predicted_label)):
	if predicted_label[index][0] == test_label[index][0]:
		accuracy += 1
accuracy /= float(len(predicted_label))
print (accuracy)
