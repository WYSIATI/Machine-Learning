"""
This is for CSCI4155 Assignment 5
"""

import scipy.io
import math

# Get data from a matlab file NewsGroup.mat.
g_data = scipy.io.loadmat('NewsGroup.mat')
g_train_data = g_data.get('TRAIN_DATA')
g_test_data = g_data.get('TEST_DATA')
g_test_label = g_data.get('TEST_LABEL')


class NaiveBayesClassifier(object):
	"""
	Implementation of Naive Bayes classifier.
	"""

	def __init__(self):
		# Split g_train_data into 2 classes.

		self.words_in_class1 = []
		self.words_in_class2 = []
		for element in g_train_data:
			if element[0] < 481:
				self.words_in_class1.append(element)
			else:
				self.words_in_class2.append(element)

	def cal_prob(self, classn):
		# Calculate probability for each word in each given classn.
		# Take a list words with given class classn as input,
		# return a dictionary word_probabilities, the name is WordID,
		# the value is the probability of each word in the class.

		word_probabilities = dict()
		for element in classn:
			word_id = element[1]  # The 2nd column is WordID.
			if word_id in word_probabilities:
				word_probabilities[word_id] += element[2]  # The 3rd column is WordCount.
			else:
				word_probabilities[word_id] = element[2]
		length = float(len(classn))
		for element in word_probabilities:
			word_probabilities[element] = word_probabilities[element] / length
		return word_probabilities

	def prob_of_class(self, classn):
		# Calculate probabilities of documents for given classn,
		# take a list classn as input, use formula
		# P(x1, x2,...., xn | c) = P(x1|c)*P(x2|c)*.....*P(xn|c)
		# to calculate the probability, logarithm is applied here. 
		# Return a dictionary probability. It will use global
		# variable g_test_data as test document.

		document = []
		probabilities = []
		for element in g_test_data:
			if len(document) == 0:
				document.append(tuple(element))  # Because tuple type is hashable.
			if element[0] == document[0][0]:
				document.append(tuple(element))  # Numpy array and list are not hashable.
			else:
				temp_prob = self.prob_of_class_helper(document, classn)
				probabilities.append(temp_prob)
				document = []
		if len(document) > 0:
			temp_prob = self.prob_of_class_helper(document, classn)
			probabilities.append(temp_prob)
		return probabilities
		
	def prob_of_class_helper(self, document, classn):
		# Helper function of prob_of_class.
		# Return the probability of each document.

		probability = 0.0
		for word in document:
			word_id = word[1]
			word_count = word[2]
			if word_id in classn:
				probability += (math.log(classn[word_id])) * word_count
			else:
				# If the word does not exist in the particular class, count
				# its number of occurrence as 1 because of laplace smoothing.
				probability += (math.log(1.0/len(classn))) * word_count
		return probability

	def compute_accuracy(self):
		# Classfy g_test_data and compute accuracy

		# Calculate probability for each word in each given class.
		# word_prob_class1 is a dictionary with each word in class 1 and its probability,
		# word_prob_class2 is a dictionary with each word in class 2 and its probability.
		word_prob_class1 = self.cal_prob(self.words_in_class1)
		word_prob_class2 = self.cal_prob(self.words_in_class2)

		# prob1 and prob2 are the probabilities of documents with given
		# classes class 1 and class 2.
		prob1 = self.prob_of_class(word_prob_class1)
		prob2 = self.prob_of_class(word_prob_class2)

		# Compare prob1 and prob2 to get predicted_label
		predicted_label = []
		for index in range(len(prob1)):
			if prob1[index] < prob2[index]:
				predicted_label.append([2])
			else:
				predicted_label.append([1])

		accuracy = 0
		for index in range(len(predicted_label)):
			if predicted_label[index][0] == g_test_label[index][0]:
				accuracy += 1
		accuracy /= float(len(predicted_label))
		print "The accuracy of naive bayes classifier is", accuracy

def run():
	naive_bayes = NaiveBayesClassifier()
	naive_bayes.compute_accuracy()

run()
