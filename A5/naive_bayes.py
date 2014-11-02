"""
This is for CSCI4155 Assignment 5
"""

import scipy.io
import math

# Get data from a matlab file NewsGroup.mat.
g_data = None
try:
	g_data = scipy.io.loadmat('NewsGroup.mat')
except:
	print "Can't find the Matlab file"
	exit(0)
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
			if element[0] < 481:  # Because the first 480 documents are class 1.
				self.words_in_class1.append(element)
			else:
				self.words_in_class2.append(element)

	def cal_prob(self, classn):
		# Calculate probability for each word in each given classn.
		# Take a list of words with given class classn as input,
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
		# for element in g_test_data:
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
		# Classfy g_test_data and compute accuracy.

		# Calculate probability for each word in each given class.
		# word_prob_class1 is a dictionary with each word in class 1 and its probability,
		# word_prob_class2 is a dictionary with each word in class 2 and its probability.
		word_prob_class1 = self.cal_prob(self.words_in_class1)
		word_prob_class2 = self.cal_prob(self.words_in_class2)

		# prob1 and prob2 are the probabilities of documents with given
		# classes class 1 and class 2 respectively.
		prob1 = self.prob_of_class(word_prob_class1)
		prob2 = self.prob_of_class(word_prob_class2)

		# Compare prob1 and prob2 to get predicted_label.
		predicted_label = []
		for index in range(len(prob1)):
			if prob1[index] < prob2[index]:
				predicted_label.append([2])
			else:
				predicted_label.append([1])

		# Compute accuracy.
		accuracy = 0
		for index in range(len(predicted_label)):
			if predicted_label[index] == g_test_label[index]:
				accuracy += 1
		accuracy /= float(len(predicted_label))
		print "The accuracy of naive bayes classifier is", accuracy


import numpy as np
from sklearn import svm

g_train_label = g_data.get('TRAIN_LABEL')


class SVMClassifier(object):
	"""
	Implementation of SVM classifier.
	"""

	def __init__(self):
		# Construct train_samples according to DocumentID

		# Because I am using np.vstack() to concatenate two numpy array, while
		# np.vstack() does not take empty array, the first DocumentId is used. Each
		# row of g_train_data is a word, the first column of the word is DocumentID
		first_document_id = g_train_data[0][0]
		self.train_samples = np.array([first_document_id])
		for element in g_train_data:
			document_id = element[0]
			if first_document_id != document_id:
				self.train_samples = np.vstack((self.train_samples, np.array([document_id])))
				first_document_id = document_id

		# Initialize sklearn.svm.SVC() object.
		self.clf = svm.SVC()

	def train(self):
		# Train svm with sklearn.svm.SVC().

		train_label = []
		for index in range(len(g_train_label)):
			train_label.append(g_train_label[index][0])

		self.clf.fit(self.train_samples, train_label)

	def test(self):
		# test svm with sklearn.svm.SVC().

		# Build test_samples.
		first_test_document_id = g_test_data[0][0]
		test_samples = np.array([first_test_document_id])
		for element in g_test_data:
			document_id = element[0]
			if first_test_document_id != document_id:
				test_samples = np.vstack((test_samples, np.array([document_id])))
				first_test_document_id = document_id

		predicted_labels = self.clf.predict(test_samples)
		accuracy = 0
		for index in range(len(predicted_labels)):
			if predicted_labels[index] == g_test_label[index][0]:
				accuracy += 1
		accuracy /= float(len(predicted_labels))
		print "The accuracy of SVM classifier is",accuracy


def run():
	naive_bayes = NaiveBayesClassifier()
	naive_bayes.compute_accuracy()

	svm_clf = SVMClassifier()
	svm_clf.train()
	svm_clf.test()

run()
