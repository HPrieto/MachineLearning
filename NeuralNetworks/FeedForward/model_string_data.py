import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import random
import pickle
from collections import Counter

lemmatizer = WordNetLemmatizer()
hm_lines = 10000000

def create_lexicon(pos, neg):
	lexicon = []
	# open text files and populate lexicon
	for fi in [pos,neg]:
		# speficy intent to read contents of file
		with open(fi,'r') as f:
			# get textfile lines
			contents = f.readlines()
			# iterate through all words
			for l in contents[:hm_lines]:
				# tokenize word
				all_words = word_tokenize(l.lower())
				# add tokenized words to lexicon
				lexicon += list(all_words)

	lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
	# convert lexicon to dictionary element {'word': 123423}
	w_counts = Counter(lexicon)
	# final lexicon
	l2 = []
	# check occurences of each work in new dictionary
	for w in w_counts:
		if 1000 > w_counts[w] > 50:
			l2.append(w)
	print('Lexicon Length: ', len(l2))
	# return final lexicon
	return l2

# Classify featuresets
def sample_handling(sample, lexicon, classification):
	# List of Features and Spefied classification
	featureset = []
	with open(sample,'r') as f:
		contents = f.readlines()
		for l in contents[:hm_lines]:
			# Tokenize and convert toLower
			current_words = word_tokenize(l.lower())
			# Lemmatize words and put in array
			current_words = [lemmatizer.lemmatize(i) for i in current_words]
			# New array of all zeros
			features = np.zeros(len(lexicon))
			# Search for word in lexicon and return index value
			for word in current_words:
				# Search for word in lexicon
				if word.lower() in lexicon:
					# Get index of word in lexicon
					index_value = lexicon.index(word.lower())
					# increment word count at index when found
					features[index_value] += 1
			# Convert features array into list array
			features = list(features)
			featureset.append([features,classification])
	return featureset

def create_feature_sets_and_labels(pos, neg, test_size=0.1):
	# Lexicon for 'pos.txt' and 'neg.txt'
	lexicon = create_lexicon(pos, neg)
	features = []
	features += sample_handling('pos.txt',lexicon,[1,0])
	features += sample_handling('neg.txt',lexicon,[0,1])
	# Shuffle for Neural Network Modeling
	random.shuffle(features)
	# Convert features array into numpy array
	features = np.array(features)
	# 10% of features
	testing_size = int(test_size * len(features))
	# Get all zeroeth elements from features up to the last 10 percent: 
	# features[[features,label],[features,label]]
	train_x = list(features[:,0][:-testing_size])
	train_y = list(features[:,1][:-testing_size])
	# Gest testing data or last 10% of data
	test_x = list(features[:,0][-testing_size:])
	test_y = list(features[:,1][-testing_size:])
	return train_x, train_y, test_x, test_y

if __name__ == '__main__':
	train_x,train_y,test_x,test_y = create_feature_sets_and_labels('pos.txt','neg.txt')
	with open('sentiment_set.pickle','wb') as f:
		pickle.dump([train_x,train_y,test_x,test_y], f)


































