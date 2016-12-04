import numpy as np
import re
import itertools
from collections import Counter
import database


def clean_str(string):
	"""
	Tokenization/string cleaning for all datasets except for SST.
	Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
	"""
	string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
	string = re.sub(r"\'s", " \'s", string)
	string = re.sub(r"\'ve", " \'ve", string)
	string = re.sub(r"n\'t", " n\'t", string)
	string = re.sub(r"\'re", " \'re", string)
	string = re.sub(r"\'d", " \'d", string)
	string = re.sub(r"\'ll", " \'ll", string)
	string = re.sub(r",", " , ", string)
	string = re.sub(r"!", " ! ", string)
	string = re.sub(r"\(", " \( ", string)
	string = re.sub(r"\)", " \) ", string)
	string = re.sub(r"\?", " \? ", string)
	string = re.sub(r"\s{2,}", " ", string)
	return string.strip().lower()

def load_titles_and_votes_DB(session, subreddit, start_time, end_time):
	"""
	Loads MR polarity data from files, splits the data into words and generates labels.
	Returns split sentences and labels.
	"""
	examples = []
	labels = []

	threads = database.subreddit_query(session, subreddit, start_time, end_time)
	for thread in threads:
		examples.append(thread.title)
		labels.append(thread.upvotes)
	

	print ("number of titles")
	print (len(examples))
	# Split by words
	examples = [clean_str(sent) for sent in examples]
	labels = np.asarray(labels).reshape((len(labels),1))
	return [examples, labels]


def load_titles_and_sent_DB(session, topic, sentiment, start_time, end_time):
	"""
	Loads MR polarity data from files, splits the data into words and generates labels.
	Returns split sentences and labels.
	"""
	examples = []
	labels = []

	threads, sentiments = database.time_query(session, topic, 0, start_time, end_time)
	for thread, sent in threads, sentiments:
		examples.append(thread.title)
		labels.append(sent.comments_sentiment)
	

	print ("number of titles")
	print (len(examples))
	# Split by words
	examples = [clean_str(sent) for sent in examples]
	labels = np.asarray(labels)

	# Generate labels

	#print(y)
	return [examples, labels]

def load_comments_and_labels_DB(session, topic, start_time, end_time):
	"""
	Loads MR polarity data from files, splits the data into words and generates labels.
	Returns split sentences and labels.
	"""
	positive_examples = []
	negative_examples = []
	neutral_examples = []
	positive_threads = database.time_query(session, topic, 1, start_time, end_time)
	for thread, _ in positive_threads:
		for comment in thread.comments:
				if type(comment).__name__ == "Comment":
					positive_examples.append(comment.body[0:100])
	negative_threads = database.time_query(session, topic, -1, start_time, end_time)
	
	for thread, _ in negative_threads:
		for comment in thread.comments:
				if type(comment).__name__ == "Comment":
					negative_examples.append(comment.body[0:100])
	neutral_threads = database.time_query(session, "Irrelevant", 0, start_time, end_time)
	for thread, _ in neutral_threads:
		for comment in thread.comments:
				if type(comment).__name__ == "Comment":
					neutral_examples.append(comment.body[0:100])

	print ("number of comments")
	print (len(positive_examples))
	print (len(negative_examples))
	print (len(neutral_examples))
	# Split by words
	denominator = min(len(positive_examples), len(negative_examples), len(neutral_examples))
	x_text = positive_examples[0:denominator] + negative_examples[0:denominator] + neutral_examples[0:denominator]
	x_text = [clean_str(sent) for sent in x_text]

	# Generate labels
	positive_labels = [[0, 1, 0] for _ in positive_examples[0:denominator]]
	negative_labels = [[1, 0, 0] for _ in negative_examples[0:denominator]]
	neutral_labels = [[0, 0, 1] for _ in neutral_examples[0:denominator]]


	y = np.concatenate([positive_labels, negative_labels, neutral_labels], 0)
	#print(y)
	return [x_text, y]


def load_data_and_labels(positive_data_file, negative_data_file):
	"""
	Loads MR polarity data from files, splits the data into words and generates labels.
	Returns split sentences and labels.
	"""
	# Load data from files
	positive_examples = list(open(positive_data_file, "r").readlines())
	positive_examples = [s.strip() for s in positive_examples]
	negative_examples = list(open(negative_data_file, "r").readlines())
	negative_examples = [s.strip() for s in negative_examples]
	# Split by words
	x_text = positive_examples + negative_examples
	x_text = [clean_str(sent) for sent in x_text]
	# Generate labels
	positive_labels = [[0, 1] for _ in positive_examples]
	negative_labels = [[1, 0] for _ in negative_examples]
	y = np.concatenate([positive_labels, negative_labels], 0)
	return [x_text, y]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
	"""
	Generates a batch iterator for a dataset.
	"""
	data = np.array(data)
	data_size = len(data)
	num_batches_per_epoch = int(len(data)/batch_size) + 1
	for epoch in range(num_epochs):
		# Shuffle the data at each epoch
		if shuffle:
			shuffle_indices = np.random.permutation(np.arange(data_size))
			shuffled_data = data[shuffle_indices]
		else:
			shuffled_data = data
		for batch_num in range(num_batches_per_epoch):
			start_index = batch_num * batch_size
			end_index = min((batch_num + 1) * batch_size, data_size)
			yield shuffled_data[start_index:end_index]


