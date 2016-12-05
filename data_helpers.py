import numpy as np
import re
import itertools
from collections import Counter
import database
from tensorflow.contrib import learn

class predictordata:
	def __init__(self, session, topic, subreddit, start_time, end_time):  
		print("Loading data...")
		print("Loading data...{:d}, {:d}, {:s}".format(start_time, end_time, topic))


		threads = database.subreddit_and_topic_query(session, topic, subreddit, start_time, end_time)
		self.threadIDs = []
		self.dayofweek = []
		self.timeofday = []
		self.userpop = []
		self.usersent = []
		self.domainpop = []
		self.domainsent = []
		self.titles = []
		self.labels = []
		self.vocab_processor = None
		for thread, sentiment in threads:
			self.threadIDs.append(thread.threadid)
			timeofday = ((thread.time-start_time)%86400) - (430200)
			self.timeofday.append(timeofday)
			dayofweek = (((thread.time-start_time)%604800)/86400) - 3
			self.dayofweek.append(dayofweek)

			self.userpop.append(thread.user_popularity)
			self.usersent.append(sentiment.user_sentiment)
			self.domainpop.append(thread.domain_popularity)
			self.domainsent.append(sentiment.domain_sentiment)

			self.titles.append(thread.title)
			self.labels.append([thread.upvotes, sentiment.comments_sentiment])
			

		print ("number of titles and lables")
		print (len(self.titles), len(self.labels))

		# Split by words
		self.threadIDs = np.asarray(self.threadIDs).reshape((len(self.threadIDs),1))
		self.dayofweek = np.asarray(self.dayofweek).reshape((len(self.dayofweek),1))
		self.dayofweek = np.asarray(self.dayofweek).reshape((len(self.dayofweek),1))
		self.timeofday = np.asarray(self.timeofday).reshape((len(self.timeofday),1))
		self.userpop = np.asarray(self.userpop).reshape((len(self.userpop),1))
		self.usersent = np.asarray(self.usersent).reshape((len(self.usersent),1))
		self.domainpop = np.asarray(self.domainpop).reshape((len(self.domainpop),1))
		self.domainsent = np.asarray(self.domainsent).reshape((len(self.domainsent),1))
		self.titles = [clean_str(sent) for sent in self.titles]
		self.labels = np.asarray(self.labels).reshape((len(self.labels),2))
		#print(self.labels.shape)

		

	def process(self, vocab = None):

		max_document_length = max([len(x.split(" ")) for x in self.titles])

		if (vocab == None):
			self.vocab_processor = learn.preprocessing.VocabularyProcessor(59)
			self.titles = np.array(list(self.vocab_processor.fit_transform(self.titles)))
		else:
			self.vocab_processor = vocab
			self.titles = np.array(list(self.vocab_processor.fit_transform(self.titles)))

		np.random.seed(10)
		shuffle_indices = np.random.permutation(np.arange(len(self.labels)))

		self.threadIDs = self.threadIDs[shuffle_indices]
		self.dayofweek = self.dayofweek[shuffle_indices]
		self.timeofday = self.timeofday[shuffle_indices]
		self.userpop = self.userpop[shuffle_indices]
		self.usersent = self.usersent[shuffle_indices]
		self.domainpop = self.domainpop[shuffle_indices]
		self.domainsent = self.domainsent[shuffle_indices]
		self.titles = self.titles[shuffle_indices]
		self.labels = self.labels[shuffle_indices]

		dev_sample_index = -1 * int(10)
		self.dayofweek_train, self.dayofweek_test = self.dayofweek[:dev_sample_index], self.dayofweek[dev_sample_index:]
		self.timeofday_train, self.timeofday_test = self.timeofday[:dev_sample_index], self.timeofday[dev_sample_index:]
		self.userpop_train, self.userpop_test = self.userpop[:dev_sample_index], self.userpop[dev_sample_index:]
		self.usersent_train, self.usersent_test = self.usersent[:dev_sample_index], self.usersent[dev_sample_index:]
		self.domainpop_train, self.domainpop_test = self.domainpop[:dev_sample_index], self.domainpop[dev_sample_index:]
		self.domainsent_train, self.domainsent_test = self.domainsent[:dev_sample_index], self.domainsent[dev_sample_index:]
		self.titles_train, self.titles_test = self.titles[:dev_sample_index], self.titles[dev_sample_index:]
		self.labels_train, self.labels_test = self.labels[:dev_sample_index], self.labels[dev_sample_index:]

		print("Vocabulary Size: {:d}".format(len(self.vocab_processor.vocabulary_)))
		print("Train/Dev split: {:d}/{:d}".format(len(self.labels_train), len(self.labels_test)))
		print("userpop: ")
		print(self.userpop)
		print("usersent: ")
		print(self.usersent)
		print("domainpop: ")
		print(self.domainpop)
		print("domainsent: ")
		print(self.domainsent)


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


