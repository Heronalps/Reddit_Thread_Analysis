#! /usr/bin/env python
from sqlalchemy.sql import func, update, select
import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import CommentCNN, TitleCNN
from tensorflow.contrib import learn
import csv
import database

# Parameters
# ==================================================

class threadobject:
	def __init__(self, thread, topic = None):
		self.thread = thread
		self.topic = topic
		self.comments = []
		self.votes = []
		for comment in thread.comments:
			if type(comment).__name__ == "Comment":
				self.comments.append(comment.body[0:100])
				self.votes.append(comment.ups)
		self.comments = [data_helpers.clean_str(sent) for sent in self.comments]
		self.transformed = None
		self.predictions = None

		
# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

def eval_comments(trainedTopic, trained_start_time, trained_stop_time, threadobjlist):
	FLAGS = tf.flags.FLAGS
	FLAGS._parse_flags()
	print("\nParameters:")
	for attr, value in sorted(FLAGS.__flags.items()):
		print("{}={}".format(attr.upper(), value))
	print("")
	#print(x_raw)
	# Map data into vocabulary
	vocab_path = os.path.join("runs", trainedTopic + str(trained_start_time)+ "-" + str(trained_stop_time), "vocab")
	vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
	for thread in threadobjlist:
		thread.transformed = np.array(list(vocab_processor.transform(thread.comments)))
	

	print("\nEvaluating...\n")

	# Evaluation
	# ==================================================
	checkpoint_file = tf.train.latest_checkpoint(os.path.join("runs", trainedTopic + str(trained_start_time)+ "-" + str(trained_stop_time), "checkpoints"))
	graph = tf.Graph()
	with graph.as_default():
		session_conf = tf.ConfigProto(
		  allow_soft_placement=FLAGS.allow_soft_placement,
		  log_device_placement=FLAGS.log_device_placement)
		sess = tf.Session(config=session_conf)
		with sess.as_default():
			# Load the saved meta graph and restore variables
			saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
			saver.restore(sess, checkpoint_file)

			# Get the placeholders from the graph by name
			input_x = graph.get_operation_by_name("input_x").outputs[0]
			# input_y = graph.get_operation_by_name("input_y").outputs[0]
			dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

			# Tensors we want to evaluate
			predictions = graph.get_operation_by_name("output/predictions").outputs[0]
			newlist = []
			# Collect the predictions here
			for thread in threadobjlist:
				if (len(thread.transformed) > 0):
					batch = np.array(thread.transformed)
					#print (x_test_batch)
					thread.predictions = (sess.run(predictions, {input_x: batch, dropout_keep_prob: 1.0}))
				newlist.append(thread)
			
				

		return newlist

def eval_title_votes(target, trained_start_time, trained_stop_time, threadobjlist):
	FLAGS = tf.flags.FLAGS
	FLAGS._parse_flags()
	print("\nParameters:")
	for attr, value in sorted(FLAGS.__flags.items()):
		print("{}={}".format(attr.upper(), value))
	print("")
	#print(x_raw)
	# Map data into vocabulary
	vocab_path = os.path.join("runs", target + str(trained_start_time)+ "-" + str(trained_stop_time), "vocab")
	vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
	for thread in threadobjlist:
		thread.transformed = np.array(list(vocab_processor.transform(thread.comments)))
	

	print("\nEvaluating...\n")

	# Evaluation
	# ==================================================
	checkpoint_file = tf.train.latest_checkpoint(os.path.join("runs", target + str(trained_start_time)+ "-" + str(trained_stop_time), "checkpoints"))
	graph = tf.Graph()
	with graph.as_default():
		session_conf = tf.ConfigProto(
		  allow_soft_placement=FLAGS.allow_soft_placement,
		  log_device_placement=FLAGS.log_device_placement)
		sess = tf.Session(config=session_conf)
		with sess.as_default():
			# Load the saved meta graph and restore variables
			saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
			saver.restore(sess, checkpoint_file)

			# Get the placeholders from the graph by name
			input_x = graph.get_operation_by_name("input_x").outputs[0]
			# input_y = graph.get_operation_by_name("input_y").outputs[0]
			dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

			# Tensors we want to evaluate
			predictions = graph.get_operation_by_name("output/predictions").outputs[0]
			newlist = []
			# Collect the predictions here
			for thread in threadobjlist:
				if (len(thread.transformed) > 0):
					batch = np.array(thread.transformed)
					#print (x_test_batch)
					thread.predictions = (sess.run(predictions, {input_x: batch, dropout_keep_prob: 1.0}))
				newlist.append(thread)
			
				

		return newlist

def eval_predictions(target, trained_start_time, trained_stop_time, predictorobj):
	FLAGS = tf.flags.FLAGS
	FLAGS._parse_flags()
	print("\nParameters:")
	for attr, value in sorted(FLAGS.__flags.items()):
		print("{}={}".format(attr.upper(), value))
	print("")
	#print(x_raw)
	# Map data into vocabulary
	vocab_path = os.path.join("runs", target + str(trained_start_time)+ "-" + str(trained_stop_time), "vocab")
	vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
	predictorobj.process(vocab_processor)

		# Evaluation
	# ==================================================
	checkpoint_file = tf.train.latest_checkpoint(os.path.join("runs", target + str(trained_start_time)+ "-" + str(trained_stop_time), "checkpoints"))
	graph = tf.Graph()
	with graph.as_default():
		session_conf = tf.ConfigProto(
		  allow_soft_placement=FLAGS.allow_soft_placement,
		  log_device_placement=FLAGS.log_device_placement)
		sess = tf.Session(config=session_conf)
		with sess.as_default():
			# Load the saved meta graph and restore variables
			saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
			saver.restore(sess, checkpoint_file)

			# Get the placeholders from the graph by name
			input_x = graph.get_operation_by_name("input_x").outputs[0]
			dayofweek = graph.get_operation_by_name("input_dayofweek").outputs[0]
			timeofday = graph.get_operation_by_name("input_timeofday").outputs[0]
			userpop = graph.get_operation_by_name("input_userpop").outputs[0]
			usersent = graph.get_operation_by_name("input_usersent").outputs[0]
			domainpop = graph.get_operation_by_name("input_domainpop").outputs[0]
			domainsent = graph.get_operation_by_name("input_domainsent").outputs[0]
			# input_y = graph.get_operation_by_name("input_y").outputs[0]
			dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

			# Tensors we want to evaluate
			predictions = graph.get_operation_by_name("output/predictions").outputs[0]
			batches = data_helpers.batch_iter(
				list(zip(predictorobj.titles, predictorobj.dayofweek,\
				predictorobj.timeofday, predictorobj.userpop, predictorobj.usersent,\
				predictorobj.domainpop, predictorobj.domainsent)), 64, 1, shuffle=False)
			# Collect the predictions here
			labels = []
			for batch in batches:
				#print (x_test_batch)
				b_titles, b_dayofweek, b_timeofday, b_userpop, b_usersent, b_domainpop, b_domainsent = zip(*batch)
				#print(type(b_dayofweek))

				feed_dict = {
				input_x: b_titles, 
				dayofweek: b_dayofweek,
				timeofday: b_timeofday, 
				userpop: b_userpop, 
				usersent: b_usersent, 
				domainpop: b_domainpop,
				domainsent: b_domainsent, 
				dropout_keep_prob: 1.0
				}
				labels= labels + (sess.run(predictions, feed_dict).tolist())
			
		#labels = np.asarray(labels).reshape((len(labels),1))
		#print (sum(labels, []))
		return (sum(labels, []))
	
	

	print("\nEvaluating...\n")

def test_comments(session, train_start, train_end, test_start, test_end, topic, weight_by_votes=False):
	
	threadlist = database.time_query(session, topic, 0, test_start, test_end)

	threadobjlist = []
	for thread, sent in threadlist:
		threadobjlist.append(threadobject(thread, sent))
	#print(threadobjlist)
	#print(topic)
	thread_array = (eval_comments(topic, train_start, train_end, threadobjlist))
	#print (thread_array)
	#print (len(threadobjlist))
	print("number of threads")
	print (len(thread_array))
	score = 0
	num = len(thread_array)
	for threadobj in thread_array:
		if (threadobj.predictions != None):
			guess_array = threadobj.predictions.tolist()
			if (weight_by_votes == True):
				thread_vote = threadobj.votes
			else:
				thread_vote = [1] * len(threadobj.votes)
			positive = 0
			negative = 0
			irrellevent = 0
			for comment, vote in zip(guess_array,thread_vote):
				if (comment == 0):
					negative += vote
				if (comment == 1):
					positive += vote
				else:
					irrellevent += vote
			sentiment_score = (positive-negative) /float(positive+negative+irrellevent)
		else:
			sentiment_score = 0
		score += sentiment_score
		#print(threadobj.thread.title)
	#print(score/num)
		threadobj.topic.comments_sentiment = sentiment_score
	session.commit()

def test_votes(session, trained_start_time, trained_stop_time, test_start, test_end, subreddit):
	threadlist = database.subreddit_query(session, subreddit, test_start, test_end)
	
	#thread = threadlist[0]
	#print(threadlist[0])
	threadobjlist = []
	for thread in threadlist:
		threadobjlist.append(threadobject(thread))

	thread_array = eval_title_votes(subreddit, trained_start_time, trained_stop_time, threadobjlist)
	#print(threadobjlist)

	print (len(threadobjlist))
	print (len(thread_array))
	for threadobj in thread_array:
		print(threadobj)
		title_popularity = threadobj.predictions
		threadobj.thread.title_popularity = title_popularity
	session.commit()

def test_sent_predictor(session, topic, subreddit, trained_start_time, trained_stop_time, test_start, test_end):
	session = database.makeSession()
	predictobj = data_helpers.predictordata(session, topic, subreddit, test_start, test_end)

	threadlist = eval_predictions(subreddit + topic + "sent_predictor", trained_start_time, trained_stop_time, predictobj)
	print(len(predictobj.threadIDs))
	print(len(threadlist))
	for threadID, thread, output in zip(predictobj.threadIDs, predictobj.threads, threadlist):
		#print (threadID, thread[1].threadid, output)
		thread[1].predicted_sentiment = output

		"""session.query(database.Sentiment).filter(database.Sentiment.threadid == threadID).\
		filter(database.Sentiment.topic == topic).update({database.Sentiment.predicted_sentiment: output})"""
		"""session.execute(update(database.Sentiment).\
		values({database.Sentiment.predicted_sentiment: output}).\
		where(database.Sentiment.threadid == threadID).\
		where(database.Sentiment.topic == topic))"""
		#where(Sentiment.threadid.in_(select([Threads.threadid]).where(Threads.subreddit == subreddit).\
		#where(Threads.time >= test_start).where(Threads.time < test_end).as_scalar())))
		session.commit()

def test_pop_predictor(session, topic, subreddit, trained_start_time, trained_stop_time, test_start, test_end):
	session = database.makeSession()
	predictobj = data_helpers.predictordata(session, topic, subreddit, test_start, test_end)

	threadlist = eval_title_sent(subreddit + topic + "pop_predictor", trained_start_time, trained_stop_time, predictobj)
	print(len(predictobj.threadIDs))
	print(len(predictobj.threads))
	print(len(threadlist))
	for threadID, output in (predictobj.threadIDs, threadlist):
		session.execute(update(Sentiment).\
		values({Sentiment.predicted_popularity: output}).\
		where(Sentiment.threadid == threadID).\
		where(Sentiment.topic == topic))
		#session.execute(update(Sentiment).\
		#values({Sentiment.predicted_popularity: output}).\
		#where(Sentiment.threadid == select([Threads.threadid]).where(Threads.subreddit == subreddit).\
		#where(Threads.time >= test_start).where(Threads.time < test_end)))
		session.commit()

if __name__ == '__main__':
	
	session = database.makeSession()
	test_sent_predictor(session, "Trump", "politics", 1475280001, 1479600000, 1479600000, 1480550401)
	#test_comments(session, 1478995200, 1480550401, 1475280001, 1480550401, "Trump", False)
	#tq = database.subreddit_and_topic_query(session, "Trump", "news", 1479600000, 1480396213)
	"""count = 0
	for thread in tq:
		count += 1
		print("Thread " + str(count))
		print(thread[1].topic + ", " + str(thread[1].comments_sentiment) + ": [" + thread[0].subreddit + "] [" + (thread[0].title) + "] [" + str(thread[0].time) + "]")
	from database import Sentiment, Threads
	print("sum test")
	for sent, ups in session.query(func.sum(Sentiment.comments_sentiment), func.sum(Threads.upvotes)).filter(Threads.threadid == Sentiment.threadid).\
		filter(Threads.subreddit == "news").filter(Sentiment.topic == "Trump").\
		filter(Threads.time >= 1479600000).filter(Threads.time < 1480396213).\
		group_by(Threads.domain).order_by(func.avg(Threads.upvotes).desc()):

		print(sent, ups)"""