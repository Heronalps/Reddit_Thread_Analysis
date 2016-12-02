#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
import csv
import database

# Parameters
# ==================================================

class threadobject:
	def __init__(self, thread):
		self.thread = thread
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

def eval_thread(trainedTopic, trained_start_time, trained_stop_time, threadobjlist):
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


def test(trainedTopic, trained_start_time, trained_stop_time, subreddit, start_time, end_time, weight_by_votes=False):
	session = database.makeSession()
	threadlist = database.subreddit_query(session, subreddit, start_time, end_time)
	#thread = threadlist[0]
	#print(thread.title)
	threadobjlist = []
	for thread in threadlist:
		threadobjlist.append(threadobject(thread))
	print(threadobjlist)
	thread_array = (eval_thread(trainedTopic, trained_start_time, trained_stop_time, threadobjlist))

	print (len(threadobjlist))
	print (len(thread_array))
	for threadobj in thread_array:
		if (threadobj.predictions != None):
			guess_array = threadobj.predictions.tolist()
			if weight_by_votes:
				thread_vote = threadobjlist.votes
			else:
				threadvote = [1] * len(threadobjlist.votes)
			for comment, vote in zip(guess_array,thread_vote):
			if (comment == 0):
				negative += vote
			if (comment == 1):
				positive += vote
			else:
				irrellevent += vote
			sentiment = (positive-negative) /(positive+negative+irrellevent)
		data_helpers.update_score(thread.threadid, sentiment)





test("Trump", 1477977013, 1480396213, "funny", 1479081600, 1479091600)

	