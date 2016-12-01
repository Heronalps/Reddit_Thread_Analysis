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


# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

def eval_thread(trainedTopic, trained_start_time, trained_stop_time, x_raw, x_array):
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
	x_test = []
	for thread in x_array:
		x_test.append(np.array(list(vocab_processor.transform(thread))))
	#print(x_test)

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

			# Generate batches for one epoch
			#batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)
			thread_predictions = []
			# Collect the predictions here
			for thread in x_test:
				all_predictions = []
				for x_test_batch in thread:
					tb = np.asarray(x_test_batch).reshape(-1,len(x_test_batch))
					batch_predictions = sess.run(predictions, {input_x: np.asarray(tb), dropout_keep_prob: 1.0})
					all_predictions = np.concatenate([all_predictions, batch_predictions])
				thread_predictions.append(all_predictions)

	return thread_predictions


def test(trainedTopic, trained_start_time, trained_stop_time, subreddit, start_time, end_time):
	session = database.makeSession()
	threadlist = database.subreddit_query(session, subreddit, start_time, end_time)
	#thread = threadlist[0]
	#print(thread.title)
	runs = []
	votes = []
	raw = []
	negativeCnt = 0
	positiveCnt = 0
	irrelleventCnt = 0
	for thread in threadlist:
		inputvars = []
		threadvotes = []
		for comment in thread.comments:
			if type(comment).__name__ == "Comment":
				inputvars.append(comment.body[0:100])
				threadvotes.append(comment.ups)
		inputvars = [data_helpers.clean_str(sent) for sent in inputvars]
		votes.append(threadvotes)
		raw.append(inputvars)
		runs.append(inputvars)
		#print(inputvars)

	thread_array = (eval_thread(trainedTopic, trained_start_time, trained_stop_time, raw, runs))
	print (len(thread_array))
	for guess_array, thread_vote in zip(thread_array, votes):
		if (type(guess_array) is list):
			continue
		guess_array = guess_array.tolist() 
		negative = 0
		positive = 0
		irrellevent = 0
		for comment, vote in zip(guess_array,thread_vote):
			if (comment == 0):
				negative += 1#vote
			if (comment == 1):
				positive += 1#vote
			else:
				irrellevent += 1#vote
		#print (negative, irrellevent, positive)
		if (len(guess_array) < 2):
			continue
		if (negative > positive and negative > irrellevent):
			negativeCnt += 1
		if (negative < positive and positive > irrellevent):
			positiveCnt += 1
		else:
			irrelleventCnt += 1
	print(negativeCnt, positiveCnt, irrelleventCnt)





test("Trump", 1477977013, 1480396213, "funny", 1476921600, 1477008000)

	