#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
import database
from text_cnn import CommentCNN, TitleCNN
from tensorflow.contrib import learn

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_size", 1000, "size of the training data to use for validation")
tf.flags.DEFINE_string("positive_data_file", "../cnn-text-classification-tf-master-moddified/data/dononly_old.csv", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "../cnn-text-classification-tf-master-moddified/data/hillonly_old.csv", "Data source for the positive data.")

# Model Hyperparameters
tf.flags.DEFINE_integer("time_layer_size", 40, "Size of the time hidden layer(default: 40)")
tf.flags.DEFINE_integer("domain_layer_size", 40, "Size of the domain hidden layer(default: 40)")
tf.flags.DEFINE_integer("user_layer_size", 40, "Size of the user hidden layer(default: 40)")
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 20, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
# Misc Parameters
tf.flags.DEFINE_boolean("use_DB", True, "use the DB instead of files")
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
	print("{}={}".format(attr.upper(), value))
print("")

class predictordata:
	def __init__(session, topic, subreddit, start_time, end_time):  
		print("Loading data...")
		print("Loading data...{:d}, {:d}, {:s}".format(start_time, end_time, topic))


		threads, sentiments = database.subreddit_and_topic_query(session, topic, subreddit, start_time, end_time)
		self.dayofweek = []
		self.timeofday = []
		self.userpop = []
		self.usersent = []
		self.domainpop = []
		self.domainsent = []
		self.titles = []
		self.labels = []
		for thread, sentiment in threads, sentiments:
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
			

		print ("number of titles")
		print (len(examples))
		# Split by words
		self.dayofweek = np.asarray(self.dayofweek).reshape((len(self.dayofweek),1))
		self.timeofday = np.asarray(self.timeofday).reshape((len(self.timeofday),1))
		self.userpop = np.asarray(self.userpop).reshape((len(self.userpop),1))
		self.usersent = np.asarray(self.usersent).reshape((len(self.usersent),1))
		self.domainpop = np.asarray(self.domainpop).reshape((len(self.domainpop),1))
		self.domainsent = np.asarray(self.domainsent).reshape((len(self.domainsent),1))
		self.titles = [clean_str(sent) for sent in self.titles]
		self.labels = np.concatenate(self.labels, 0)

		max_document_length = max([len(x.split(" ")) for x in self.titles])
		self.vocab_processor = learn.preprocessing.VocabularyProcessor(59)
		self.titles = np.array(list(vocab_processor.fit_transform(self.titles)))
		# Randomly shuffle data
		np.random.seed(10)
		shuffle_indices = np.random.permutation(np.arange(len(self.labels)))

		self.dayofweek = self.dayofweek[shuffle_indices]
		self.timeofday = self.timeofday[shuffle_indices]
		self.userpop = self.userpop[shuffle_indices]
		self.usersent = self.usersent[shuffle_indices]
		self.domainpop = self.domainpop[shuffle_indices]
		self.domainsent = self.domainsent[shuffle_indices]
		self.titles = self.titles[shuffle_indices]
		self.labels = self.labels[shuffle_indices]

		dev_sample_index = -1 * int(FLAGS.dev_sample_size)
		self.dayofweek_train, self.dayofweek_test = self.dayofweek[:dev_sample_index], self.dayofweek[dev_sample_index:]
		self.timeofday_train, self.timeofday_test = self.timeofday[:dev_sample_index], self.timeofday[dev_sample_index:]
		self.userpop_train, self.userpop_test = self.userpop[:dev_sample_index], self.userpop[dev_sample_index:]
		self.usersent_train, self.usersent_test = self.usersent[:dev_sample_index], self.usersent[dev_sample_index:]
		self.domainpop_train, self.domainpop_test = self.domainpop[:dev_sample_index], self.domainpop[dev_sample_index:]
		self.domainsent_train, self.domainsent_test = self.domainsent[:dev_sample_index], self.domainsent[dev_sample_index:]
		self.titles_train, self.titles_test = self.titles[:dev_sample_index], self.titles[dev_sample_index:]
		self.labels_train, self.labels_test = self.labels[:dev_sample_index], self.labels[dev_sample_index:]

		print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
		print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))





# Data Preparatopn
# ==================================================
# Load data
def load_comment_vote_set(session, start_time, stop_time, target):
	print("Loading data...")
	print("Loading data...{:d}, {:d}, {:s}".format(start_time, stop_time, target))


	x_text, y = data_helpers.load_comments_and_labels_DB(session, target, start_time, stop_time)
   
	# Build vocabulary
	max_document_length = max([len(x.split(" ")) for x in x_text])
	vocab_processor = learn.preprocessing.VocabularyProcessor(59)
	#print (list(vocab_processor.fit_transform(x_text)))
	x = np.array(list(vocab_processor.fit_transform(x_text)))
	print(max_document_length)

	# Randomly shuffle data
	np.random.seed(10)
	shuffle_indices = np.random.permutation(np.arange(len(y)))
	x_shuffled = x[shuffle_indices]
	y_shuffled = y[shuffle_indices]

	# Split train/test set
	# TODO: This is very crude, should use cross-validation
	dev_sample_index = -1 * int(FLAGS.dev_sample_size)
	x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
	y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
	print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
	print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
	return x_train, x_dev, y_train, y_dev, vocab_processor

def load_title_vote_set(session, start_time, stop_time, subreddit):
	print("Loading data...")
	print("Loading data...{:d}, {:d}, {:s}".format(start_time, stop_time, subreddit))
	
	x_text, y = data_helpers.load_titles_and_votes_DB(session, subreddit, start_time, stop_time)
	
	# Build vocabulary
	max_document_length = max([len(x.split(" ")) for x in x_text])
	vocab_processor = learn.preprocessing.VocabularyProcessor(59)
	#print (list(vocab_processor.fit_transform(x_text)))
	x = np.array(list(vocab_processor.fit_transform(x_text)))
	# Randomly shuffle data
	np.random.seed(10)
	shuffle_indices = np.random.permutation(np.arange(len(y)))
	x_shuffled = x[shuffle_indices]
	y_shuffled = y[shuffle_indices]

	# Split train/test set
	# TODO: This is very crude, should use cross-validation
	dev_sample_index = -1 * int(FLAGS.dev_sample_size)
	x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
	y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
	print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
	print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
	return x_train, x_dev, y_train, y_dev, vocab_processor



# Training
# ==================================================

def run_comment(x_train, x_dev, y_train, y_dev, vocab_processor, topic, start_time, stop_time):
	with tf.Graph().as_default():
		session_conf = tf.ConfigProto(
		  allow_soft_placement=FLAGS.allow_soft_placement,
		  log_device_placement=FLAGS.log_device_placement)
		sess = tf.Session(config=session_conf)
		with sess.as_default():
			cnn = CommentCNN(
				sequence_length=x_train.shape[1],
				num_classes=y_train.shape[1],
				vocab_size=len(vocab_processor.vocabulary_),
				embedding_size=FLAGS.embedding_dim,
				filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
				num_filters=FLAGS.num_filters,
				l2_reg_lambda=FLAGS.l2_reg_lambda)

			# Define Training procedure
			global_step = tf.Variable(0, name="global_step", trainable=False)
			optimizer = tf.train.AdamOptimizer(1e-3)
			grads_and_vars = optimizer.compute_gradients(cnn.loss)
			train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

			# Keep track of gradient values and sparsity (optional)
			grad_summaries = []
			for g, v in grads_and_vars:
				if g is not None:
					grad_hist_summary = tf.histogram_summary("{}/grad/hist".format(v.name), g)
					sparsity_summary = tf.scalar_summary("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
					grad_summaries.append(grad_hist_summary)
					grad_summaries.append(sparsity_summary)
			grad_summaries_merged = tf.merge_summary(grad_summaries)

			# Output directory for models and summaries
			timestamp = str(int(time.time()))
			out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", topic + str(start_time)+ "-" + str(stop_time)))
			print("Writing to {}\n".format(out_dir))

			# Summaries for loss and accuracy
			loss_summary = tf.scalar_summary("loss", cnn.loss)
			acc_summary = tf.scalar_summary("accuracy", cnn.accuracy)

			# Train Summaries
			train_summary_op = tf.merge_summary([loss_summary, acc_summary, grad_summaries_merged])
			train_summary_dir = os.path.join(out_dir, "summaries", "train")
			train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph)

			# Dev summaries
			dev_summary_op = tf.merge_summary([loss_summary, acc_summary])
			dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
			dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, sess.graph)

			# Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
			checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
			checkpoint_prefix = os.path.join(checkpoint_dir, "model")
			if not os.path.exists(checkpoint_dir):
				os.makedirs(checkpoint_dir)
			saver = tf.train.Saver(tf.all_variables())

			# Write vocabulary
			vocab_processor.save(os.path.join(out_dir, "vocab"))

			# Initialize all variables
			sess.run(tf.initialize_all_variables())
			print("initialize_all_variables complete")

			def train_step(x_batch, y_batch):
				"""
				A single training step
				"""
				feed_dict = {
				  cnn.input_x: x_batch,
				  cnn.input_y: y_batch,
				  cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
				}
				_, step, summaries, loss, accuracy = sess.run(
					[train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
					feed_dict)
				time_str = datetime.datetime.now().isoformat()
				print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
				train_summary_writer.add_summary(summaries, step)

			def dev_step(x_batch, y_batch, writer=None):
				"""
				Evaluates model on a dev set
				"""
				feed_dict = {
				  cnn.input_x: x_batch,
				  cnn.input_y: y_batch,
				  cnn.dropout_keep_prob: 1.0
				}
				step, summaries, loss, accuracy = sess.run(
					[global_step, dev_summary_op, cnn.loss, cnn.accuracy],
					feed_dict)
				time_str = datetime.datetime.now().isoformat()
				print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
				if writer:
					writer.add_summary(summaries, step)

			# Generate batches
			batches = data_helpers.batch_iter(
				list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
			# Training loop. For each batch...
			for batch in batches:
				x_batch, y_batch = zip(*batch)
				train_step(x_batch, y_batch)
				#print(y_batch)
				current_step = tf.train.global_step(sess, global_step)
				if current_step % FLAGS.evaluate_every == 0:
					print("\nEvaluation:")
					dev_step(x_dev, y_dev, writer=dev_summary_writer)
					print("")
				if current_step % FLAGS.checkpoint_every == 0:
					path = saver.save(sess, checkpoint_prefix, global_step=current_step)
					print("Saved model checkpoint to {}\n".format(path))

def run_title(x_train, x_dev, y_train, y_dev, vocab_processor, target, start_time, stop_time):
	with tf.Graph().as_default():
		session_conf = tf.ConfigProto(
		  allow_soft_placement=FLAGS.allow_soft_placement,
		  log_device_placement=FLAGS.log_device_placement)
		sess = tf.Session(config=session_conf)
		with sess.as_default():
			cnn = TitleCNN(
				sequence_length=x_train.shape[1],
				num_classes=y_train.shape[1],
				vocab_size=len(vocab_processor.vocabulary_),
				embedding_size=FLAGS.embedding_dim,
				filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
				num_filters=FLAGS.num_filters,
				l2_reg_lambda=FLAGS.l2_reg_lambda)

			# Define Training procedure
			global_step = tf.Variable(0, name="global_step", trainable=False)
			optimizer = tf.train.AdamOptimizer(1e-3)
			grads_and_vars = optimizer.compute_gradients(cnn.loss)
			train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

			# Keep track of gradient values and sparsity (optional)
			grad_summaries = []
			for g, v in grads_and_vars:
				if g is not None:
					grad_hist_summary = tf.histogram_summary("{}/grad/hist".format(v.name), g)
					sparsity_summary = tf.scalar_summary("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
					grad_summaries.append(grad_hist_summary)
					grad_summaries.append(sparsity_summary)
			grad_summaries_merged = tf.merge_summary(grad_summaries)

			# Output directory for models and summaries
			timestamp = str(int(time.time()))
			out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", target  + str(start_time)+ "-" + str(stop_time)))
			print("Writing to {}\n".format(out_dir))

			# Summaries for loss and accuracy
			loss_summary = tf.scalar_summary("loss", cnn.loss)
			acc_summary = tf.scalar_summary("accuracy", cnn.accuracy)

			# Train Summaries
			train_summary_op = tf.merge_summary([loss_summary, acc_summary, grad_summaries_merged])
			train_summary_dir = os.path.join(out_dir, "summaries", "train")
			train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph)

			# Dev summaries
			dev_summary_op = tf.merge_summary([loss_summary, acc_summary])
			dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
			dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, sess.graph)

			# Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
			checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
			checkpoint_prefix = os.path.join(checkpoint_dir, "model")
			if not os.path.exists(checkpoint_dir):
				os.makedirs(checkpoint_dir)
			saver = tf.train.Saver(tf.all_variables())

			# Write vocabulary
			vocab_processor.save(os.path.join(out_dir, "vocab"))

			# Initialize all variables
			sess.run(tf.initialize_all_variables())
			print("initialize_all_variables complete")

			def train_step(x_batch, y_batch):
				"""
				A single training step
				"""
				feed_dict = {
				  cnn.input_x: x_batch,
				  cnn.input_y: y_batch,
				  cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
				}
				_, step, summaries, loss, accuracy = sess.run(
					[train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
					feed_dict)
				time_str = datetime.datetime.now().isoformat()
				print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
				train_summary_writer.add_summary(summaries, step)

			def dev_step(x_batch, y_batch, writer=None):
				"""
				Evaluates model on a dev set
				"""
				feed_dict = {
				  cnn.input_x: x_batch,
				  cnn.input_y: y_batch,
				  cnn.dropout_keep_prob: 1.0
				}
				step, summaries, loss, accuracy = sess.run(
					[global_step, dev_summary_op, cnn.loss, cnn.accuracy],
					feed_dict)
				time_str = datetime.datetime.now().isoformat()
				print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
				if writer:
					writer.add_summary(summaries, step)

			# Generate batches
			batches = data_helpers.batch_iter(
				list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
			# Training loop. For each batch...
			for batch in batches:
				#print (batch)
				x_batch, y_batch = zip(*batch)
				train_step(x_batch, y_batch)
				#print(y_batch)
				current_step = tf.train.global_step(sess, global_step)
				if current_step % FLAGS.evaluate_every == 0:
					print("\nEvaluation:")
					dev_step(x_dev, y_dev, writer=dev_summary_writer)
					print("")
				if current_step % FLAGS.checkpoint_every == 0:
					path = saver.save(sess, checkpoint_prefix, global_step=current_step)
					print("Saved model checkpoint to {}\n".format(path))

def run_predict(predictobj, target, start_time, stop_time):
	with tf.Graph().as_default():
		session_conf = tf.ConfigProto(
		  allow_soft_placement=FLAGS.allow_soft_placement,
		  log_device_placement=FLAGS.log_device_placement)
		sess = tf.Session(config=session_conf)
		with sess.as_default():
			cnn = preCNN(
				sequence_length=x_train.shape[1],
				num_classes=y_train.shape[1],
				vocab_size=len(predictobj.vocab_processor.vocabulary_),
				time_layer_size = FLAGS.time_layer_size, 
				user_layer_size = FLAGS.user_layer_size,
				domain_layer_size = FLAGS.domain_layer_size,
				embedding_size=FLAGS.embedding_dim,
				filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
				num_filters=FLAGS.num_filters,
				l2_reg_lambda=FLAGS.l2_reg_lambda)

			# Define Training procedure
			global_step = tf.Variable(0, name="global_step", trainable=False)
			optimizer = tf.train.AdamOptimizer(1e-3)
			grads_and_vars = optimizer.compute_gradients(cnn.loss)
			train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

			# Keep track of gradient values and sparsity (optional)
			grad_summaries = []
			for g, v in grads_and_vars:
				if g is not None:
					grad_hist_summary = tf.histogram_summary("{}/grad/hist".format(v.name), g)
					sparsity_summary = tf.scalar_summary("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
					grad_summaries.append(grad_hist_summary)
					grad_summaries.append(sparsity_summary)
			grad_summaries_merged = tf.merge_summary(grad_summaries)

			# Output directory for models and summaries
			timestamp = str(int(time.time()))
			out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", target  + str(start_time)+ "-" + str(stop_time)))
			print("Writing to {}\n".format(out_dir))

			# Summaries for loss and accuracy
			loss_summary = tf.scalar_summary("loss", cnn.loss)
			acc_summary = tf.scalar_summary("accuracy", cnn.accuracy)

			# Train Summaries
			train_summary_op = tf.merge_summary([loss_summary, acc_summary, grad_summaries_merged])
			train_summary_dir = os.path.join(out_dir, "summaries", "train")
			train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph)

			# Dev summaries
			dev_summary_op = tf.merge_summary([loss_summary, acc_summary])
			dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
			dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, sess.graph)

			# Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
			checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
			checkpoint_prefix = os.path.join(checkpoint_dir, "model")
			if not os.path.exists(checkpoint_dir):
				os.makedirs(checkpoint_dir)
			saver = tf.train.Saver(tf.all_variables())

			# Write vocabulary
			predictobj.vocab_processor.save(os.path.join(out_dir, "vocab"))

			# Initialize all variables
			sess.run(tf.initialize_all_variables())
			print("initialize_all_variables complete")

			def train_step(titles, labels, dayofweek, timeofday, userpop, usersent, domainpop, domainsent):
				"""
				A single training step
				"""
				feed_dict = {
					cnn.input_x: titles,
					cnn.input_y: labels,
					cnn.dayofweek: dayofweek,
					cnn.timeofday: timeofday,
					cnn.userpop: userpop,
					cnn.usersent: usersent,
					cnn.domainpop: domainpop,
					cnn.domainsent: domainsent,
					cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
				}
				_, step, summaries, loss, accuracy = sess.run(
					[train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
					feed_dict)
				time_str = datetime.datetime.now().isoformat()
				print("{}: step {}, loss {:g}, percent err {:g}".format(time_str, step, loss, accuracy))
				train_summary_writer.add_summary(summaries, step)

			def dev_step(titles, labels, dayofweek, timeofday, userpop, usersent, domainpop, domainsent, writer=None):
				"""
				Evaluates model on a dev set
				"""
				feed_dict = {
					cnn.input_x: titles,
					cnn.input_y: labels,
					cnn.dayofweek: dayofweek,
					cnn.timeofday: timeofday,
					cnn.userpop: userpop,
					cnn.usersent: usersent,
					cnn.domainpop: domainpop,
					cnn.domainsent: domainsent,
					cnn.dropout_keep_prob: 1.0
				}
				step, summaries, loss, accuracy = sess.run(
					[global_step, dev_summary_op, cnn.loss, cnn.accuracy],
					feed_dict)
				time_str = datetime.datetime.now().isoformat()
				print("{}: step {}, loss {:g}, percent err {:g}".format(time_str, step, loss, accuracy))
				if writer:
					writer.add_summary(summaries, step)

			# Generate batches
			batches = data_helpers.batch_iter(
				list(zip(predictobj.titles, predictobj.labels, predictobj.dayofweek, predictobj.timeofday, predictobj.userpop, predictobj.usersent, predictobj.domainpop, predictobj.domainsent)), FLAGS.batch_size, FLAGS.num_epochs)
			# Training loop. For each batch...
			for batch in batches:
				#print (batch)
				titles, labels, dayofweek, timeofday, userpop, usersent, domainpop, domainsent = zip(*batch)
				train_step(titles, labels, dayofweek, timeofday, userpop, usersent, domainpop, domainsent)
				#print(y_batch)
				current_step = tf.train.global_step(sess, global_step)
				if current_step % FLAGS.evaluate_every == 0:
					print("\nEvaluation:")
					dev_step(titles, labels, dayofweek, timeofday, userpop, usersent, domainpop, domainsent, writer=dev_summary_writer)
					print("")
				if current_step % FLAGS.checkpoint_every == 0:
					path = saver.save(sess, checkpoint_prefix, global_step=current_step)
					print("Saved model checkpoint to {}\n".format(path))

def train_comments_from_db(start_time, stop_time, topic):
	session = database.makeSession()
	x_train, x_dev, y_train, y_dev, vocab_processor = load_comment_vote_set(session, start_time, stop_time, topic)
	run_comment(x_train, x_dev, y_train, y_dev, vocab_processor, topic, start_time, stop_time)

def train_title_votes_from_db(start_time, stop_time, subreddit):
	session = database.makeSession()
	x_train, x_dev, y_train, y_dev, vocab_processor = load_title_vote_set(session, start_time, stop_time, subreddit)
	run_title(x_train, x_dev, y_train, y_dev, vocab_processor, subreddit + "votes", start_time, stop_time)

def train_title_sent_from_db(start_time, stop_time, subreddit):
	session = database.makeSession()
	x_train, x_dev, y_train, y_dev, vocab_processor = load_train_set(session, start_time, stop_time, target, title = True, sentiment=True)
	run(x_train, x_dev, y_train, y_dev, vocab_processor, subreddit + "sentiment", start_time, stop_time)

def predict_from_db(start_time, end_time, subreddit, topic):
	session = database.makeSession()
	predictobj = predictor(session, topic, subreddit, start_time, end_time)
	run_predict(predictobj, subreddit + topic + "predictor", start_time, end_time)


train_comments_from_db(1479880024, 1480484824, "Trump")

#train_comments_from_db(1479880024, 1480484824, "Trump")