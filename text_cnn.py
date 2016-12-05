import tensorflow as tf
import numpy as np


class CommentCNN(object):
	"""
	A CNN for text classification.
	Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
	"""
	def __init__(
	  self, sequence_length, num_classes, vocab_size,
	  embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):

		# Placeholders for input, output and dropout
		self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
		self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
		self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

		# Keeping track of l2 regularization loss (optional)
		l2_loss = tf.constant(0.0)

		# Embedding layer
		with tf.device('/cpu:0'), tf.name_scope("embedding"):
			W = tf.Variable(
				tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
				name="W")
			self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
			self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

		# Create a convolution + maxpool layer for each filter size
		pooled_outputs = []
		for i, filter_size in enumerate(filter_sizes):
			with tf.name_scope("conv-maxpool-%s" % filter_size):
				# Convolution Layer
				filter_shape = [filter_size, embedding_size, 1, num_filters]
				W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
				b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
				conv = tf.nn.conv2d(
					self.embedded_chars_expanded,
					W,
					strides=[1, 1, 1, 1],
					padding="VALID",
					name="conv")
				# Apply nonlinearity
				h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
				# Maxpooling over the outputs
				pooled = tf.nn.max_pool(
					h,
					ksize=[1, sequence_length - filter_size + 1, 1, 1],
					strides=[1, 1, 1, 1],
					padding='VALID',
					name="pool")
				pooled_outputs.append(pooled)

		# Combine all the pooled features
		num_filters_total = num_filters * len(filter_sizes)
		self.h_pool = tf.concat(3, pooled_outputs)
		self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

		# Add dropout
		with tf.name_scope("dropout"):
			self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

		with tf.name_scope("Hidden"):
			h_W = tf.get_variable(
				"h_W",
				shape=[num_filters_total, num_filters_total],
				initializer=tf.contrib.layers.xavier_initializer())
			h_b = tf.Variable(tf.constant(0.1, shape=[num_filters_total]), name="h_b")
			l2_loss += tf.nn.l2_loss(h_W)
			l2_loss += tf.nn.l2_loss(b)
			#print(self.h_drop)
			#rint(h_W)
			
			h = tf.nn.xw_plus_b(self.h_drop, h_W, h_b, name="hidden_scores")
			self.hidden_scores = tf.nn.relu(h, name="relu")
			#print(self.hidden_scores)


		with tf.name_scope("hidden_dropout"):
			self.hh_drop = tf.nn.dropout(self.hidden_scores, self.dropout_keep_prob)
		# Final (unnormalized) scores and predictions
		with tf.name_scope("output"):
			W = tf.get_variable(
				"W",
				shape=[num_filters_total, num_classes],
				initializer=tf.contrib.layers.xavier_initializer())
			b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
			l2_loss += tf.nn.l2_loss(W)
			l2_loss += tf.nn.l2_loss(b)
			#print(self.h_drop)
			#print(W)
			self.scores = tf.nn.xw_plus_b(self.hh_drop, W, b, name="scores")
			self.predictions = tf.argmax(self.scores, 1, name="predictions")

		# CalculateMean cross-entropy loss
		with tf.name_scope("loss"):
			losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.input_y)
			self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

		# Accuracy
		with tf.name_scope("accuracy"):
			correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
			self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

class TitleCNN(object):
	"""
	A CNN for text regression.
	Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
	"""
	def __init__(
	  self, sequence_length, num_classes, vocab_size,
	  embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):

		# Placeholders for input, output and dropout
		self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
		self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
		self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

		# Keeping track of l2 regularization loss (optional)
		l2_loss = tf.constant(0.0)

		# Embedding layer
		with tf.device('/cpu:0'), tf.name_scope("embedding"):
			W = tf.Variable(
				tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
				name="W")
			self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
			self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

		# Create a convolution + maxpool layer for each filter size
		pooled_outputs = []
		for i, filter_size in enumerate(filter_sizes):
			with tf.name_scope("conv-maxpool-%s" % filter_size):
				# Convolution Layer
				filter_shape = [filter_size, embedding_size, 1, num_filters]
				W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
				b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
				conv = tf.nn.conv2d(
					self.embedded_chars_expanded,
					W,
					strides=[1, 1, 1, 1],
					padding="VALID",
					name="conv")
				# Apply nonlinearity
				h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
				# Maxpooling over the outputs
				pooled = tf.nn.max_pool(
					h,
					ksize=[1, sequence_length - filter_size + 1, 1, 1],
					strides=[1, 1, 1, 1],
					padding='VALID',
					name="pool")
				pooled_outputs.append(pooled)

		# Combine all the pooled features
		num_filters_total = num_filters * len(filter_sizes)
		print(num_filters_total)
		self.h_pool = tf.concat(3, pooled_outputs)
		self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

		# Add dropout
		with tf.name_scope("dropout"):
			self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)
			
		# Final (unnormalized) scores and predictions
		with tf.name_scope("Hidden"):
			h_W = tf.get_variable(
				"h_W",
				shape=[num_filters_total, num_filters_total],
				initializer=tf.contrib.layers.xavier_initializer())
			h_b = tf.Variable(tf.constant(0.1, shape=[num_filters_total]), name="h_b")
			l2_loss += tf.nn.l2_loss(h_W)
			l2_loss += tf.nn.l2_loss(b)
			#print(self.h_drop)
			#rint(h_W)
			h_s = tf.nn.xw_plus_b(self.h_drop, h_W, h_b, name="hidden_scores")
			self.hidden_scores = tf.tanh(h_s, name="hidden_time")
			#print(self.hidden_scores)


		with tf.name_scope("hidden_dropout"):
			self.hh_drop = tf.nn.dropout(self.hidden_scores, self.dropout_keep_prob)
		# Final (unnormalized) scores and predictions
		with tf.name_scope("output"):
			W = tf.get_variable(
				"W",
				shape=[num_filters_total, num_classes],
				initializer=tf.contrib.layers.xavier_initializer())
			b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
			l2_loss += tf.nn.l2_loss(W)
			l2_loss += tf.nn.l2_loss(b)
			#print(self.h_drop)
			#print(W)
			self.scores = tf.nn.xw_plus_b(self.hh_drop, W, b, name="scores")
			self.predictions = tf.nn.xw_plus_b(self.hh_drop, W, b, name="predictions")
			#print (self.scores)
			#self.predictions = (self.scores, 1, name="predictions")

		# CalculateMean cross-entropy loss
		with tf.name_scope("loss"):
			#losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.input_y)
			losses = tf.square(self.scores - self.input_y)
			self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

		# Accuracy
		with tf.name_scope("accuracy"):
			rmsloss = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(self.predictions, self.input_y))))
			percent_err= tf.divide(rmsloss, self.input_y)
			self.accuracy = tf.reduce_mean(percent_err, name="percent_error")
			print (self.accuracy)

class PredictorCNN(object):
	"""
	A CNN for text regression.
	Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
	"""
	def __init__(
	  self, sequence_length, num_classes, vocab_size, time_layer_size, user_layer_size,
	  domain_layer_size, embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):

		# Placeholders for input, output and dropout
		self.timeofday = tf.placeholder(tf.float32, [None, 1], name="input_timeofday")
		self.dayofweek = tf.placeholder(tf.float32, [None, 1], name="input_dayofweek")
		self.userpop = tf.placeholder(tf.float32, [None, 1], name="input_userpop")
		self.usersent = tf.placeholder(tf.float32, [None, 1], name="input_usersent")
		self.domainpop = tf.placeholder(tf.float32, [None, 1], name="input_domainpop")
		self.domainsent = tf.placeholder(tf.float32, [None, 1], name="input_domainsent")
		self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="target_popularity")
		self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="target_sentiment")
		self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

		# Keeping track of l2 regularization loss (optional)
		l2_loss = tf.constant(0.0)

		# Embedding layer
		with tf.device('/cpu:0'), tf.name_scope("embedding"):
			W = tf.Variable(
				tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
				name="W")
			self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
			self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

		# Create a convolution + maxpool layer for each filter size
		pooled_outputs = []
		for i, filter_size in enumerate(filter_sizes):
			with tf.name_scope("conv-maxpool-%s" % filter_size):
				# Convolution Layer
				filter_shape = [filter_size, embedding_size, 1, num_filters]
				W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
				b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
				conv = tf.nn.conv2d(
					self.embedded_chars_expanded,
					W,
					strides=[1, 1, 1, 1],
					padding="VALID",
					name="conv")
				# Apply nonlinearity
				h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
				# Maxpooling over the outputs
				pooled = tf.nn.max_pool(
					h,
					ksize=[1, sequence_length - filter_size + 1, 1, 1],
					strides=[1, 1, 1, 1],
					padding='VALID',
					name="pool")
				pooled_outputs.append(pooled)

		# Combine all the pooled features
		num_filters_total = num_filters * len(filter_sizes)
		print(num_filters_total)
		self.h_pool = tf.concat(3, pooled_outputs)
		self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total],)

		with tf.name_scope("timelayer"):
			time_vec = tf.concat(1, [self.timeofday, self.dayofweek])
			#print( time_vec)
			t_W = tf.get_variable(
				"t_W",
				shape=[2, time_layer_size],
				initializer=tf.contrib.layers.xavier_initializer())
			t_b = tf.Variable(tf.constant(0.001, shape=[time_layer_size]), name="time_b")
			l2_loss += tf.nn.l2_loss(t_W)
			l2_loss += tf.nn.l2_loss(t_b)
			h_t = tf.nn.xw_plus_b(time_vec, t_W, t_b)
			self.hidden_time = tf.tanh(h_t, name="hidden_time")

		with tf.name_scope("userlayer"):
			user_vec = tf.concat(1, [self.userpop, self.usersent])
			u_W = tf.get_variable(
				"u_W",
				shape=[2, user_layer_size],
				initializer=tf.contrib.layers.xavier_initializer())
			u_b = tf.Variable(tf.constant(0.001, shape=[user_layer_size]), name="user_b")
			l2_loss += tf.nn.l2_loss(u_W)
			l2_loss += tf.nn.l2_loss(u_b)
			h_u = tf.nn.xw_plus_b(user_vec, u_W, u_b)
			self.hidden_user = tf.tanh(h_u, name="hidden_user")

		with tf.name_scope("domainlayer"):
			domain_vec = tf.concat(1, [self.domainpop, self.domainsent])
			d_W = tf.get_variable(
				"d_W",
				shape=[2, domain_layer_size],
				initializer=tf.contrib.layers.xavier_initializer())
			d_b = tf.Variable(tf.constant(0.001, shape=[domain_layer_size]), name="domain_b")
			l2_loss += tf.nn.l2_loss(d_W)
			l2_loss += tf.nn.l2_loss(d_b)
			h_d = tf.nn.xw_plus_b(domain_vec, d_W, d_b)
			self.hidden_domain = tf.tanh(h_d, name="hidden_domain")



		# Add dropout
		with tf.name_scope("concat_dropout"):
			self.concat = tf.concat(1, [self.hidden_time, self.hidden_user, self.hidden_domain, self.h_pool_flat], name="concat")
			self.h_drop = tf.nn.dropout(self.concat, self.dropout_keep_prob)
			
		# Final (unnormalized) scores and predictions
		with tf.name_scope("Hidden"):
			h_W = tf.get_variable(
				"h_W",
				shape=[num_filters_total+domain_layer_size+user_layer_size+time_layer_size, num_filters_total],
				initializer=tf.contrib.layers.xavier_initializer())
			h_b = tf.Variable(tf.constant(0.001, shape=[num_filters_total]), name="h_b")
			l2_loss += tf.nn.l2_loss(h_W)
			l2_loss += tf.nn.l2_loss(h_b)
			#print(self.h_drop)
			#rint(h_W)
			h = tf.nn.xw_plus_b(self.h_drop, h_W, h_b)
			self.hidden_scores = tf.tanh(h, name="hidden_scores")
			#print(self.hidden_scores)


		with tf.name_scope("hidden_dropout"):
			self.hh_drop = tf.nn.dropout(self.hidden_scores, self.dropout_keep_prob)
		# Final (unnormalized) scores and predictions
		with tf.name_scope("output"):
			W = tf.get_variable(
				"W",
				shape=[num_filters_total, num_classes],
				initializer=tf.contrib.layers.xavier_initializer())
			b = tf.Variable(tf.constant(0.001, shape=[num_classes]), name="b")
			l2_loss += tf.nn.l2_loss(W)
			l2_loss += tf.nn.l2_loss(b)
			#print(self.h_drop)
			#print(W)
			self.scores = tf.nn.xw_plus_b(self.hh_drop, W, b, name="scores")
			self.predictions = tf.nn.xw_plus_b(self.hh_drop, W, b, name="predictions")
			#print (self.scores)
			#self.predictions = (self.scores, 1, name="predictions")

		# CalculateMean cross-entropy loss
		with tf.name_scope("loss"):
			#losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.input_y)
			losses = tf.square(self.scores - self.input_y)
			self.loss = tf.reduce_mean(tf.reduce_mean(losses) + l2_reg_lambda * l2_loss)

		# Accuracy
		with tf.name_scope("accuracy"):
			rmsloss = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(self.predictions, self.input_y))))
			percent_err= tf.divide(rmsloss, self.input_y)
			self.accuracy = tf.reduce_mean(percent_err, name="percent_error")
			print (self.accuracy)