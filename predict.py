from sqlalchemy.sql import func
class prediction:
	def __init__(self, name, ups, sent, count):
		self.name = name
		self.tot_ups = ups
		self.tot_sent = sent
		if count > 0:
			self.avg_ups = float(ups) / float(count)
			self.avg_sent = float(sent) / float(count)
			self.count = count
		else:
			self.avg_ups = 0
			self.avg_sent = 0
			self.count = 0

# session: db session 
# train_start: start time for training (unix time)
# train_end: end time for training set (unix time)
# test_start: start time for testing set (unix time)
# test_end: end time for testing set (unix time)
# subreddit: subreddit to run on
# percent: percentage of top users to track (float (0.0-1.0))
def predictUsers(session, train_start, train_end, test_start, test_end, subreddit, topic, percent):
	preds = []
	# query training data
	for name, ups, sent, count in session.query(Threads.username, func.sum(Threads.upvotes), func.sum(Sentiment.comments_sentiment), func.count(Threads.username)).\
	filter(Threads.subreddit = subreddit).filter(Sentiment.topic = topic).\
	filter(Threads.time >= train_start).filter(Threads.time < train_end).\
	group_by(Threads.username).all().order_by(func.avg(Threads.upvotes).desc()):
		preds.append(prediction(name, ups, sent, count))
	tot_users = percent * float(len(preds))
	results = []
	uk_ups = 0
	uk_sent = 0
	uk_count = 0
	# learn top users and their average pop/sent
	# learn average for unknown users
	for i in range(len(preds)):
		if i < tot_users:
			results.append(preds[i])
		else:
			uk_ups += preds[i].tot_ups
			uk_sent += preds[i].tot_sent
			uk_count += preds[i].count
	unknown = prediction("Unknown", uk_ups, uk_sent, uk_count)
	#results.append(unkown)
	# set all test rows to unknown value
	session.query().\
	filter(Threads.subreddit = subreddit).filter(Sentiment.topic = topic).\
	filter(Threads.time >= test_start).filter(Threads.time < test_end).\
	update({Threads.user_popularity = unknown.avg_ups}).\
	update({Sentiment.user_sentiment = unknown.avg_sent})
	
	# set all of the top users to their value
	for u in results:
		session.query().\
		filter(Threads.user == u.name).\
		filter(Threads.subreddit == subreddit).filter(Sentiment.topic == topic).\
		filter(Threads.time >= test_start).filter(Threads.time < test_end).\
		update({Threads.user_popularity = u.avg_ups}).\
		update({Sentiment.user_sentiment = u.avg_sent})
	# commit results to db
	session.commit()
	
# session: db session 
# train_start: start time for training (unix time)
# train_end: end time for training set (unix time)
# test_start: start time for testing set (unix time)
# test_end: end time for testing set (unix time)
# subreddit: subreddit to run on
# percent: percentage of top domains to track (float (0.0-1.0))	
def predictDomains(session, train_start, train_end, test_start, test_end, subreddit, topic, percent):
	preds = []
	# query training data
	for name, ups, sent, count in session.query(Threads.domain, func.sum(Threads.upvotes), func.sum(Sentiment.comments_sentiment), func.count(Threads.domain)).\
	filter(Threads.subreddit = subreddit).filter(Sentiment.topic = topic).\
	filter(Threads.time >= train_start).filter(Threads.time < train_end).\
	group_by(Threads.domain).all().order_by(func.avg(Threads.upvotes).desc()):
		preds.append(prediction(name, ups, sent, count))
	tot_domains = percent * float(len(preds))
	results = []
	uk_ups = 0
	uk_sent = 0
	uk_count = 0
	# learn top domains and their average pop/sent
	# learn average for unknown domains
	for i in range(len(preds)):
		if i < tot_domains:
			results.append(preds[i])
		else:
			uk_ups += preds[i].tot_ups
			uk_sent += preds[i].tot_sent
			uk_count += preds[i].count
	unknown = prediction("Unknown", uk_ups, uk_sent, uk_count)
	#results.append(unkown)
	# set all test rows to unknown value
	session.query().\
	filter(Threads.subreddit = subreddit).filter(Sentiment.topic = topic).\
	filter(Threads.time >= test_start).filter(Threads.time < test_end).\
	update({Threads.domain_popularity = unknown.avg_ups}).\
	update({Sentiment.domain_sentiment = unknown.avg_sent})
	
	# set all of the top domains to their value
	for u in results:
		session.query().\
		filter(Threads.domain == u.name).\
		filter(Threads.subreddit == subreddit).filter(Sentiment.topic == topic).\
		filter(Threads.time >= test_start).filter(Threads.time < test_end).\
		update({Threads.domain_popularity = u.avg_ups}).\
		update({Sentiment.domain_sentiment = u.avg_sent})
	# commit results to db
	session.commit()

def predictUser(session, username):
	total_ups = 0
	total_sent = 0
	count = 0
	for ups, sent in session.query(Threads.upvotes, Threads.assigned_label).filter(Threads.username == username):
		total_ups += ups
		total_sent += sent
		count += 1
	avg_upvotes = float(total_ups) / float(count)
	avg_sent = float(total_sent) / float(count)
	return (avg_upvotes, avg_sent)
	
def predictDomain(session, domain):
	total_ups = 0
	total_sent = 0
	count = 0
	for ups, sent in session.query(Threads.upvotes, Threads.assigned_label).filter(Threads.domain == domain):
		total_ups += ups
		total_sent += sent
		count += 1
	avg_upvotes = float(total_ups) / float(count)
	avg_sent = float(total_sent) / float(count)
	return (avg_upvotes, avg_sent)
