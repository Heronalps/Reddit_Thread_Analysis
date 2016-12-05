
from sqlalchemy.sql import func, update, select
from database import Threads, Sentiment, makeSession
class prediction:
	def __init__(self, name, ups, sent, count):
		self.name = name
		self.tot_ups = ups
		self.tot_sent = sent
		if count > 0:
			self.avg_ups = float(ups) / float(count)
			self.avg_sent = float(self.tot_sent) / float(count)
			#self.avg_sent = 69
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
	for name, ups, sent, count in session.query(Threads.user, func.sum(Threads.upvotes), func.sum(Sentiment.comments_sentiment), func.count(Threads.user)).\
	filter(Threads.threadid == Sentiment.threadid).\
	filter(Threads.subreddit == subreddit).filter(Sentiment.topic == topic).\
	filter(Threads.time >= train_start).filter(Threads.time < train_end).\
	group_by(Threads.user).order_by(func.avg(Threads.upvotes).desc()):
		#print("Preds[" + str(len(preds)) + "]" + " = " + name + " " + str(ups) + " " + str(count))
		preds.append(prediction(name, ups, sent, count))
	tot_users = percent * float(len(preds))
	tot_users = int(tot_users)
	print("Tracking a total of " + str(tot_users) + " users")
	#results = []
	#uk_ups = 0
	#uk_sent = 0
	#uk_count = 0
	# learn top users and their average pop/sent
	# learn average for unknown users
	print("Calculating User results")
	#for i in range(len(preds)):
	#	if i < tot_users:
	#		print("Name: [" + preds[i].name + "] ups: [" + str(preds[i].tot_ups) + "] count: [" + str(preds[i].count) + "]")
	#		results.append(preds[i])
	#	else:
	#		uk_ups += preds[i].tot_ups
	#		uk_sent += preds[i].tot_sent
	#		uk_count += preds[i].count		
	results = preds[:tot_users]
	uk_ups = sum([p.tot_ups for p in preds[tot_users:]])
	uk_sent = sum([p.tot_sent for p in preds[tot_users:]])
	uk_count = sum([p.count for p in preds[tot_users:]])
	unknown = prediction("Unknown", uk_ups, uk_sent, uk_count)
	#results.append(unkown)
	# set all test rows to unknown value
	#for thread, sent in session.query(Threads, Sentiment).\
	#filter(Threads.threadid == Sentiment.threadid).\
	#filter(Threads.subreddit == subreddit).filter(Sentiment.topic == topic).\
	#filter(Threads.time >= test_start).filter(Threads.time < test_end):
	#	#update({Threads.user_popularity : unknown.avg_ups}).\
	#	#update({Sentiment.user_sentiment : unknown.avg_sent})
	#	thread.user_popularity = unknown.avg_ups
	#	sent.user_sentiment = unknown.avg_sent
	session.execute(update(Threads).\
	values({Threads.user_popularity: unknown.avg_ups}).\
	where(Threads.subreddit == subreddit).\
	where(Threads.time >= test_start).where(Threads.time < test_end))
	session.commit()
	session.execute(update(Sentiment).\
	values({Sentiment.user_sentiment: unknown.avg_sent}).\
	where(Sentiment.threadid.in_(select([Threads.threadid]).where(Threads.subreddit == subreddit).\
	where(Threads.time >= test_start).where(Threads.time < test_end).as_scalar())))
	session.commit()
	
	# set all of the top users to their value
	for u in results:
	
		session.execute(update(Threads).\
		values({Threads.user_popularity: u.avg_ups}).\
		where(Threads.subreddit == subreddit).where(Threads.user == u.name).\
		where(Threads.time >= test_start).where(Threads.time < test_end))
		session.commit()
		
		session.execute(update(Sentiment).\
		values({Sentiment.user_sentiment: u.avg_sent}).\
		where(Sentiment.threadid.in_(select([Threads.threadid]).where(Threads.subreddit == subreddit).where(Threads.user == u.name).\
		where(Threads.time >= test_start).where(Threads.time < test_end).as_scalar())))
		session.commit()
		continue
		for thread, sent in session.query(Threads, Sentiment).\
		filter(Threads.threadid == Sentiment.threadid).\
		filter(Threads.user == u.name).\
		filter(Threads.subreddit == subreddit).filter(Sentiment.topic == topic).\
		filter(Threads.time >= test_start).filter(Threads.time < test_end):
		#update({Threads.user_popularity : u.avg_ups}).\
		#update({Sentiment.user_sentiment : u.avg_sent})
			thread.user_popularity = u.avg_ups
			sent.user_sentiment = u.avg_sent
		
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
	filter(Threads.threadid == Sentiment.threadid).\
	filter(Threads.subreddit == subreddit).filter(Sentiment.topic == topic).\
	filter(Threads.time >= train_start).filter(Threads.time < train_end).\
	group_by(Threads.domain).order_by(func.avg(Threads.upvotes).desc()):
		#print("Name: [" + name + "] ups: [" + str(ups) + "] count: [" + str(count) + "]")
		#print("Preds[" + str(len(preds)) + "]" + " = " + name + " " + str(ups) + " " + str(count))
		preds.append(prediction(name, ups, sent, count))
	tot_domains = percent * float(len(preds))
	tot_domains = int(tot_domains)
	print("Tracking a total of " + str(tot_domains) + " domains")
	#results = []
	#uk_ups = 0
	#uk_sent = 0
	#uk_count = 0
	# learn top domains and their average pop/sent
	# learn average for unknown domains
	print("Calculating domain results")
	#for i in range(len(preds)):
	#	if i < tot_domains:
	#		print("Name: [" + preds[i].name + "] ups: [" + str(preds[i].tot_ups) + "] count: [" + str(preds[i].count) + "]")
	#		results.append(preds[i])
	#	else:
	#		uk_ups += preds[i].tot_ups
	#		uk_sent += preds[i].tot_sent
	#		uk_count += preds[i].count
	results = preds[:tot_domains]
	uk_ups = sum([p.tot_ups for p in preds[tot_domains:]])
	uk_sent = sum([p.tot_sent for p in preds[tot_domains:]])
	uk_count = sum([p.count for p in preds[tot_domains:]])
	#print("Length of new results: " + str(len(new_results)))
	#print("newups: [" + str(new_ups) + "] new_sent: [" + str(new_sent) + "] new_count: [" + str(new_count) + "]")
	#print("Length of old results: " + str(len(results)))
	#print("oldups: [" + str(uk_ups) + "] old_sent: [" + str(uk_sent) + "] old_count: [" + str(uk_count) + "]")
	unknown = prediction("Unknown", uk_ups, uk_sent, uk_count)
	#print("Name: [" + unknown.name + "] ups: [" + str(unknown.tot_ups) + "] count: [" + str(unknown.count) + "]")
	#results.append(unkown)
	# set all test rows to unknown value
	print("Updating unknown domains")
	#for thread, sent in session.query(Threads,Sentiment).\
	#filter(Threads.threadid == Sentiment.threadid).\
	#filter(Threads.subreddit == subreddit).filter(Sentiment.topic == topic).\
	#filter(Threads.time >= test_start).filter(Threads.time < test_end):
	##update({Threads.domain_popularity: unknown.avg_ups}).\
	##update({Sentiment.domain_sentiment : unknown.avg_sent})
	#	thread.domain_popularity = unknown.avg_ups
	#	sent.domain_sentiment = unknown.avg_sent
	#session.commit()
	
	session.execute(update(Threads).\
	values({Threads.domain_popularity: unknown.avg_ups}).\
	where(Threads.subreddit == subreddit).\
	where(Threads.time >= test_start).where(Threads.time < test_end))
	session.commit()
	session.execute(update(Sentiment).\
	values({Sentiment.domain_sentiment: unknown.avg_sent}).\
	where(Sentiment.topic == topic).\
	where(Sentiment.threadid.in_(select([Threads.threadid]).where(Threads.subreddit == subreddit).\
	where(Threads.time >= test_start).where(Threads.time < test_end).as_scalar())))
	session.commit()
	
	
	# set all of the top domains to their value
	print("Updating known domains (" + str(len(results)) + ")")
	for u in results:
		#ids = session.execute(select([Threads.threadid]).where(Threads.subreddit == subreddit).where(Threads.domain == u.name).\
		#where(Threads.time >= test_start).where(Threads.time < test_end))
		#print(ids.domain)
		#session.commit()
		session.execute(update(Threads).\
		values({Threads.domain_popularity: u.avg_ups}).\
		where(Threads.subreddit == subreddit).where(Threads.domain == u.name).\
		where(Threads.time >= test_start).where(Threads.time < test_end))
		session.commit()
		
		session.execute(update(Sentiment).\
		values({Sentiment.domain_sentiment: u.avg_sent}).\
		where(Sentiment.topic == topic).\
		where(Sentiment.threadid.in_(select([Threads.threadid]).where(Threads.subreddit == subreddit).where(Threads.domain == u.name).\
		where(Threads.time >= test_start).where(Threads.time < test_end).as_scalar())))
		session.commit()
		continue
		for d_pop, d_sent in session.query(Threads, Sentiment).\
		filter(Threads.threadid == Sentiment.threadid).\
		filter(Threads.domain == u.name).\
		filter(Threads.subreddit == subreddit).filter(Sentiment.topic == topic).\
		filter(Threads.time >= test_start).filter(Threads.time < test_end):
		#update({Threads.domain_popularity: u.avg_ups}).\
		#update({Threads.sentiments.domain_sentiment: 42})
			#thread.domain_popularity = u.avg_ups
			#sent.domain_sentiment = u.avg_sent
			d_pop = u.avg_ups
			d_sent = u.avg_sent

	# commit results to db
	session.commit()

def predictUser(session, username):
	total_ups = 0
	total_sent = 0
	count = 0
	for ups, sent in session.query(Threads.upvotes, Threads.assigned_label).filter(Threads.user == username):
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

if __name__ == "__main__":
	session = makeSession()
	test_start = 1479600000
	test_end = 1480396213
	topic = "Trump"
	subreddit = "news"
	predictDomains(session, 1479513600, 1479600000, 1479600000, 1480396213, "news", "Trump", .5)
	for thread, sent in session.query(Threads, Sentiment).\
		filter(Threads.threadid == Sentiment.threadid).\
		filter(Threads.subreddit == subreddit).filter(Sentiment.topic == topic).\
		filter(Threads.time >= test_start).filter(Threads.time < test_end):
		#update({Threads.domain_popularity: u.avg_ups}).\
		#update({Sentiment.domain_sentiment : u.avg_sent})
			print("Thread [" + thread.domain + "] sent: [" + str(sent.domain_sentiment) + "] pop: [" + str(thread.domain_popularity) + "]")
	predictUsers(session, 1479513600, 1479600000, 1479600000, 1480396213, "news", "Trump", .5)
	for thread, sent in session.query(Threads, Sentiment).\
		filter(Threads.threadid == Sentiment.threadid).\
		filter(Threads.subreddit == subreddit).filter(Sentiment.topic == topic).\
		filter(Threads.time >= test_start).filter(Threads.time < test_end):
		#update({Threads.domain_popularity: u.avg_ups}).\
		#update({Sentiment.domain_sentiment : u.avg_sent})
			print("Thread [" + thread.user + "] sent: [" + str(sent.user_sentiment) + "] pop: [" + str(thread.user_popularity) + "]")

