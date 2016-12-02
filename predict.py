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

def predictUsers(session, percent):
	preds = []
	for name, ups, sent, count in session.query(Threads.username, func.avg(Threads.upvotes), func.avg(threads.comments_sentiment), func.count(Threads.username)).group_by(Threads.username).all().order_by(func.avg(Threads.upvotes).desc()):
		preds.append(prediction(name, ups, sent, count))
	tot_users = percent * float(len(preds))
	results = []
	uk_ups = 0
	uk_sent = 0
	uk_count = 0
	for i in range(len(preds)):
		if i < tot_users:
			results.append(preds[i])
		else:
			uk_ups += preds[i].tot_ups
			uk_sent += preds[i].tot_sent
			uk_count += preds[i].count
	unknown = prediction("Unknown", uk_ups, uk_sent, uk_count)
	results.append(unkown)
	

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
