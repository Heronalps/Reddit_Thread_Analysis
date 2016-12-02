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
