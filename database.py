from sqlalchemy import Table, Column, ForeignKey, BigInteger, Integer, Float, String, PickleType, Boolean, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy import create_engine
from sqlalchemy.sql import exists
import argparse
import re
Base = declarative_base()


class Threads(Base):
	__tablename__ = 'threads'
	
	threadid = Column(String(255), primary_key=True)
#	topic = Column(String(255), primary_key=True)
#	sentiment = Column(Integer)
#	comments_sentiment = Column(Integer)
	# One-to-many relationship
	sentiments = relationship("Sentiment", back_populates="thread")
	
	title = Column(Text)
	time = Column(BigInteger)
	subreddit = Column(String(255))
	selfpost = Column(Boolean)
	selftext = Column(Text)
	domain = Column(String(255))
	upvotes = Column(Integer)
	comments = Column(PickleType)
	user = Column(String(255))

	# Per-attribute predicted popularity
	title_popularity = Column(Float)
	domain_popularity = Column(Float)
	user_popularity = Column(Float)
	time_of_day_popularity = Column(Float)
	time_of_week_popularity = Column(Float)
	
class Sentiment(Base):
	__tablename__ = 'sentiments'
	topic = Column(String(255), primary_key=True)
	threadid = Column(String(255), ForeignKey('threads.threadid'), primary_key=True)
	# Many-to-one relationship
	thread = relationship("Threads", back_populates="sentiments")
	
	# Label for thread sentiment:
	# 1 if from biased pro-topic subreddit
	# 0 if from "neutral" subreddit
	# -1 if from biased anti-topic subreddit
	sentiment = Column(Integer)
	# Predicted overall sentiment
	predicted_sentiment = Column(Float)
	# Predicted overall popularity
	predicted_popularity = Column(Float)

	# Sentiment label given by comments (provided by CNN)
	comments_sentiment = Column(Float)
	# Per-attributed predicted sentiments
	title_sentiment = Column(Float)
	domain_sentiment = Column(Float)
	user_sentiment = Column(Float)
	time_of_day_sentiment = Column(Float)
	time_of_week_sentiment = Column(Float)

	
	
def create(name):
	engine = create_engine('sqlite:///' + name)
	Base.metadata.create_all(engine)
	
def query(session, topic, sentiment):
	threads = []
	for thread in session.query(Threads, Sentiment).filter(Threads.threadid == Sentiment.threadid).filter(Sentiment.topic==topic).filter(Sentiment.sentiment == sentiment):
		threads.append(thread)
	return threads

def time_query(session, topic, sentiment, start_time, end_time):
	threads = []
	for thread, sent in session.query(Threads, Sentiment).\
	filter(Threads.threadid == Sentiment.threadid).\
	filter(Sentiment.topic==topic).filter(Sentiment.sentiment == sentiment).\
	filter(Threads.time >= start_time).filter(Threads.time < end_time):
		threads.append((thread, sent))
	return threads

def subreddit_and_topic_query(session, topic, subreddit, start_time, end_time):
	threads = []
	for thread, sent in session.query(Threads, Sentiment).\
	filter(Threads.threadid == Sentiment.threadid).\
	filter(Threads.subreddit == subreddit).\
	filter(Sentiment.topic==topic).\
	filter(Threads.time >= start_time).filter(Threads.time < end_time):
		threads.append((thread, sent))
	return threads

def subreddit_query(session, subreddit, start_time, end_time):
	threads = []
	for thread in session.query(Threads).\
	filter(Threads.subreddit==subreddit).\
	filter(Threads.time >= start_time).filter(Threads.time < end_time):
		threads.append(thread)
	return threads
	
def testSession():
	engine = create_engine('sqlite:///test.db')
	Base.metadata.bind = engine
	DBSession = sessionmaker(bind = engine)
	session = DBSession()
	return session
	
def makeSession():
	engine = create_engine('sqlite:///reddit.db.bk')
	Base.metadata.bind = engine
	DBSession = sessionmaker(bind = engine)
	session = DBSession()
	return session
	
def addThread(session, tpc, sntmnt, thrd):
	t = session.query(Threads).filter(Threads.threadid == thrd.threadid).one_or_none()
	#print(str(t))
	if t is not None:
		print("Thread [" + toAscii(thrd.title) + "] already in db")
		s = t.sentiments
		if tpc in [x.topic for x in s]:
			print("Topic [" + toAscii(tpc) + "] for thread [" + toAscii(thrd.title) + "] already in db")
			return
		else:
			newsent = Sentiment(topic=tpc, sentiment = sntmnt)
			t.sentiments.append(newsent)
			session.commit()
			return
	print("Thread [" + toAscii(thrd.title) + "] not in db, adding for topic [" + toAscii(tpc) + "]")
	t = Threads(threadid = thrd.threadid, title = thrd.title, time = thrd.time, subreddit = thrd.subreddit, selfpost = thrd.selfpost, selftext = thrd.selftext, domain = thrd.domain, upvotes = thrd.upvotes, comments = thrd.comments, user = thrd.user)
	s = Sentiment(topic=tpc, sentiment = sntmnt)
	t.sentiments.append(s)
	session.add(t)
	session.commit()
#	except:
#		print("Thread already in db")
def toAscii(text):
	return re.sub(r'[^\x00-\x7F]+',' ', text)
	
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-c", "--create", help="create database file (default: reddit.db)", action="store_true")
	parser.add_argument("-p", "--print", help="print contents of database", action="store_true")
	parser.add_argument("-t", "--test", help="test database functionality", action="store_true")
	args = parser.parse_args()
	if args.test:
		#print("Tests not yet implemented")
		#exit()
		#create("test.db")
		session = makeSession()
		tq = time_query(session, "Trump", 1, 1478977013, 1480396213)
		count = 0
		for thread in tq:
			count += 1
			print("Thread " + str(count))
			print(thread[1].topic + ", " + str(thread[1].sentiment) + ": [" + thread[0].subreddit + "] [" + toAscii(thread[0].title) + "] [" + str(thread[0].time) + "]")
			#for s in thread.sentiments:
			#	print(s.topic + ", " + str(s.sentiment) + ": [" + thread.subreddit + "] [" + toAscii(thread.title) + "] [" + str(thread.time) + "]")
	if args.create:
		create("reddit.db")
	if args.print:
		session = makeSession()
		count = 0
		for thread in session.query(Threads):
			count += 1
			print("Thread " + str(count))
			for s in thread.sentiments:
				print(s.topic + ", " + str(s.sentiment) + ": [" + thread.subreddit + "] [" + toAscii(thread.title) + "] [" + str(thread.time) + "]")
			#for comment in thread.comments:
			#	print(toAscii(comment.body) + "\n")
		print("Numthreads: " + str(count))
	exit()
