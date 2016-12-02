from sqlalchemy import Table, Column, ForeignKey, BigInteger, Integer, String, PickleType, Boolean, Text
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
	topic = Column(String(255), primary_key=True)
	sentiment = Column(Integer)
	assigned_label = Column(Integer)
	
	title = Column(Text)
	time = Column(BigInteger)
	subreddit = Column(String(255))
	selfpost = Column(Boolean)
	selftext = Column(Text)
	domain = Column(String(255))
	upvotes = Column(Integer)
	comments = Column(PickleType)
	user = Column(String(255))
	
def create():
	engine = create_engine('sqlite:///reddit.db')
	Base.metadata.create_all(engine)
	
def query(session, topic, sentiment):
	threads = []
	for thread in session.query(Threads).filter(Threads.topic==topic).filter(Threads.sentiment == sentiment):
		threads.append(thread)
	return threads

def time_query(session, topic, sentiment, start_time, end_time):
	threads = []
	for thread in session.query(Threads).\
	filter(Threads.topic==topic).filter(Threads.sentiment == sentiment).\
	filter(Threads.time >= start_time).filter(Threads.time < end_time):
		threads.append(thread)
	return threads

def title_query(session, start_time, end_time):
	threads = []
	for thread in session.query(Threads).\
	filter(Threads.time >= start_time).filter(Threads.time < end_time):
		threads.append(thread)
	return threads

def subreddit_query(session, subreddit, start_time, end_time):
	threads = []
	for thread in session.query(Threads).\
	filter(Threads.subreddit == subreddit).\
	filter(Threads.time >= start_time).filter(Threads.time < end_time):
		threads.append(thread)
	return threads
	
def makeSession():
	engine = create_engine('sqlite:///reddit.db')
	Base.metadata.bind = engine
	DBSession = sessionmaker(bind = engine)
	session = DBSession()
	return session
	
def addThread(session, tpc, sntmnt, thrd):
	if session.query(Threads).filter(Threads.threadid == thrd.threadid).count():
		print("Thread already in db")
		return
	t = Threads(threadid = thrd.threadid, topic=tpc, sentiment = sntmnt, title = thrd.title, time = thrd.time, subreddit = thrd.subreddit, selfpost = thrd.selfpost, selftext = thrd.selftext, domain = thrd.domain, upvotes = thrd.upvotes, comments = thrd.comments, user = thrd.user)
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
	args = parser.parse_args()
	if args.create:
		create()
	if args.print:
		session = makeSession()
		count = 0
		for thread in session.query(Threads):
			count += 1
			print(thread.topic + ", " + str(thread.sentiment) + ": [" + thread.subreddit + "] [" + toAscii(thread.title) + "] [" + str(thread.time) + "]")
			#for comment in thread.comments:
			#	print(toAscii(comment.body) + "\n")
		print("Numthreads: " + str(count))
	exit()