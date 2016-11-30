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
	topic = Column(String(255))
	sentiment = Column(Integer)
	
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
	
def makeSession():
	engine = create_engine('sqlite:///reddit.db')
	Base.metadata.bind = engine
	DBSession = sessionmaker(bind = engine)
	session = DBSession()
	return session
	
def addThread(session, tpc, sntmnt, thrd):
	t = Threads(threadid = thrd.threadid, topic=tpc, sentiment = sntmnt, title = thrd.title, time = thrd.time, subreddit = thrd.subreddit, selfpost = thrd.selfpost, selftext = thrd.selftext, domain = thrd.domain, upvotes = thrd.upvotes, comments = thrd.comments, user = thrd.user)
	session.add(t)
	session.commit()
	
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
		for thread in session.query(Threads):
			print(thread.topic + ", " + str(thread.sentiment) + ": [" + thread.subreddit + "] [" + thread.title + "] [" + str(thread.time) + "]")
			#for comment in thread.comments:
			#	print(toAscii(comment.body) + "\n")
	exit()