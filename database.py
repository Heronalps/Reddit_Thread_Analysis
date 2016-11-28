from sqlalchemy import Table, Column, ForeignKey, BigInteger, Integer, String, PickleType, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy import create_engine
from sqlalchemy.sql import exists

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
	
if __name__ == "__main__":
	create()
	exit()