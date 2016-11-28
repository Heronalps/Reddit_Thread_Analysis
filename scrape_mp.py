import sys
import multiprocessing as mp
import argparse
import queue
import praw

done = False
lock = mp.Lock()

class Thread:
	def __init__(self, thread):
		self.subreddit = thread.subreddit
		self.threadid = thread.id
		self.title = thread.title
		self.time = thread.created_utc
		self.domain = thread.domain
		self.upvotes = thread.ups
		self.comments = thread.comments
		self.selftext = thread.selftext
		self.selfpost = thread.is_self
		self.user = thread.author.name
			
		
		
	def addComment(self, comment):
		self.comments.append(comment)

class Query:
	def __init__(self, sub, start, end):
		# subreddit (string)
		self.sub = sub
		# start time (time)
		self.start = start
		# end time (time)
		self.end = end


def setDone(val):
	global done
	lock.acquire()
	done = val
	lock.release()

def getDone():
	global done
	lock.acquire()
	d = done
	lock.release()
	return d

def parseFile(file, delim):
	with open(file) as f:
		lines = f.readlines()
	return [Query(l.split(delim)[0],l.split(delim)[1],l.split(delim)[2]) for l in lines]

def makeRequest(reddit, query):
	srch = "timestamp:" + query.start + ".." + query.end
	return reddit.search(srch, subreddit=query.sub, sort='top',syntax='cloudsearch')

def producerFunc(numworkers, rq, queries):
	for query in queries:
		rq.put(query)
	print("no more queries")
	for i in range(numworkers):
		rq.put("STOP")




def consumerFunc(rq, wq):
	reddit = praw.Reddit(user_agent="Sentiment Analyzer 1.0 by /u/FacialHare")
	while True:
		if (rq.empty() == False):
			query = rq.get()
			if (query == "STOP"):
				return
			gen = makeRequest(reddit, query)
			#print ("found {0} threads".format(len(list(gen))))
			if gen is not None:
				for p in gen:
					thread = Thread(p)
					print("puting obj on queue")
					wq.put(thread)

def testWriterFunc(wq):
	print("set up test")
	while True:
		if (wq.empty() == False):
			thread = wq.get()
			if (thread == "STOP"):
				return
			#print("grab from queue")
			name = str(thread.subreddit)
			f = open(name, "a")
			for comment in thread.comments:
					if type(comment).__name__ == "Comment":
						f.write(comment.body)
						f.write("\n")
			f.close()

	

def main(argv):
	parser = argparse.ArgumentParser()
	parser.add_argument("file", help="name of file with list of queries. file should have format sub starttime endtime (split by delimiter)")
	parser.add_argument("-d", "--delim", help="delimiter used in query file (default: \' \')")
	parser.add_argument("-n", "--numworkers", help="number of worker threads (default: 10)")
	parser.add_argument("-t", "--test", help="tests opperation and writes data to file")
	args = parser.parse_args()
	
	file = args.file
	if args.delim:
		delim = args.delim
	else:
		delim = " "
	if args.numworkers:
		numworkers = int(args.numworkers)
	else:
		numworkers = 10
	rq = mp.Queue()
	wq = mp.Queue()
	queries = parseFile(file, delim)
	producer = mp.Process(target=producerFunc, args = (numworkers, rq, queries))
	producer.start()
	if(args.test):
		tester = mp.Process(target=testWriterFunc, args = (wq,))
		tester.start()
	consumers = []
	for i in range(0, numworkers):
		consumer = mp.Process(target=consumerFunc, args = (rq, wq))
		consumers.append(consumer)
		consumer.start()
	producer.join()
	print("producer thread closed")
	print((wq.qsize()))
	for c in consumers:
		c.join()
	wq.put("STOP")
	if(args.test):
		tester.join()
	print("Exiting main thread")
if __name__ == "__main__":
	main(sys.argv[1:])
