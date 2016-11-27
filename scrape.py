import sys
import threading
import argparse
import queue
import praw

done = False
lock = threading.Lock()

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

def producerFunc(rq, queries):
	for query in queries:
		rq.put(query)
	print("no more queries")
	setDone(True)


def consumerFunc(rq, wq):
	reddit = praw.Reddit(user_agent="Sentiment Analyzer 1.0 by /u/FacialHare")
	while True:
		try:
			query = rq.get_nowait()
		except:
			#print("Read queue empty")
			if getDone():
				return
			else:
				#print("Not done yet")
				continue
		gen = makeRequest(reddit, query)
		print ("found {0} threads".format(len(list(gen))))
		if gen is not None:
			for p in gen:
				print("puting obj on queue")
				wq.put(p)

def testWriterFunc(queries, wq):
	while True:
		try:
			thread = wq.get_nowait()
		except:
			print("Write queue empty")
			if getDone():
				return
			else:
				continue
		name = string(thread.subreddit)
		f = open(name, "a")
		for comment in thread.comments:
				if type(comment).__name__ == "Comment":
					f.write(comment.body)

	

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
		numworkers = args.numworkers
	else:
		numworkers = 10
	rq = queue.Queue()
	wq = queue.Queue()
	queries = parseFile(file, delim)
	producer = threading.Thread(target=producerFunc, args = (rq, queries))
	producer.start()
	consumers = []
	for i in range(0, numworkers):
		consumer = threading.Thread(target=consumerFunc, args = (rq, wq))
		consumers.append(consumer)
		consumer.start()
	producer.join()
	print("producer thread closed")
	print((wq.qsize()))
	for c in consumers:
		c.join()
	print("Exiting main thread")


if __name__ == "__main__":
	main(sys.argv[1:])
