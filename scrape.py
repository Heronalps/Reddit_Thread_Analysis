import sys
import threading
import argparse

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
	lock.acquire()
	done = val
	lock.release()

def getDone():
	lock.acquire()
	d = done
	lock.release()
	return d

def parseFile(file, delim):
	with open(file) as f:
		lines = f.readlines()
	return [Query(l.split(delim)[0],l.split(delim)[1],l.split(delim)[2]) for l in lines]

def makeRequest(query):
	# TODO: implement python api call for subreddit sub from start time to end time
	return None

def producerFunc(rq, queries):
	for query in queries:
		rq.put(query)
	setDone(True)


def consumerFunc(rq, wq):
	while True:
		try:
			query = rq.get_nowait()
		except:
			print("Read queue empty")
			if getDone():
				return
			else:
				continue
		res = makeRequest(query)
		if res is not None:
			wq.put(res)
			

def main(argv):
	parser = argparse.ArgumentParser()
	parser.add_argument("file", help="name of file with list of queries. file should have format sub starttime endtime (split by delimiter)")
	parser.add_argument("-d", "--delim", help="delimiter used in query file (default: \' \')")
	parser.add_argument("-n", "--numworkers", help="number of worker threads (default: 10)")
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
	rq = Queue()
	wq = Queue()
	queries = parseFile(file, delim)
	producer = threading.Thread(target=producerFunc, args = (rq, queries))
	producer.start()
	consumers = []
	for i in range(0, numworkers):
		consumer = threading.Thread(target=consumerFunc, args = (rq, wq))
		consumers.append(consumer)
		consumer.start()
	producer.join()
	for c in consumers:
		c.join()
	print("Exiting main thread")


if __name__ == "__main__":
	main(argv[1:])
