import argparse
from scrape_mp import Query


def parseFile(file, delim):
	with open(file) as f:
		lines = f.readlines()
	queries = []
	for l in lines:
		topic = l.split(delim)[0]
		sentiment = l.split(delim)[1]
		sub = l.split(delim)[2]
		start = int(l.split(delim)[3])
		end = int(l.split(delim)[4])
		blocksize = int(l.split(delim)[5])
		while start < end:
			queries.append(Query(topic, sentiment, sub, start, start + blocksize))
			start = start + blocksize
	return queries


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("infile", help="input file (topic sentiment subreddit start end blocksize)")
	parser.add_argument("outfile", help="output file (topic sentiment subreddit start end)")
	parser.add_argument("-d", "--delim", help="delimiter used in query file (default: \' \')")
	args = parser.parse_args()

	delimiter = " "
	if args.delim:
		delimiter = args.delim
	infile = args.infile
	queries = parseFile(infile, delimiter)
	
	outfile = open(args.outfile, 'w')
	for q in queries:
		l = q.topic + " " + str(q.sentiment) + " " + q.sub + " " + str(q.start) + " " + str(q.end) + "\n"
		outfile.write(l)
	outfile.close()