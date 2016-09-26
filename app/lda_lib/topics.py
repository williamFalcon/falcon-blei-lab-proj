#! /usr/bin/python

# usage: python topics.py <beta file> <vocab file> <num words>
#
# <beta file> is output from the lda-c code
# <vocab file> is a list of words, one per line
# <num words> is the number of words to print from each topic

import sys

def print_topics(beta_file, vocab_file, nwords = 25):

    # get the vocabulary

    vocab = file(vocab_file, 'r').readlines()
    # vocab = map(lambda x: x.split()[0], vocab)
    vocab = map(lambda x: x.strip(), vocab)

    # for each line in the beta file

    indices = range(len(vocab))
    topic_no = 0
    topics = []
    for topic in file(beta_file, 'r'):
        topic = map(float, topic.split())
        indices.sort(lambda x,y: -cmp(topic[x], topic[y]))
        topic_words = []
        for i in range(nwords):
            topic_words.append(vocab[indices[i]])
        topics.append((topic_no, topic_words))
        topic_no = topic_no + 1
    return topics

if (__name__ == '__main__'):

    if (len(sys.argv) != 4):
       print 'usage: python topics.py <beta-file> <vocab-file> <num words>\n'
       sys.exit(1)

    beta_file = sys.argv[1]
    vocab_file = sys.argv[2]
    nwords = int(sys.argv[3])
    print_topics(beta_file, vocab_file, nwords)
