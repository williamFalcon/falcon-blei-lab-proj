#!/usr/bin/python

import subprocess
from gensim import corpora
from six import iteritems
from collections import Counter

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import os
from lda_lib import topics
import numpy as np

punctuation_regex = '?:!.,;-'


#-----------------------------------
# PRE PROCESSING
#-----------------------------------
def pre_process_doc(doc, stop_words, stemmer):
    tokens = []
    for token in doc:
        formatted_token = strip_end_punctuation(token)
        long_enough = len(formatted_token) > 1
        white_listed = token not in stop_words
        
        if long_enough and white_listed:
            formatted_token = stemmer.stem(formatted_token)
            tokens.append(formatted_token)
    
    return tokens
    

def get_words_to_ignore(fname, english_stop_words, apply_stopwords=True):    
    '''
    Create stop list based on content. 
    Ignore top 50 most common and all that appear only once.
    '''
    most_common_threshold = 50

    counter = Counter()
    for line in open(fname):
        tokens = line.lower().split()
        
        #ignore doc ids        
        tokens = tokens[2:]
        
        # remove punctuation and spacing
        tokens = [strip_end_punctuation(token) for token in tokens]
    
        # remove stop words if requested
        if apply_stopwords:
            tokens = [token for token in tokens if token not in english_stop_words]
        counter.update(tokens)

    # ignore most common words
    most_common_tuples = counter.most_common(most_common_threshold)
    most_common_words = map(lambda x: x[0],most_common_tuples)

    # ignore words len <= 1 or that occur only once
    single_occurence_words = [word for word, value in counter.iteritems() if value == 1 or len(word) <= 1]
    
    return list(set(most_common_words + single_occurence_words))


def strip_end_punctuation(word):
    '''
    Iterative removal of end-punctuation
    '''
    changed = True
    clean_word = word
    while changed:
        original = len(clean_word)
        clean_word = clean_word.rstrip(punctuation_regex).strip()
        changed = original != len(clean_word)
    return clean_word

def get_topic_words():
    topic_results = topics.print_topics('./lda_lib/model/final.beta', './lda_lib/dat/vocab.txt')
    print(topic_results)

def generate_stop_words(filename):
    # for our stop words we'll use standard stopwords and local context stopwords
    print('generating stopwords...')
    english_stop_words = set(stopwords.words('english'))
    stop_words = set(get_words_to_ignore(filename, english_stop_words))
    stop_words.update(english_stop_words)
    return stop_words

def generate_corpus(filename, stop_words):
    print('generating corpus for %s ' %filename)
    name = filename.split('.')[1].split('/')[-1]
    
    # generate corpus with cleaned words
    print('tokenizing %s set...' %name)
    stemmer = PorterStemmer()
    corpus = []
    for line in open(filename):
        doc = line.lower().split()[2:]
        pre_processed = pre_process_doc(doc, stop_words, stemmer)
        corpus.append(pre_processed)
    return corpus


def generate_vocab_dictionary(filename, stop_words, in_corpus=None):
    corpus = generate_corpus(filename, stop_words) if in_corpus is None else in_corpus

    # make the .vocab file
    print('writing .vocab file...')
    dictionary = corpora.Dictionary(corpus)
    tuples = [(k, v) for k, v in dictionary.iteritems()]
    tuples = sorted(tuples)
    vocab = ''
    for k,v in tuples:
        vocab += '%s\n' %(v)

    with open("./lda_lib/dat/vocab.txt", "w") as text_file:
        text_file.write(vocab)
    return dictionary


def convert_file_to_lda_c_file(filename, stop_words, dictionary, in_corpus=None):
    corpus = generate_corpus(filename, stop_words) if in_corpus is None else in_corpus

    name = filename.split('.')[1].split('/')[-1]

    print('writing %s.dat file...' %name)
    token_to_id_corpus = [dictionary.doc2bow(doc) for doc in corpus]
    corpora.BleiCorpus.serialize('./lda_lib/dat/%s.dat' %(name), token_to_id_corpus)

    # remove extra files not needed for LDA
    print('cleaning up...')
    os.remove('./lda_lib/dat/%s.dat.vocab' %name)
    os.remove('./lda_lib/dat/%s.dat.index' %name)
    print('lda-c files generated!\n\n')

#-----------------------------
# LDA
#-----------------------------
def train_c_lda(dat_file_name):
    subprocess.call("./lda_lib/lda est 0.1 20 ./lda_lib/settings.txt %s random ./lda_lib/model/" %dat_file_name, shell=True)

def inference_test_set(dat_file_name):
    subprocess.call("./lda_lib/lda inf ./lda_lib/settings.txt ./lda_lib/model/final %s ./lda_lib/test_output/test_inf" %dat_file_name, shell=True)

#-----------------------------
# DISTANCE MEASURES
#-----------------------------
def kl_dist(p, q):
    '''
    Expects 2 numpy arrays as input
    '''
    return sum(p * np.log10(p/q))

def run():
    if os.path.isfile('./lda_lib/test_output/test_inf-gamma.dat'):
        print('saved model found! Will not generate new one\nTo generate new model delete contents of ./lda_lib/test_output\n\n')
        return

    train_path = './data/train_arxiv.txt'
    test_path = './data/test_arxiv.txt'

    #build vocab
    stop_words = generate_stop_words(train_path)
    train_corpus = generate_corpus(train_path, stop_words)
    dictionary = generate_vocab_dictionary(train_path, stop_words, train_corpus)

    # convert to lda-c format
    convert_file_to_lda_c_file(train_path, stop_words, dictionary, train_corpus)
    convert_file_to_lda_c_file(test_path, stop_words, dictionary)

    #train & inference
    train_c_lda('./lda_lib/dat/train_arxiv.dat')
    inference_test_set('./lda_lib/dat/test_arxiv.dat')

def start_fake_server():
    print()
    while True:
        testVar = raw_input("Enter doc id:  ")
        print(testVar)    

if __name__ == '__main__':
    run()
    start_fake_server()