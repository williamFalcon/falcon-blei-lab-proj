#!/usr/bin/python

import subprocess
from gensim import corpora
from six import iteritems
from collections import Counter

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import os
import numpy as np
import pandas as pd

from app import lda_wrapper as lda

punctuation_regex = '?:!.,;-'
number_of_topics = 20

# -----------------------------------
# PRE PROCESSING
# -----------------------------------

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
    subprocess.call("./lda_lib/lda est 0.1 %d ./lda_lib/settings.txt %s random ./lda_lib/model/" %(number_of_topics, dat_file_name), shell=True)

def inference_test_set(dat_file_name):
    subprocess.call("./lda_lib/lda inf ./lda_lib/settings.txt ./lda_lib/model/final %s ./lda_lib/test_output/test_inf" %dat_file_name, shell=True)

def train():
    # don't train if we already have the fitness models
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

def index_test_docs():
    print('indexing kl and js distance pairs...')
    # load drichlets and turn into valid distribution
    posterior_drichlets = pd.read_csv('/Users/waf/Developer/blei/lda_lib/test_output/test_inf-gamma.dat',sep=' ', names=[str(x) for x in range(0,20)])
    posterior_drichlets = posterior_drichlets.div(posterior_drichlets.sum(axis=1), axis=0)

    # generate nxn kl and js distances
    kl_index = []
    js_index = []

    for a in posterior_drichlets.iterrows():
        kl_row = []
        js_row = []

        for b in posterior_drichlets.iterrows():
            # kl dist
            kl_dist_val = kl_dist( a[1].values,  b[1].values)
            kl_row.append(kl_dist_val)

            # js dist
            js_dist_val = js_dist( a[1].values,  b[1].values)
            js_row.append(js_dist_val)

        kl_index.append(kl_row)
        js_index.append(js_row)

    # return as dataframes for easy access
    kl_index = pd.DataFrame(kl_index)
    js_index = pd.DataFrame(js_index)
    print('indexing complete')
    return kl_index, js_index


def get_article_ids(filename):
    article_ids = []
    for line in open(filename):
        article_id = line.lower().split()[0]
        article_ids.append(article_id)

    ids_to_idx = {}
    for i, item in enumerate(article_ids):
        ids_to_idx[item] = i

    return article_ids, ids_to_idx

def start_fake_server():
    # distance indexes
    kl_index, js_index = index_test_docs()

    # resolve index <--> article id
    index_to_article, article_to_index = get_article_ids('./data/test_arxiv.txt')

    while True:
        # resolve doc index
        doc_id = raw_input("Enter doc id:  ")
        doc_index =  article_to_index[doc_id]

        # find closest kl
        closest_kl = kl_index.iloc[doc_index]
        closest_kl.sort_values(inplace=True)
        closest_kl = closest_kl[1:11]

        # find closest js
        closest_js = js_index.iloc[doc_index]
        closest_js.sort_values(inplace=True)
        closest_js = closest_js[1:11]

        # print kl
        print('------------------------------')
        print('KL_closest...')
        for i, kl_distance in closest_kl.iteritems():
            print('%s %s %s' %(doc_id,index_to_article[i],kl_distance))

        # print js
        print('\n\n------------------------------')
        print('JS_closest...')
        for i, js_distance in closest_js.iteritems():
            print('%s %s %s' %(doc_id,index_to_article[i],js_distance))

#-----------------------------
# DISTANCE MEASURES
#-----------------------------
def kl_dist(p, q):
    '''
    Expects 2 numpy arrays as input
    Formula from: https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
    '''
    return sum(p * np.log(p/q))

def js_dist(p, q):
    '''
    Expects 2 numpy arrays as input
    Formula from: https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence
    '''
    m = ((p + q)/2)
    return (kl_dist(p, m)/2) + (kl_dist(q, m)/2)


if __name__ == '__main__':
    lda.train()
    # train()
    # start_fake_server()