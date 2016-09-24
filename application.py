#!/usr/bin/python

import subprocess
from gensim import corpora
from six import iteritems
from collections import Counter

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import os

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


def generate_c_lda_dat_file():
    print('parsing training corpus to lda-c input files...')

    # for our stop words we'll use standard stopwords and local context stopwords
    print('generating stopwords...')
    english_stop_words = set(stopwords.words('english'))
    stop_words = set(get_words_to_ignore('train_arxiv.txt', english_stop_words))
    stop_words.update(english_stop_words)
    
    # generate corpus with cleaned words
    print('tokenizing training set...')
    stemmer = PorterStemmer()
    corpus = []
    for line in open('train_arxiv.txt'):
        doc = line.lower().split()[2:]
        pre_processed = pre_process_doc(doc, stop_words, stemmer)
        corpus.append(pre_processed)
    
    print('writing .dat file...')
    dictionary = corpora.Dictionary(corpus)

    token_to_id_corpus = [dictionary.doc2bow(doc) for doc in corpus]
    corpora.BleiCorpus.serialize('./lda-lib/corpus.dat', token_to_id_corpus)

    # make the .vocab file
    print('writing .vocab file...')
    tuples = [(k, v) for k, v in dictionary.iteritems()]
    tuples = sorted(tuples)
    vocab = ''
    for k,v in tuples:
        vocab += '%s\n' %(v)

    with open("./lda-lib/vocab.txt", "w") as text_file:
        text_file.write(vocab)

    # remove extra files not needed for LDA
    print('cleaning up...')
    os.remove('./lda-lib/corpus.dat.vocab')
    os.remove('./lda-lib/corpus.dat.index')
    print('lda-c files generated!\n\n')

def run():
    generate_c_lda_dat_file()


if __name__ == '__main__':
    run()

#subprocess.call("./lda est 0.1 20 settings.txt ap.dat random ./")