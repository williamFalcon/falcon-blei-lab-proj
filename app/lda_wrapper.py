#!/usr/bin/python

import subprocess

import os
import numpy as np
import pandas as pd
from scipy.special import kl_div
import corpus_processor

number_of_topics = 20

#--------------------------------
# PUBLIC API
#--------------------------------
def train():
    full_path = os.path.dirname(os.path.realpath(__file__))

    # don't train if we already have the fitness models
    if os.path.isfile('%s/lda_lib/test_output/test_inf-gamma.dat' % full_path):
        print('saved model found! Will not generate new one\nTo generate new model delete contents of ./lda_lib/test_output\n\n')
        return

    train_path = '%s/data/train_arxiv.txt' % full_path
    test_path = '%s/data/test_arxiv.txt' % full_path

    # convert data files to lda-c format
    corpus_processor.generate_lda_c_files(train_path, test_path)

    # train & inference
    _train_c_lda('/lda_lib/dat/train_arxiv.dat')
    _inference_test_set('/lda_lib/dat/test_arxiv.dat')


#-----------------------------
# LDA
#-----------------------------
def _current_dir_path(path):
    full_path = os.path.dirname(os.path.realpath(__file__))
    return full_path + path

def _train_c_lda(dat_file_name):
    # format cli request
    lda_bin = _current_dir_path('/lda_lib/lda')
    settings = _current_dir_path('/lda_lib/settings.txt')
    output_dir = _current_dir_path('/lda_lib/model/')
    dat_path = _current_dir_path(dat_file_name)
    cli_command = '%s est 0.1 %d %s %s random %s' % (lda_bin, number_of_topics, settings, dat_path, output_dir)

    # call c code
    subprocess.call(cli_command, shell=True)

def _inference_test_set(dat_file_name):
    # format cli command
    lda_bin = _current_dir_path('/lda_lib/lda')
    settings = _current_dir_path('/lda_lib/settings.txt')
    model_path = _current_dir_path('/lda_lib/model/final')
    inf_output = _current_dir_path('/lda_lib/test_output/test_inf')
    dat_name = _current_dir_path(dat_file_name)
    cli_command = '%s inf %s %s %s %s' %(lda_bin, settings, model_path, dat_name, inf_output)

    # call c code
    subprocess.call(cli_command, shell=True)

#-----------------------
# SERVER
#-----------------------
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
    train()
    start_fake_server()