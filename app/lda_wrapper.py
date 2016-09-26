#!/usr/bin/python

import subprocess

import os
import numpy as np
import pandas as pd
from scipy.special import kl_div
import corpus_processor

number_of_topics = 10

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