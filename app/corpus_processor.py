from collections import Counter
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from gensim import corpora
import os

punctuation_regex = '?:!.,;-'
LDA_DAT_PATH = '/lda_lib/dat/'

#-----------------------------
# PUBLIC API
#-----------------------------
def generate_lda_c_files(train_path, test_path):
    #build vocab
    stop_words = _generate_stop_words(train_path)
    train_corpus = _generate_corpus(train_path, stop_words)
    dictionary = _generate_vocab_dictionary(train_path, stop_words, train_corpus)

    # convert to lda-c format
    _convert_file_to_lda_c_file(train_path, stop_words, dictionary, train_corpus)
    _convert_file_to_lda_c_file(test_path, stop_words, dictionary)

#-----------------------------
# STOP WORDS
#-----------------------------
def _generate_stop_words(filename):
    '''
    Generates stopwords from english stopwords list (nltk) and
    words derived from the given corpus
    '''
    # for our stop words we'll use standard stopwords and local context stopwords
    print('generating stopwords...')
    english_stop_words = set(stopwords.words('english'))
    stop_words = set(_generate_stop_words_from_docset(filename, english_stop_words))
    stop_words.update(english_stop_words)
    return stop_words

def _generate_stop_words_from_docset(fname, english_stop_words, apply_stopwords=True):    
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
        tokens = [_strip_end_punctuation(token) for token in tokens]
    
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

#-----------------------------
# CORPUS INTERFACE
#-----------------------------
def _generate_corpus(filename, stop_words):
    '''
    Generates corpus for a given set of docs
    '''
    print('generating corpus for %s ' %filename)
    name = filename.split('.')[0].split('/')[-1]
    
    # generate corpus with cleaned words
    print('tokenizing %s set...' %name)
    stemmer = PorterStemmer()
    corpus = []
    for line in open(filename):
        doc = line.lower().split()[2:]
        pre_processed = _pre_process_doc(doc, stop_words, stemmer)
        corpus.append(pre_processed)
    return corpus

def _generate_vocab_dictionary(filename, stop_words, in_corpus=None):
    '''
    Generates a dictionary of the vocab mapping for the given corpus
    '''
    corpus = _generate_corpus(filename, stop_words) if in_corpus is None else in_corpus

    # make the .vocab file
    print('writing .vocab file...')
    dictionary = corpora.Dictionary(corpus)
    tuples = [(k, v) for k, v in dictionary.iteritems()]
    tuples = sorted(tuples)
    vocab = ''
    for k,v in tuples:
        vocab += '%s\n' %(v)

    full_path = os.path.dirname(os.path.realpath(__file__)) + LDA_DAT_PATH
    with open("%svocab.txt" %(full_path), "w") as text_file:
        text_file.write(vocab)
    return dictionary

#-----------------------------
# LDA-C FORMATTER
#-----------------------------
def _convert_file_to_lda_c_file(filename, stop_words, dictionary, in_corpus=None):
    '''
    Converts a filename to lda-c format
    Prints out a file.dat under /dat directory
    '''
    corpus = _generate_corpus(filename, stop_words) if in_corpus is None else in_corpus
    full_path = os.path.dirname(os.path.realpath(__file__))

    name = filename.split('.')[0].split('/')[-1]
    full_filename = full_path + LDA_DAT_PATH + name

    print('writing %s.dat file...' %name)
    token_to_id_corpus = [dictionary.doc2bow(doc) for doc in corpus]
    corpora.BleiCorpus.serialize('%s.dat' %(full_filename), token_to_id_corpus)

    # remove extra files not needed for LDA
    print('cleaning up...')
    os.remove('%s.dat.vocab' %(full_filename))
    os.remove('%s.dat.index' %(full_filename))
    print('lda-c files generated!\n\n')


#-----------------------------------
# PRE PROCESSING UTILS
#-----------------------------------
def _pre_process_doc(doc, stop_words, stemmer):
    '''
    Applies pre-processing steps to a document
    Currently:
    - Removes end of word punctuation
    - Removes words where len(word) <= 1
    - Removes words in stop_list
    - Stems
    '''
    tokens = []
    for token in doc:
        formatted_token = _strip_end_punctuation(token)
        long_enough = len(formatted_token) > 1
        white_listed = token not in stop_words
        
        if long_enough and white_listed:
            formatted_token = stemmer.stem(formatted_token)
            tokens.append(formatted_token)
    
    return tokens
    

def _strip_end_punctuation(word):
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

