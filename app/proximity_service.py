import os
import numpy as np
import pandas as pd

class DistanceIndex:
    
    kl_distances = None
    js_distances = None
    index_to_article = None
    article_to_index = None

    def __init__(self):
        self.__generate_article_ids_to_array_index_lookups()
        self.__generate_distance_indexes()

    #-----------------------------
    # PUBLIC API
    #-----------------------------
    def get_closest(self, doc_id, n):
        try:
            doc_index = self.article_to_index[doc_id]

            # find closest kl
            closest_kl = self.kl_distances.iloc[doc_index]
            closest_kl.sort_values(inplace=True)
            closest_kl = closest_kl[1:n + 1]

            # find closest js
            closest_js = self.js_distances.iloc[doc_index]
            closest_js.sort_values(inplace=True)
            closest_js = closest_js[1:n + 1]

            # generate kl
            kl_results = []
            for i, kl_distance in closest_kl.iteritems():
                result = (doc_id, self.index_to_article[i], kl_distance)
                kl_results.append(result)

            # generate js
            js_results = []
            for i, js_distance in closest_js.iteritems():
                result = (doc_id, self.index_to_article[i], js_distance)
                js_results.append(result)

            return {'closest_kl': kl_results, 'closest_js': js_results}

        except Exception as e:
            return {'error': e}

    #-----------------------------
    # DISTANCE INDEXING
    #-----------------------------
    def __generate_distance_indexes(self):
        kl_index, js_index = self.__index_test_docs()
        self.kl_distances = kl_index
        self.js_distances = js_index

    def __index_test_docs(self):
        print('indexing kl and js distance pairs...')
        # load drichlets and turn into valid distribution
        gamma_path = self.__current_dir_path('/lda_lib/test_output/test_inf-gamma.dat')
        posterior_drichlets = pd.read_csv(gamma_path, sep=' ', names=[str(x) for x in range(0,20)])
        posterior_drichlets = posterior_drichlets.div(posterior_drichlets.sum(axis=1), axis=0)

        # generate nxn kl and js distances
        kl_index = []
        js_index = []

        for a in posterior_drichlets.iterrows():
            kl_row = []
            js_row = []

            for b in posterior_drichlets.iterrows():
                # kl dist
                kl_dist_val = self.__kl_dist( a[1].values,  b[1].values)
                kl_row.append(kl_dist_val)

                # js dist
                js_dist_val = self.__js_dist( a[1].values,  b[1].values)
                js_row.append(js_dist_val)

            kl_index.append(kl_row)
            js_index.append(js_row)

        # return as dataframes for easy access
        kl_index = pd.DataFrame(kl_index)
        js_index = pd.DataFrame(js_index)
        print('indexing complete')
        return kl_index, js_index


    #-----------------------------
    # MAPPING FOR ARTICLE_ID TO ARTICLE_INDEX
    #-----------------------------
    def __generate_article_ids_to_array_index_lookups(self):
        test_path = self.__current_dir_path('/data/test_arxiv.txt')
        index_to_article, article_to_index = self.__get_article_ids(test_path)
        self.index_to_article = index_to_article
        self.article_to_index = article_to_index


    def __get_article_ids(self, filename):
        article_ids = []
        for line in open(filename):
            article_id = line.lower().split()[0]
            article_ids.append(article_id)

        ids_to_idx = {}
        for i, item in enumerate(article_ids):
            ids_to_idx[item] = i

        return article_ids, ids_to_idx

    #-----------------------------
    # DISTANCE MEASURES
    #-----------------------------
    def __kl_dist(self, p, q):
        '''
        Expects 2 numpy arrays as input
        Formula from: https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
        '''
        return sum(p * np.log(p/q))

    def __js_dist(self, p, q):
        '''
        Expects 2 numpy arrays as input
        Formula from: https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence
        '''
        m = ((p + q)/2)
        return (self.__kl_dist(p, m)/2) + (self.__kl_dist(q, m)/2)

    def __current_dir_path(self, path):
        full_path = os.path.dirname(os.path.realpath(__file__))
        return full_path + path
