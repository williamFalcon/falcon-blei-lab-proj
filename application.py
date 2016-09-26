#!/usr/bin/python
import flask
from flask import Flask, Response, request, current_app, jsonify

from app import lda_wrapper as lda
from app.proximity_service import DistanceIndex

application = Flask(__name__)
def start_cli_server():
    idx = DistanceIndex()

    while True:
        # resolve doc index
        doc_id = raw_input("Enter doc id:  ")
        results = idx.get_closest(doc_id, 10)
        if 'error' in results:
            print('error. Enter a valid document id')
            print(results['error'])
        else:
            print('-----------------------------')
            print('KL RESULTS')
            for i, kl in enumerate(results['closest_kl']):
                print('%s. %s %s %s' %(i+1, kl[0], kl[1], kl[2]))
        
            print('\n\n-----------------------------')
            print('JS RESULTS')
            for i, js in enumerate(results['closest_js']):
                print('%s. %s %s %s' %(i+1, js[0], js[1], js[2]))

            print('\n\n')


if __name__ == '__main__':
    lda.train()

    application.run(host='0.0.0.0')
    
    start_cli_server()