#!/usr/bin/python
import flask
from flask import Flask, Response, request, current_app, jsonify
import json

from app import lda_wrapper as lda
from app.proximity_service import DistanceIndex
import sys

application = Flask(__name__)
number_of_topics = 200

def start_cli_server():
    idx = DistanceIndex(number_of_topics)

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

#-------------------------
# REST API
#-------------------------
@application.route('/api/v1.0/status')
def status():
    data = {'status': 'ok', 'version': 1.0, 'description': 'REST API for blei project'}
    js = json.dumps(data)
    resp = Response(js, status=200, mimetype='application/json')
    return resp

@application.route('/api/v1.0/closest_to_doc')
def closest_to_doc():
    doc_id = request.args.get('doc_id')
    results = application.distance_index.get_closest(doc_id, 10)
    print(results)
    js = json.dumps({'doc_id': doc_id, 'results': results})

    resp = Response(js, status=200, mimetype='application/json')
    return resp

if __name__ == '__main__':
    print('training model... Will train once and use cached model thereafter')
    lda.train(number_of_topics)

    if len(sys.argv) == 2:
        with application.app_context():
            application.distance_index = DistanceIndex(number_of_topics)

        application.run(host='0.0.0.0')
    else:
        start_cli_server()