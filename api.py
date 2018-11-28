from flask import Flask, request, send_from_directory, redirect, render_template, flash, url_for, jsonify, \
    make_response, abort
import os
import sys
import numpy as np
import time
import datetime

#firebase
import pyrebase
config = {
  "apiKey": "",
  "authDomain": "matar-184107.firebaseapp.com",
  "databaseURL": "https://matar.firebaseio.com",
  "storageBucket": "matar-184107.appspot.com",
  "serviceAccount": "matar-184107-firebase-adminsdk-.json"
}
firebase = pyrebase.initialize_app(config)

db = firebase.database()

def main():
    current_dir = os.path.dirname(__file__)
    sys.path.append(os.path.join(current_dir, '..'))
    current_dir = current_dir if current_dir is not '' else '.'

    from keras_sentiment_analysis.library.lstm import WordVecBidirectionalLstmSoftmax

    app = Flask(__name__)
    app.config.from_object(__name__)  # load config from this file , flaskr.py

    # Load default config and override config from an environment variable
    app.config.from_envvar('FLASKR_SETTINGS', silent=True)
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

    bidirectional_lstm_softmax_c = WordVecBidirectionalLstmSoftmax()


    @app.route('/')
    def home():
        return "Text Sentiment API"


    @app.route('/predict', methods=['POST', 'GET'])
    def bidirectional_lstm_softmax():
        if request.method == 'POST':
            if 'sentence' not in request.form:
                flash('No sentence post')
                redirect(request.url)
            elif request.form['sentence'] == '':
                flash('No sentence')
                redirect(request.url)
            else:
                sent = request.form['sentence']
                classId = request.form['classId']
                sentiments = bidirectional_lstm_softmax_c.predict(sent)
                array_sentiment = np.array(sentiments)
                result = array_sentiment.tolist()
                result_class = bidirectional_lstm_softmax_c.predict_class(sent)
                ts = time.time()
                timenow = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
                data = {
                    "sentence": sent,
                    "result": result_class,
                    "neutral": result[4],
                    "audio": result[2],
                    "positive": result[0],
                    "video": result[1],
                    "negative": result[3],
                    "time_created": timenow
                }
                db.child("chat_data").child(classId).push(data)
                return jsonify(data)

    @app.errorhandler(404)
    def not_found(error):
        return make_response(jsonify({'error': 'Not found'}), 404)

    model_dir_path = os.path.join(current_dir, '../demo/testmodel')

    bidirectional_lstm_softmax_c.load_model(model_dir_path)

    bidirectional_lstm_softmax_c.test_run('this platform sucks')
    app.run(host = '0.0.0.0',port=5005)

    app.run(debug=True, use_reloader=False)


if __name__ == '__main__':
    main()
