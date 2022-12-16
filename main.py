from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from read import predict
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/')
@cross_origin()
def index():
   return "HELLO"
   
@app.route('/api/v1/predict/')
@cross_origin()
def process():
   url = request.args.get('url')
   class_name , confidence_score = predict(url)
   class_name = class_name[2:-1]
   confidence_score = confidence_score[:5]
   return jsonify(class_name = class_name, confidence_score = confidence_score)

if __name__ == '__main__':
   app.run(host='127.0.0.1', port = 8080, debug = True)