import keras.applications.xception as xception
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from read import predict

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
   app.run(debug = True)