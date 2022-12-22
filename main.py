from flask import Flask, request
from flask_cors import CORS, cross_origin
import urllib.request, json 

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
   url = "".join(['http://127.0.0.1:8080/api/v1/predict/?url=',url])
   with urllib.request.urlopen(url) as url:
    data = json.load(url)
    return(data)

if __name__ == '__main__':
   app.run(host='127.0.0.1', port = 5000, debug = True)