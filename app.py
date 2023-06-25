from flask import Flask, render_template, request, jsonify
import json

from search_code import searchCode

from flask_cors import CORS, cross_origin
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'




@app.route('/upload', methods=['POST','GET'])
@cross_origin()
def upload():
    text = json.loads(request.get_data())
    print(text['text'])

    prediction = searchCode(text['text'])
    print(prediction)
    return jsonify(prediction)

if __name__ == "__main__":
    app.run()