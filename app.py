import pickle
from flask import Flask, request, jsonify, app

import numpy as np
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']

    new_data = list(data.values())
    output = model.predict([new_data])
    #print(output[0])
    return jsonify(output[0])


if __name__ == '__main__':
    app.run(debug=True)