
from pickle import load
from flask import Flask, jsonify, request
import pandas as pd
model = load(open('model.pkl', 'rb'))
scaler = load(open('scaler.pkl', 'rb'))
column = load(open('columns.pkl', 'rb'))

app = Flask(__name__)
# routes
@app.route('/', methods=['POST'])

def predict():
    json = request.get_json(force=True)
    dft = pd.DataFrame(json, index = [0])
    for c in column['cat columns']:
        dft[c] = dft[c].astype('int64')
    x = pd.get_dummies(dft, columns = column['cat columns'])
    cols = set(column['columns']) - set(x.columns)
    for c in cols:
        x[c] = 0
    x[column['scalar']] = scaler.transform(x[column['scalar']])
    x = x[column['columns']]
    result = model.predict(x)
    output = {'results': int(result[0])}
    return jsonify(results=output)
if __name__ == '__main__':
    app.run(port = 5000, debug=True)

