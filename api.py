from flask import Flask, jsonify, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

pipeline = joblib.load(open('models/lr_model_pipeline_retrained_from_checkpoint.joblib', 'rb'))
pipeline = pipeline['pipeline']


@app.route('/api', methods=['POST'])

def predict():
    data = request.get_json(force=True)
    # use the threshold we identified works the best through experimenation
    pipeline['threshold'] = threshold
    data_df = pd.DataFrame(data)
    probability = model.predict_proba(data_df)
    prediction = [1 if pred >= threshold else 0 for pred in probability[:, 1]]
    return jsonify({'prediction': str(prediction)})

if __name__ == '__main__':
	app.run(port=8080, debug=True)