import pandas as pd
import numpy as np
import torch

from flask import Flask, request, jsonify, render_template

model_path = 'Candidate_Recommender_model.pth'

with open(model_path, 'rb') as file:
    model = torch.load(file)


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    inputs = torch.tensor(data['features']).float()
    output = model(inputs).detach().numpy().tolist()
    return jsonify({'prediction': output})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)