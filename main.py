from flask import Flask, request, jsonify
import os
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the trained RNN model from the pickled file
with open('best_model.pkl', 'rb') as f:
    model_rnn = pickle.load(f)

# Define column names based on your dataset
columns = ['A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score', 'A6_Score',
           'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score', 'age', 'gender',
           'ethnicity', 'jundice', 'austim', 'contry_of_res', 'used_app_before',
           'result', 'age_desc', 'relation', 'Class/ASD']

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    # Create a DataFrame from the input data
    input_df = pd.DataFrame(data, index=[0])

    # Perform label encoding for categorical columns
    encoder = LabelEncoder()
    for col in input_df.columns:
        if col in ['gender', 'ethnicity', 'jundice', 'austim', 'contry_of_res', 'used_app_before', 'relation', 'Class/ASD']:
            input_df[col] = encoder.fit_transform(input_df[col])

    # Drop the 'age_desc' column
    input_df = input_df.drop('age_desc', axis=1)

    # Assuming 'scaler' is the trained StandardScaler
    # Standardize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(input_df)

    # Reshape the input data for RNN
    X_rnn = X.reshape((X.shape[0], X.shape[1], 1))

    # Make predictions using the model
    predictions = model_rnn.predict(X_rnn)

    return jsonify(predictions.tolist())


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(port=port)

