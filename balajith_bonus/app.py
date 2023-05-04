from flask import Flask, render_template, request
import pickle
import numpy as np
from model import Net
import torch

# Load the pickled model
with open(r'C:\Users\BALAJI\Music\ML_Alina\app\model.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve user input from form data
    f1 = float(request.form['f1'])
    f2 = float(request.form['f2'])
    f3 = float(request.form['f3'])
    f4 = float(request.form['f4'])
    f5 = float(request.form['f5'])
    f6 = float(request.form['f6'])
    f7 = float(request.form['f7'])

    # Create a tensor with user input
    input_data = torch.tensor([f1, f2, f3, f4, f5, f6, f7])

    # Reshape tensor to match model's input shape
    input_data = torch.reshape(input_data, (1, 7))

    # Use the model to make predictions on the input data
    prediction = model(input_data)

    # Format the prediction as a string
    if prediction.item()< 0.5:
        output = "Class 0"
    else:
        output = "Class 1"

    # Return the prediction to the user
    return render_template('index.html', prediction_text='Prediction: {}'.format(output))


if __name__ == '__main__':
    app.run(debug=True)
