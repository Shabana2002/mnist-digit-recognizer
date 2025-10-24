# MNIST Digit Recognizer

## Description
A simple CNN-based deep learning project to recognize handwritten digits (0-9). Users can upload a digit image and get prediction with confidence score.

## How to Run
1. Install dependencies:
pip install -r requirements.txt
2. Run Streamlit app:
streamlit run app.py
3. Open the link in your browser and upload a digit image.

## Folder Structure
- app.py : Streamlit web app
- model/mnist_cnn.h5 : Trained CNN model
- utils/predict.py : Prediction helper functions