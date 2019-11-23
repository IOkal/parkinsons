import os
import json
from tensorflow.keras.models import load_model
from fundamental_frequency import calculate_fundamental_frequency_features
import numpy as np
import scipy
from flask import Flask, request
from google.cloud import storage

MODEL_BUCKET = os.environ['MODEL_BUCKET']
MODEL_FILENAME = os.environ['MODEL_FILENAME']
MODEL = None

app = Flask(__name__)


@app.before_first_request
def _load_model():
    global MODEL
    client = storage.Client()
    bucket = client.get_bucket(MODEL_BUCKET)
    blob = bucket.get_blob(MODEL_FILENAME)
    s = blob.download_as_string()
    MODEL = load_model(s)


def download_wav(file_name):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket("voice-audio")
    blob = bucket.blob(file_name)
    file_path = "/audio/%s.wav" % file_name
    blob.download_to_filename(file_path)
    return scipy.io.wavfile.read(file_path)


@app.route('/predict', methods=['POST'])
def predict():
    # Get the WAV file name from the request. Must include the .wav extension.
    file_name = request.get_json()['file_name']

    # Download the sound file from gcp
    sound_file = download_wav(file_name)

    # Calculate features
    fundamental_frequency_features = calculate_fundamental_frequency_features(sound_file)

    # Concatenate features in the order the model expects, then make a prediction.
    model_input = np.concatenate(fundamental_frequency_features)
    prediction_array = MODEL.predict(model_input)

    # We only process one sound file so there should only be one prediction to return.
    prediction = prediction_array[0]

    return json.dumps({'prediction': prediction}), 200
