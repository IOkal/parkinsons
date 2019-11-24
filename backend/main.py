import os
import json
from tensorflow.python.lib.io import file_io
from tensorflow.keras.models import load_model
from fundamental_frequency import calculate_fundamental_frequency_features
from feature_engineering import engineer_features
import numpy as np
import scipy
import parselmouth
from flask import Flask, request
from google.cloud import storage

MODEL_BUCKET = os.environ['MODEL_BUCKET']
MODEL_FILENAME = os.environ['MODEL_FILENAME']
MODEL = None

app = Flask(__name__)


@app.before_first_request
def _load_model():
    global MODEL
    model_file = file_io.FileIO('gs://'+MODEL_BUCKET+'/'+MODEL_FILENAME, mode='rb')
    temp_model_location = './temp_model.h5'
    temp_model_file = open(temp_model_location, 'wb')
    temp_model_file.write(model_file.read())
    temp_model_file.close()
    model_file.close()
    MODEL = load_model(temp_model_location)


def download_wav(file_name):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket("voice-audio")
    blob = bucket.blob(file_name)
    file_path = "/audio/%s.wav" % file_name
    blob.download_to_filename(file_path)
    return file_path


@app.route('/predict', methods=['POST'])
def predict():
    # Get the WAV file name from the request. Must include the .wav extension.
    file_name = request.get_json()['file_name']

    # Download the sound file from gcp
    sound_file_path = download_wav(file_name)
    sound_file = scipy.io.wavfile.read(sound_file_path)
    sound = parselmouth.Sound(sound_file_path)

    # Calculate features
    fundamental_frequency_features = calculate_fundamental_frequency_features(sound_file)
    other_features = engineer_features(sound)

    # Concatenate features in the order the model expects, then make a prediction.
    model_input = np.concatenate(fundamental_frequency_features, other_features)
    prediction_array = MODEL.predict(model_input)

    # We only process one sound file so there should only be one prediction to return.
    prediction = prediction_array[0]

    return json.dumps({'prediction': prediction}), 200


if __name__ == '__main__':
    # This is used when running locally. Gunicorn is used to run the
    # application on Google App Engine. See entrypoint in app.yaml.
    app.run(host='127.0.0.1', port=8080, debug=True)
