import os
import json
from tensorflow.python.lib.io import file_io
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Sequential
from fundamental_frequency import calculate_fundamental_frequency_features
from feature_engineering import engineer_features
import numpy as np
import scipy
import parselmouth
import wave # new import for binary conversion
from flask import Flask, request
from google.cloud import storage

MODEL_BUCKET = os.environ['MODEL_BUCKET']
MODEL_FILENAME = os.environ['MODEL_FILENAME']
MODEL = None

app = Flask(__name__)


def create_model():
    # Create a neural network
    model = Sequential()
    model.add(Input((15,)))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


@app.before_first_request
def _load_model():
    global MODEL
    model_file = file_io.FileIO('gs://'+MODEL_BUCKET+'/'+MODEL_FILENAME, mode='rb')
    temp_model_location = './temp_model.h5'
    temp_model_file = open(temp_model_location, 'wb')
    temp_model_file.write(model_file.read())
    temp_model_file.close()
    model_file.close()
    MODEL = create_model()
    MODEL.load_weights(temp_model_location)

'''
def download_wav(file_name):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket("voice-audio")
    blob = bucket.blob(file_name)
    file_path = "/audio/%s" % file_name
    open(file_path, 'a').close()  # Create an empty file at path
    blob.download_to_filename(file_path)
    return file_path
'''


@app.route('/predict', methods=['POST'])
def predict():
    # Get the WAV file name from the request. Must include the .wav extension.
    binary_file_data = request.get_json()['file']

    packedData = map(lambda v:struct.pack('h',v), binary_file_data)
    frames = b''.join(packedData)
    wav_file = output_wave('output.wav', frames)

    # Download the sound file from gcp
        # sound_file_path = download_wav(file_name)
    sound_file = scipy.io.wavfile.read(wav_file) # replace sound_file_path with wav_file
    sound = parselmouth.Sound(wav_file)

    # Calculate features
    fundamental_frequency_features = calculate_fundamental_frequency_features(sound_file)
    other_features = engineer_features(sound)

    # Concatenate features in the order the model expects, then make a prediction.
    model_input = np.concatenate(fundamental_frequency_features, other_features)
    prediction_array = MODEL.predict(model_input)

    # We only process one sound file so there should only be one prediction to return.
    prediction = prediction_array[0]

    return json.dumps({'prediction': prediction, 'features': model_input.tolist()}), 200


if __name__ == '__main__':
    # This is used when running locally. Gunicorn is used to run the
    # application on Google App Engine. See entrypoint in app.yaml.
    app.run(host='127.0.0.1', port=8080, debug=True)
