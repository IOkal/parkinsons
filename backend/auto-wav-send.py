import wave
w = wave.open("audio-file.wav", "rb")
binary_data = w.readframes(w.getnframes())
w.close()
