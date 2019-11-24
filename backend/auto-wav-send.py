import wave
w = wave.open("parkinsons.wav", "rb")
binary_data = w.readframes(w.getnframes())
w.close()
