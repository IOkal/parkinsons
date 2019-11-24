import numpy as np
import matplotlib.pyplot as plt


def calculate_fundamental_frequency_features(sound_file):
    i = sound_file[1].size

    x = sound_file[1]

    v1 = np.arange(float (i)/float(2))
    c = np.c_[v1, x]

    cc = c.T  # Transpose

    x2 = cc[2]

    stop = (float(i)/float(2))
    step_time = 1.25
    step = int(step_time*sound_file[0])
    intervals = np.arange(0, int(stop), step)

    # Chop up the time series
    fundamental_frequencies = np.array([])

    for delta_t in intervals:
        x_part = x2[delta_t:delta_t+step]

        # Sonogram
        Pxx, freqs, bins, im = plt.specgram(x_part, NFFT=int(sound_file[0]*0.008), Fs=sound_file[0], noverlap =int(sound_file[0]*0.005))

        # Filtering freq in the sonogram
        Pxx = Pxx[(freqs >= 100) & (freqs <= 8000)]
        freqs = freqs[(freqs >= 100) & (freqs <= 8000)]
        Pxx_transposed = Pxx.T
        max_intensity = []
        frec_max = []

        # Find the fundamental frequencies
        for j, element in enumerate(Pxx_transposed):
            max_intensity_index = np.argmax(element)
            max_intensity.append(element[max_intensity_index])
            frec_max.append(freqs[max_intensity_index])

        # Store fundamental frequencies for the interval
        fundamental_frequencies = np.append(fundamental_frequencies, frec_max)

    mean_fundamental_frequency = np.mean(fundamental_frequencies)
    max_fundamental_frequency = np.max(fundamental_frequencies)
    min_fundamental_frequency = np.min(fundamental_frequencies)

    return np.array([mean_fundamental_frequency, max_fundamental_frequency, min_fundamental_frequency])
