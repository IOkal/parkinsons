import numpy as np

from parselmouth.praat import call


def engineer_features(sound):
    f0min = 75
    f0max = 500

    pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)

    jitter = call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)  # jitter
    jitterAbs = call(pointProcess, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)  # localabsoluteJitter
    jitterRap = call(pointProcess, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)  # rapJitter

    jitterPpq5 = call(pointProcess, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)  # ppq5Jitter

    jitterDdp = call(pointProcess, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3)  # ddpJitter

    shimmer = call([sound, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)  # localShimmer
    shimmerDb = call([sound, pointProcess], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)  # localdbShimmer

    apq3Shimmer = call([sound, pointProcess], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq5Shimmer = call([sound, pointProcess], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq11Shimmer = call([sound, pointProcess], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

    ddaShimmer = call([sound, pointProcess], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    hnr = call(harmonicity, "Get mean", 0, 0)

    # Return the features in the order the model will need them in
    return np.array([jitter, jitterAbs, jitterRap, jitterPpq5, jitterDdp, shimmer, shimmerDb, apq3Shimmer, apq5Shimmer, apq11Shimmer, ddaShimmer, hnr])
