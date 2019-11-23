modulePath = './src/'
import sys
sys.path.append(modulePath)

import generalUtility
import dspUtil
import matplotlibUtil
import praatUtil
import os

praatUtil.calculateF0("audio-file.wav", readProgress=0.01, acFreqMin=60, voicingThreshold=0.45, veryAccurate=False, fMax=2000, octaveJumpCost=0.35, silenceThreshold=0.03, octaveCost=0.01, voicedUnvoicedCost=0.14, maxNumCandidates=15, verbose=False)
