import numpy as np
import matplotlib.pyplot as plt
import h5py

# from simulationManager import *

class ExperimentResults:
    def __init__(self, frequency = None, frequencyAmplitude = None):
        self.frequency = frequency
        self.frequencyAmplitude = frequencyAmplitude

    def readFromSimulationManager(self, simulationManager):
        self.frequency = 1*simulationManager.frequency
        self.frequency = 1*simulationManager.frequencyAmplitude

    def plot(self, archive, testSignal):
        plt.figure()
        plt.plot(testSignal.frequency, testSignal.frequencyAmplitude, "-k")
        plt.plot(self.frequency, self.frequencyAmplitude, "xr")
        plt.legend(["Fourier Transform", "Measured"])
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude (Hz)")
        plt.xlim([0, 2000])
        # plt.ylim([-0.08, 0.08])
        plt.grid()
        plt.title(archive.executionTimeString + "Measured Frequency Amplitude")
        plt.savefig(archive.plotPath + "measuredFrequencyAmplitude.pdf")
        plt.savefig(archive.plotPath + "measuredFrequencyAmplitude.png")
        plt.show()

    def writeToArchive(self, archive):
        archiveGroupExperimentResults = archive.archiveFile.require_group("experimentResults")
        archiveGroupExperimentResults["frequency"] = self.frequency
        archiveGroupExperimentResults["frequencyAmplitude"] = self.frequencyAmplitude