import numpy as np
import math
import matplotlib.pyplot as plt
from numba import cuda
import numba as nb

cudaDebug = False

class TimeProperties:
    """
    Grouped details about time needed for simulations and reconstructions
    """
    def __init__(self, timeStepCoarse = 1e-5, timeStepFine = 2.5e-7, timeEndPoints = np.asarray([0, 1e-1])):
        self.timeStepCoarse = timeStepCoarse            # Sampled timestep (s)
        self.timeStepFine = timeStepFine                # Simulated timestep (s)
        self.timeEndPoints = timeEndPoints              # When the signal starts and ends (s) [start time / end time]
        self.timeIndexMax = int((self.timeEndPoints[1] - self.timeEndPoints[0])/self.timeStepCoarse)
        self.timeCoarse = np.empty(self.timeIndexMax)   # The timeseries, mainly for x axis for plots (s) [time index]

    def writeToFile(self, archive):
        """
        Saves the time properties to a hdf5 file
        """
        archive["timeProperties"] = self.timeCoarse
        archive["timeProperties"].attrs["timeStepCoarse"] = self.timeStepCoarse
        archive["timeProperties"].attrs["timeStepFine"] = self.timeStepFine
        archive["timeProperties"].attrs["timeEndPoints"] = self.timeEndPoints
        archive["timeProperties"].attrs["timeIndexMax"] = self.timeIndexMax

    def readFromFile(self, archive):
        """
        Loads the time properties from a hdf5 file
        """
        self.timeCoarse = archive["timeProperties"]
        self.timeStepCoarse = archive["timeProperties"].attrs["timeStepCoarse"]
        self.timeStepFine = archive["timeProperties"].attrs["timeStepFine"]
        self.timeEndPoints = archive["timeProperties"].attrs["timeEndPoints"]
        self.timeIndexMax = archive["timeProperties"].attrs["timeIndexMax"]

class NeuralPulse:
    """
    Details about a simulated neural pulse
    """
    def __init__(self, timeStart = 0, amplitude = 10, frequency = 1e3):
        self.timeStart = timeStart
        self.amplitude = amplitude
        self.frequency = frequency

    def writeToFile(self, archive, index):
        """
        Writes the pulse details to a hdf5 file
        """
        archive[str(index)] = 0
        archive[str(index)].attrs["timeStart"] = self.timeStart
        archive[str(index)].attrs["amplitude"] = self.amplitude
        archive[str(index)].attrs["frequency"] = self.frequency

class TestSignal:
    """
    Details for a simulated neural pulse sequence ONLY (no noise or control signals)
    """
    def __init__(self, neuralPulses = [NeuralPulse()], timeProperties = TimeProperties(), doEvaluate = True):
        self.timeProperties = timeProperties                        # Details about time for the signal (time properties object)
        self.neuralPulses = neuralPulses                            # List of individual neural pulses that make up the signal (neural pulse object) [pulse index]
        self.amplitude = np.empty_like(timeProperties.timeCoarse)   # Time series of the signal (Hz) [time index]
        self.frequency = np.empty_like(timeProperties.timeCoarse)   # List of frequencies corresponding to the Fourier transform of the pulse sequence (Hz) [frequency index]
        self.frequencyAmplitude = np.empty_like(self.frequency)     # The sine Fourier transform of the pulse sequence (Hz) [frequency index]
        if doEvaluate:
            self.getAmplitude()
            self.getFrequencyAmplitude()

    def getAmplitude(self):
        """
        Finds the timeseries representation of the neural pulse sequence
        """
        # Unwrap the neural pulse objects
        neuralPulseIndexMax = len(self.neuralPulses)
        timeStart = []
        amplitude = []
        frequency = []
        for neuralPulse in self.neuralPulses:
            timeStart += [neuralPulse.timeStart]
            amplitude += [neuralPulse.amplitude]
            frequency += [neuralPulse.frequency]
        timeStart = np.asarray(timeStart)
        amplitude = np.asarray(amplitude)
        frequency = np.asarray(frequency)

        # GPU control variables
        threadsPerBlock = 128
        blocksPerGrid = (self.timeProperties.timeIndexMax + (threadsPerBlock - 1)) // threadsPerBlock
        # Run GPU code
        getAmplitude[blocksPerGrid, threadsPerBlock](self.timeProperties.timeCoarse, self.timeProperties.timeEndPoints, self.timeProperties.timeStepCoarse, neuralPulseIndexMax, timeStart, amplitude, frequency, self.amplitude)

    def getFrequencyAmplitude(self):
        """
        Takes a sine Fourier transform of the pulse sequence using a dot product
        """
        # GPU control variables
        threadsPerBlock = 128
        blocksPerGrid = (self.timeProperties.timeIndexMax + (threadsPerBlock - 1)) // threadsPerBlock
        # Run GPU code
        getFrequencyAmplitude[blocksPerGrid, threadsPerBlock](self.timeProperties.timeEndPoints, self.timeProperties.timeCoarse, self.timeProperties.timeStepCoarse, self.amplitude, self.frequency, self.frequencyAmplitude)

    def plotFrequencyAmplitude(self, archive):
        """
        Plots the sine Fourier transform of the pulse sequence
        """
        plt.plot(self.frequency, self.frequencyAmplitude, "+--")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude (Hz)")
        plt.xlim([0, 2000])
        plt.grid()
        plt.title(archive.executionTimeString + "Fourier Transform Frequency Amplitude")
        plt.savefig(archive.plotPath + "fourierTransformFrequencyAmplitude.pdf")
        plt.savefig(archive.plotPath + "fourierTransformFrequencyAmplitude.png")
        plt.show()
        
    def writeToFile(self, archive):
        """
        Writes the pulse sequence description to a hdf5 file
        """
        archiveGroup = archive.require_group("testSignal")
        archiveGroup["amplitude"] = self.amplitude
        archiveGroup["frequency"] = self.frequency
        archiveGroup["frequencyAmplitude"] = self.frequencyAmplitude

        self.timeProperties.writeToFile(archiveGroup)
        
        archiveGroupNeuralPulsesGroup = archiveGroup.require_group("neuralPulses")
        for neuralPulseIndex, neuralPulse in enumerate(self.neuralPulses):
            neuralPulse.writeToFile(archiveGroupNeuralPulsesGroup, neuralPulseIndex)

@cuda.jit(debug = cudaDebug)
def getAmplitude(
    timeCoarse, timeEndPoints, timeStepCoarse,
    neuralPulseIndexMax, timeStart, amplitude, frequency,
    timeAmplitude):
    """
    Writes the timeseries for a pulse sequence
    """
    timeIndex = cuda.threadIdx.x + cuda.blockIdx.x*cuda.blockDim.x  # The time array index the current thread is working on
    if timeIndex < timeCoarse.size:                                 # Check we haven't overflowed the array
        # Evaluate time
        timeCoarse[timeIndex] = timeIndex*timeStepCoarse

        # Evaluate signal
        timeAmplitude[timeIndex] = 0
        for neuralPulseIndex in range(neuralPulseIndexMax):
            if timeCoarse[timeIndex] > timeStart[neuralPulseIndex] and timeCoarse[timeIndex] < timeStart[neuralPulseIndex] + 1/frequency[neuralPulseIndex]:
                timeAmplitude[timeIndex] += amplitude[neuralPulseIndex]*math.sin(2*math.pi*frequency[neuralPulseIndex]*(timeCoarse[timeIndex] - timeStart[neuralPulseIndex]))

@cuda.jit(debug = cudaDebug, max_registers = 31)
def getFrequencyAmplitude(timeEndPoints, timeCoarse, timeStepCoarse, timeAmplitude, frequency, frequencyAmplitude):
    """
    Takes the sine Fourier transform of the pulse sequence, given a time series.
    Max registers set to 31 => my GPU is at full capacity. Might be worth removing or increasing if too slow.
    """
    frequencyIndex = cuda.threadIdx.x + cuda.blockIdx.x*cuda.blockDim.x # The frequency array index the current thread is working on
    if frequencyIndex < frequencyAmplitude.size:                        # Check we haven't overflowed the array
        frequencyStep = 0.125/(timeEndPoints[1] - timeEndPoints[0])
        frequency[frequencyIndex] = frequencyStep*frequencyIndex        # The frequency this thread is calculating the coefficient for
        frequencyAmplitudeTemporary = 0                                 # It's apparently more optimal to repeatedly write to a register (this) than the output array in memory. Though honestly can't see the improvement
        frequencyTemporary = frequency[frequencyIndex]                  # And read from a register rather than memory
        for timeIndex in nb.prange(timeCoarse.size):
            frequencyAmplitudeTemporary += timeAmplitude[timeIndex]*math.sin(2*math.pi*timeCoarse[timeIndex]*frequencyTemporary)    # Dot product
        frequencyAmplitude[frequencyIndex] = frequencyAmplitudeTemporary*timeStepCoarse/(timeEndPoints[1] - timeEndPoints[0])       # Scale to make an integral
