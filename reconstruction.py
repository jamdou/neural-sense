import numpy as np
import matplotlib.pyplot as plt
import math
from numba import cuda
import numba as nb
import time as tm

from testSignal import *

class Reconstruction():
    """
    Controls a reconstruction of the signal from Fourier sine coefficients using compressive sensing
    """
    def __init__(self, timeProperties, stepSizeSparse = 0.1, stepSizeManifold = 100000):
        self.timeProperties = timeProperties
        self.frequency = None
        self.frequencyAmplitude = None
        self.amplitude = None

        self.stepSizeSparse = stepSizeSparse
        self.stepSizeManifold = stepSizeManifold

    def readFrequenciesDirectly(self, frequency, frequencyAmplitude, numberOfSamples = 100, frequencyCutoff = 3000):
        """
        Import arbitrary frequency values for reconstruction
        """
        permutation = np.random.choice(range(np.sum(frequency < frequencyCutoff)), numberOfSamples)
        self.frequency = np.ascontiguousarray(frequency[permutation])
        self.frequencyAmplitude = np.ascontiguousarray(frequencyAmplitude[permutation])

    def readFrequenciesFromExperimentResults(self, experimentResults, numberOfSamples = 100, frequencyCutoff = 100000):
        """
        Import frequency values from an experimental results object
        """
        self.readFrequenciesDirectly(experimentResults.frequency, experimentResults.frequencyAmplitude, numberOfSamples, frequencyCutoff)

    def readFrequenciesFromTestSignal(self, testSignal):
        """
        Import frequency values from the dot product Fourier transform of a signal
        """
        self.readFrequenciesDirectly(testSignal.frequency, testSignal.frequencyAmplitude)

    def writeToFile(self, archive):
        """
        Save the reconstruction results to a hdf5 file
        """
        archiveGroupReconstruction = archive.require_group("reconstruction")
        self.timeProperties.writeToFile(archiveGroupReconstruction)
        archiveGroupReconstruction["frequency"] = self.frequency
        archiveGroupReconstruction["frequencyAmplitude"] = self.frequencyAmplitude
        archiveGroupReconstruction["amplitude"] = self.amplitude

    def plot(self, archive, testSignal):
        """
        Plot the reconstruction signal, possibly against a template test signal
        """
        plt.figure()
        if testSignal is not None:
            plt.plot(testSignal.timeProperties.timeCoarse, testSignal.amplitude, "-k")
        plt.plot(self.timeProperties.timeCoarse, self.amplitude, "-r")
        if testSignal is not None:
            plt.legend(["Original", "Reconstruction"])
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude (Hz)")
        plt.grid()
        plt.title(archive.executionTimeString + "Reconstruction")
        plt.savefig(archive.plotPath + "reconstruction.pdf")
        plt.savefig(archive.plotPath + "reconstruction.png")
        plt.show()

    def evaluateISTAComplete(self):
        """
        Run compressive sensing based on the Iterative Shrinkage Thresholding Algorithm (ISTA)
        """
        print("\033[33mStarting reconstruction...\033[0m")

        # Start timing reconstruction
        executionTimeEndPoints = np.empty(2)
        executionTimeEndPoints[0] = tm.time()
        executionTimeEndPoints[1] = executionTimeEndPoints[0]

        self.amplitude = np.empty_like(self.timeProperties.timeCoarse)                          # Reconstructed signal
        frequencyAmplitudePrediction = np.empty_like(self.frequencyAmplitude)                   # Partial sine Fourier transform of reconstructed signal
        fourierTransform = np.empty((self.frequency.size, self.timeProperties.timeCoarse.size)) # Storage for sine Fourier transform operator

        # Setup GPU block and grid sizes
        threadsPerBlock = 128
        blocksPerGridTime = (self.timeProperties.timeCoarse.size + (threadsPerBlock - 1)) // threadsPerBlock

        reconstructISTAComplete[blocksPerGridTime, threadsPerBlock](self.timeProperties.timeCoarse, self.amplitude, self.frequency, self.frequencyAmplitude, frequencyAmplitudePrediction, fourierTransform, self.timeProperties.timeStepCoarse, 0, 100.0, 10.0)

        print(str(tm.time() - executionTimeEndPoints[1]) + "s")
        print("\033[32mDone!\033[0m")
        executionTimeEndPoints[1] = tm.time()

    def evaluateFISTA(self):
        """
        Run compressive sensing based on the Fast Iterative Shrinkage Thresholding Algorithm (FISTA)
        """
        print("\033[33mStarting reconstruction...\033[0m")

        # Start timing reconstruction
        executionTimeEndPoints = np.empty(2)
        executionTimeEndPoints[0] = tm.time()
        executionTimeEndPoints[1] = executionTimeEndPoints[0]

        self.amplitude = np.empty_like(self.timeProperties.timeCoarse)                          # Reconstructed signal
        frequencyAmplitudePrediction = np.empty_like(self.frequencyAmplitude)                   # Partial sine Fourier transform of reconstructed signal
        fourierTransform = np.empty((self.frequency.size, self.timeProperties.timeCoarse.size)) # Storage for sine Fourier transform operator

        # Setup GPU block and grid sizes
        threadsPerBlock = 128
        blocksPerGridTime = (self.timeProperties.timeCoarse.size + (threadsPerBlock - 1)) // threadsPerBlock
        blocksPerGridFrequency = (self.frequency.size + (threadsPerBlock - 1)) // threadsPerBlock

        # Initialise 
        reconstructISTAInitialisationStep[blocksPerGridTime, threadsPerBlock](self.frequency, self.frequencyAmplitude, self.timeProperties.timeStepCoarse, self.timeProperties.timeCoarse, self.amplitude, fourierTransform)

        amplitudePrevious = 0*self.amplitude    # The last amplitude, used in the fast step, and to check (Cauchy) convergence
        fastStepSize = 1                        # Initialise the fast step size to one
        fastStepSizePrevious = 1

        while sum((self.amplitude - amplitudePrevious)**2) > 1e0:   # Stop if signal has converged
            amplitudePrevious = 1*self.amplitude    # Keep track of previous amplitude

            # Run ISTA steps
            reconstructISTAPredictionStep[blocksPerGridFrequency, threadsPerBlock](self.frequencyAmplitude, self.amplitude, fourierTransform, frequencyAmplitudePrediction)
            reconstructISTAManifoldStep[blocksPerGridTime, threadsPerBlock](frequencyAmplitudePrediction, self.stepSizeManifold, fourierTransform, self.amplitude)
            reconstructISTASparseStep[blocksPerGridTime, threadsPerBlock](self.stepSizeSparse, self.amplitude)

            # Run the fast step
            fastStepSizePrevious = fastStepSize
            fastStepSize = (1 + math.sqrt(1 + 4*fastStepSize**2))/2
            self.amplitude = self.amplitude + ((fastStepSizePrevious - 1)/fastStepSize)*(self.amplitude - amplitudePrevious)

        print(str(tm.time() - executionTimeEndPoints[1]) + "s")
        print("\033[32mDone!\033[0m")
        executionTimeEndPoints[1] = tm.time()
    
    def evaluateNaiveISTA(self):
        """
        Run compressive sensing based on the Iterative Shrinkage Thresholding Algorithm (ISTA).
        The same as FISTA, but without the fast step.
        """
        self.amplitude = np.empty_like(self.timeProperties.timeCoarse)
        frequencyAmplitudePrediction = np.empty_like(self.frequencyAmplitude)
        fourierTransform = np.empty((self.frequency.size, self.timeProperties.timeCoarse.size))

        threadsPerBlock = 128
        blocksPerGridTime = (self.timeProperties.timeCoarse.size + (threadsPerBlock - 1)) // threadsPerBlock
        blocksPerGridFrequency = (self.frequency.size + (threadsPerBlock - 1)) // threadsPerBlock
        reconstructNaiveInitialisationStep[blocksPerGridTime, threadsPerBlock](self.frequency, self.frequencyAmplitude, self.timeProperties.timeStepCoarse, self.timeProperties.timeCoarse, self.amplitude, fourierTransform)
        # manifoldStepSize = 100000
        reconstructNaivePredictionStep[blocksPerGridFrequency, threadsPerBlock](self.frequencyAmplitude, self.amplitude, fourierTransform, frequencyAmplitudePrediction)
        squareLoss = sum(frequencyAmplitudePrediction**2)

        # frequency = cuda.to_device(frequency)
        # frequencyAmplitude = cuda.to_device(frequencyAmplitude)
        # frequencyAmplitudePrediction = cuda.to_device(frequencyAmplitudePrediction)
        # timeCoarse = cuda.to_device(timeCoarse)
        # amplitude = cuda.to_device(amplitude)
        # print(squareLoss)

        # plt.figure()
        # plt.plot(timeCoarse, amplitude)
        # plt.show()
        amplitudePrevious = 0*self.amplitude
        while sum((self.amplitude - amplitudePrevious)**2) > 1:
            # if iterationIndex == 0:
            #     manifoldStepSize = 2
            # else:
            #     manifoldStepSize = 0.00005
            # if squareLoss < 1e-4:
            amplitudePrevious = 1*self.amplitude
            reconstructISTASparseStep[blocksPerGridTime, threadsPerBlock](self.stepSizeSparse, self.amplitude)
            reconstructISTAPredictionStep[blocksPerGridFrequency, threadsPerBlock](self.frequencyAmplitude, self.amplitude, fourierTransform, frequencyAmplitudePrediction)
            # squareLossPrevious = squareLoss
            # squareLoss = sum(frequencyAmplitudePrediction**2)
            # if squareLoss > squareLossPrevious:
            #     manifoldStepSize *= 2
            # else:
            #     manifoldStepSize /= 2
            reconstructISTAManifoldStep[blocksPerGridTime, threadsPerBlock](frequencyAmplitudePrediction, self.stepSizeManifold, fourierTransform, self.amplitude)
            # if iterationIndex % 1 == 0:
            #     # print(squareLoss)
            #     # print(frequencyAmplitudePrediction)

            #     plt.figure()
            #     plt.plot(timeCoarse, amplitude)
            #     plt.show()

        # timeCoarse = timeCoarse.copy_to_host()
        # amplitude = amplitude.copy_to_host()
        # plt.figure()
        # plt.plot(self.timeProperties.timeCoarse, self.amplitude)
        # plt.show()

@cuda.jit()
def reconstructISTAComplete(
    timeCoarse, amplitude,                                          # Time
    frequency, frequencyAmplitude, frequencyAmplitudePrediction,    # Frequency
    fourierTransform, timeStepCoarse,                               # Parameters
    sparsePenalty, minAccuracy, expectedAmplitude                   # Parameters
):
    # Initialise
    timeIndex = cuda.threadIdx.x + cuda.blockIdx.x*cuda.blockDim.x
    if timeIndex < timeCoarse.size:
        amplitude[timeIndex] = 0.0
        for frequencyIndex in range(frequency.size):
            # Find the Fourier transform coefficient
            fourierTransform[frequencyIndex, timeIndex] = math.sin(2*math.pi*frequency[frequencyIndex]*timeCoarse[timeIndex])*timeStepCoarse/(timeCoarse[timeCoarse.size - 1] - timeCoarse[0])
            # # Apply the Moore–Penrose inverse of the Fourier transform, based off its SVD
            # amplitude[timeIndex] += fourierTransform[frequencyIndex, timeIndex]*(2.0*(timeCoarse[timeCoarse.size - 1] - timeCoarse[0])/(timeStepCoarse))*frequencyAmplitude[frequencyIndex]

    stepSize = (timeCoarse[timeCoarse.size - 1] - timeCoarse[0])/timeStepCoarse
    maxIterationIndex = math.ceil((((expectedAmplitude/(1e3*timeStepCoarse))**2)/minAccuracy)/stepSize)
    for iterationIndex in range(maxIterationIndex):
        # Prediction
        cuda.syncthreads()
        frequencyIndex = cuda.threadIdx.x + cuda.blockIdx.x*cuda.blockDim.x
        if frequencyIndex < frequencyAmplitudePrediction.size:
            frequencyAmplitudePrediction[frequencyIndex] = -frequencyAmplitude[frequencyIndex]
            for timeIndex in range(timeCoarse.size):
                frequencyAmplitudePrediction[frequencyIndex] += fourierTransform[frequencyIndex, timeIndex]*0.0#amplitude[timeIndex]

        if cuda.threadIdx.x + cuda.blockIdx.x*cuda.blockDim.x == 0:
            print(frequencyAmplitudePrediction[frequencyIndex], frequencyAmplitude[frequencyIndex], amplitude[timeCoarse.size - 1])

        # Linear inverse
        cuda.syncthreads()
        timeIndex = cuda.threadIdx.x + cuda.blockIdx.x*cuda.blockDim.x
        if timeIndex < timeCoarse.size:
            for frequencyIndex in range(frequencyAmplitudePrediction.size):
                amplitude[timeIndex] -= 2*fourierTransform[frequencyIndex, timeIndex]*(frequencyAmplitudePrediction[frequencyIndex])*stepSize

        # Shrinkage
        amplitudeTemporary = math.fabs(amplitude[timeIndex]) - stepSize*sparsePenalty
        if amplitudeTemporary > 0:
            amplitude[timeIndex] = math.copysign(amplitudeTemporary, amplitude[timeIndex])  # Apparently normal "sign" doesn't exist, but this weird thing does :P
        else:
            amplitude[timeIndex] = 0


@cuda.jit()
def reconstructISTAInitialisationStep(frequency, frequencyAmplitude, timeStepCoarse, timeCoarse, amplitude, fourierTransform):
    """
    Generate the Fourier transform matrix, and use the Moore–Penrose inverse to initialise the
    reconstruction to an allowed (but not optimal) solution
    """
    timeIndex = cuda.threadIdx.x + cuda.blockIdx.x*cuda.blockDim.x
    if timeIndex < timeCoarse.size:
        amplitude[timeIndex] = 0
        for frequencyIndex in range(frequency.size):
            # Find the Fourier transform coefficient
            fourierTransform[frequencyIndex, timeIndex] = math.sin(2*math.pi*frequency[frequencyIndex]*timeCoarse[timeIndex])*timeStepCoarse/timeCoarse[timeCoarse.size - 1]
            # Apply the Moore–Penrose inverse of the Fourier transform, based off its SVD
            amplitude[timeIndex] += 2*fourierTransform[frequencyIndex, timeIndex]*(timeCoarse[timeCoarse.size - 1]/(timeStepCoarse))*frequencyAmplitude[frequencyIndex]

@cuda.jit()
def reconstructISTASparseStep(stepSize, amplitude):
    """
    Use gradient decent to minimise the one norm of the reconstruction (ie make it sparse)

    min Z(r),
    Z(r) = norm1(r) (in time),
    dZ(r)/dr(t) = sign(r(t))

    This algorithm is equivalent to
    
    r = sign(r)*ReLU(abs(r) - stepSize)

    from the paper on FISTA.
    """
    timeIndex = cuda.threadIdx.x + cuda.blockIdx.x*cuda.blockDim.x
    if timeIndex < amplitude.size:
        amplitudePrevious = amplitude[timeIndex]
        if amplitudePrevious > 0:
            amplitude[timeIndex] -= stepSize
        elif amplitudePrevious < 0:
            amplitude[timeIndex] += stepSize
        # Set to zero rather than oscillate
        if amplitude[timeIndex]*amplitudePrevious < 0:
            amplitude[timeIndex] = 0

@cuda.jit()
def reconstructISTAPredictionStep(frequencyAmplitude, amplitude, fourierTransform, frequencyAmplitudePrediction):
    """
    Take the sine Fourier transform of the reconstructed signal, and compare it to the measured frequency components. Returns the difference.
    """
    frequencyIndex = cuda.threadIdx.x + cuda.blockIdx.x*cuda.blockDim.x
    if frequencyIndex < frequencyAmplitudePrediction.size:
        frequencyAmplitudePrediction[frequencyIndex] = -frequencyAmplitude[frequencyIndex]
        for timeIndex in range(amplitude.size):
            frequencyAmplitudePrediction[frequencyIndex] += fourierTransform[frequencyIndex, timeIndex]*amplitude[timeIndex]

@cuda.jit()
def reconstructISTAManifoldStep(frequencyAmplitudePrediction, stepSize, fourierTransform, amplitude):
    """
    Use gradient decent to bring the reconstruction closer to having the correct partial sine Fourier transform.

    min X(r),
    X(r) = norm2(S r - s) (in frequency)
    dX(r)/dr = 2 S^T (S r - s)
    """
    timeIndex = cuda.threadIdx.x + cuda.blockIdx.x*cuda.blockDim.x
    if timeIndex < amplitude.size:
        for frequencyIndex in range(frequencyAmplitudePrediction.size):
            amplitude[timeIndex] -= fourierTransform[frequencyIndex, timeIndex]*(frequencyAmplitudePrediction[frequencyIndex])*stepSize