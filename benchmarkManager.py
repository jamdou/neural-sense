import numpy as np
import matplotlib.pyplot as plt
import time as tm
from numba import cuda

from simulationManager import *
from simulationUtilities import *
from benchmarkResults import *
from archive import *

def plotBenchmarkComparison(archive, archiveTimes, legend, title):
    """
    Plots multiple benchmarks on one plot from previous archives
    """
    plt.figure()
    for archiveTime in archiveTimes:
        archivePrevious = Archive(archive.archivePath[:-25], "")
        archivePrevious.openArchiveFile(archiveTime)
        benchmarkResults = BenchmarkResults.readFromArchive(archivePrevious)
        benchmarkResults.plot(None, False)
        archivePrevious.closeArchiveFile(False)
    plt.legend(legend)
    plt.title(archive.executionTimeString + "\n" + title)
    plt.savefig(archive.plotPath + "benchmarkComparison.pdf")
    plt.savefig(archive.plotPath + "benchmarkComparison.png")
    plt.show()

def newBenchmarkTrotterCutoffMatrix(archive, trotterCutoff, normBound = 1):
    """
    Runs a benchmark for the trotter exponentiator using arbitrary matrixes
    """
    print("\033[33mStarting benchmark...\033[0m")
    timeIndexMax = 1000000
    result = np.empty((timeIndexMax, 3, 3), dtype = np.complex128)
    resultBench = np.empty((timeIndexMax, 3, 3), dtype = np.complex128)
    trotterCutoff = np.asarray(trotterCutoff)
    error = np.empty_like(trotterCutoff, dtype = np.double)
    
    threadsPerBlock = 128
    blocksPerGrid = (timeIndexMax + (threadsPerBlock - 1)) // threadsPerBlock
    benchmarkTrotterCutoffMatrix[blocksPerGrid, threadsPerBlock](normBound, trotterCutoff[0], resultBench)

    for trotterCutoffIndex in range(trotterCutoff.size):
        benchmarkTrotterCutoffMatrix[blocksPerGrid, threadsPerBlock](normBound, trotterCutoff[trotterCutoffIndex], result)
        resultDifference = (result - resultBench)
        error[trotterCutoffIndex] = np.sqrt(np.sum(np.real(resultDifference*np.conj(resultDifference))))/timeIndexMax

    print("\033[32mDone!\033[0m")

    benchmarkResults = BenchmarkResults(BenchmarkType.TROTTER_CUTOFF_MATRIX, trotterCutoff, error)
    benchmarkResults.writeToArchive(archive)
    benchmarkResults.plot(archive)

    return benchmarkResults

@cuda.jit
def benchmarkTrotterCutoffMatrix(normBound, trotterCutoff, result):
    """
    Runs the exponentiations for the trotter matrix benchmark
    """
    exponent = cuda.local.array((3, 3), dtype = nb.complex128)

    timeIndex = cuda.threadIdx.x + cuda.blockIdx.x*cuda.blockDim.x
    if timeIndex < result.shape[0]:
        x = normBound*math.cos(1.0*timeIndex)/4
        y = normBound*math.cos(2.0*timeIndex)/4
        z = normBound*math.cos(4.0*timeIndex)/4
        q = normBound*math.cos(8.0*timeIndex)/4

        exponent[0, 0] = -1j*(z + q/3)
        exponent[1, 0] = -1j*(x + 1j*y)/math.sqrt(2.0)
        exponent[2, 0] = 0.0

        exponent[0, 1] = -1j*(x - 1j*y)/math.sqrt(2.0)
        exponent[1, 1] = -1j*(-2/3)*q
        exponent[2, 1] = -1j*(x + 1j*y)/math.sqrt(2.0)

        exponent[0, 2] = 0.0
        exponent[1, 2] = -1j*(x - 1j*y)/math.sqrt(2.0)
        exponent[2, 2] = -1j*(-z + q/3)

        matrixExponentialLieTrotter(exponent, result[timeIndex, :], trotterCutoff)

def newBenchmarkTrotterCutoff(archive, signal, frequency, trotterCutoff):
    """
    Runs a benchmark for the trotter exponentiator using the integrator
    """
    stateOutput = []
    error = []
    simulationManager = SimulationManager(signal, frequency, archive, stateOutput, trotterCutoff)
    simulationManager.evaluate(False)
    for trotterCutoffIndex in range(trotterCutoff.size):
        errorTemp = 0
        for frequencyIndex in range(frequency.size):
            stateDifference = stateOutput[frequencyIndex + trotterCutoffIndex*frequency.size] - stateOutput[frequencyIndex]
            errorTemp += np.sum(np.sqrt(np.real(np.conj(stateDifference)*stateDifference)))
        error += [errorTemp/(frequency.size*stateOutput[0].size)]
    
    trotterCutoff = np.asarray(trotterCutoff)
    error = np.asarray(error)

    benchmarkResults = BenchmarkResults(BenchmarkType.TROTTER_CUTOFF, trotterCutoff, error)
    benchmarkResults.writeToArchive(archive)
    benchmarkResults.plot(archive)

    return benchmarkResults

def newBenchmarkTimeStepFine(archive, signalTemplate, frequency, timeStepFine):
    """
    Runs a benchmark to test error induced by raising the size of the time step in the integrator, comparing the output state
    """
    timeStepFine = np.asarray(timeStepFine)
    stateOutput = []
    error = []

    signal = []
    for timeStepFineInstance in timeStepFine:
        timeProperties = TimeProperties(signalTemplate.timeProperties.timeStepCoarse, timeStepFineInstance)
        signalInstance = TestSignal(signalTemplate.neuralPulses, timeProperties, False)
        signal += [signalInstance]

    simulationManager = SimulationManager(signal, frequency, archive, stateOutput)
    simulationManager.evaluate(False)

    for timeStepFineIndex in range(timeStepFine.size):
        errorTemp = 0
        for frequencyIndex in range(frequency.size):
            stateDifference = stateOutput[frequencyIndex + timeStepFineIndex*frequency.size] - stateOutput[frequencyIndex]
            errorTemp += np.sum(np.sqrt(np.real(np.conj(stateDifference)*stateDifference)))
        error += [errorTemp/(frequency.size*stateOutput[0].size)]
    
    error = np.asarray(error)

    benchmarkResults = BenchmarkResults(BenchmarkType.TIME_STEP_FINE, timeStepFine, error)
    benchmarkResults.writeToArchive(archive)
    benchmarkResults.plot(archive)

    return benchmarkResults

def newBenchmarkFineStepFrequencyDrift(archive, signalTemplate, timeStepFines, dressingFrequency = np.arange(500, 1500, 10)):
    """
    Runs a benchmark to test error induced by raising the size of the time step in the integrator, comparing measured frequency coefficients
    """
    dressingFrequency = np.asarray(dressingFrequency)
    signalTemplate.getAmplitude()
    signalTemplate.getFrequencyAmplitude()

    signals = []
    for timeStepFine in timeStepFines:
        timeProperties = TimeProperties(signalTemplate.timeProperties.timeStepCoarse, timeStepFine)
        signal = TestSignal(signalTemplate.neuralPulses, timeProperties)
        signals += [signal]

    simulationManager = SimulationManager(signals, dressingFrequency, archive)
    simulationManager.evaluate()

    frequencyDrift = np.zeros(len(signals))
    for signalIndex in range(len(signals)):
        for frequencyIndex, frequency in enumerate(dressingFrequency):
            frequencyDrift[signalIndex] += np.abs(simulationManager.frequencyAmplitude[frequencyIndex + signalIndex*dressingFrequency.size] - signalTemplate.frequencyAmplitude[signalTemplate.frequency == frequency])
        frequencyDrift[signalIndex] /= dressingFrequency.size

    benchmarkResults = BenchmarkResults(BenchmarkType.TIME_STEP_FINE_FREQUENCY_DRIFT, np.asarray(timeStepFines), frequencyDrift)
    benchmarkResults.writeToArchive(archive)
    benchmarkResults.plot(archive)

    return benchmarkResults