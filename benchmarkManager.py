import numpy as np
import matplotlib.pyplot as plt
import time as tm
from numba import cuda

from simulationManager import *
from simulationUtilities import *
from simulation import *
from benchmarkResults import *
from archive import *

def plotBenchmarkComparison(archive, archiveTimes, legend, title):
    """
    Plots multiple benchmarks on one plot from previous archives.

    Parameters
    ----------
    archive : :class:`archive.Archive`
        Specifies the path to save the plot to.
    archiveTimes : `list` of `string`
        The identifiers of the archvies containing the benchmark results to be compared.
    legend : `list` of `string`
        Labels that describe what each of the benchmark result curves respresent.
    title : `string`
        What this comparison is trying to compare.
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

def newBenchmarkTrotterCutoffMatrix(archive, trotterCutoff, normBound = 1.0):
    """
    Runs a benchmark for the trotter exponentiator :func:`simulationUtilities.matrixExponentialLieTrotter()` using arbitrary matrices. Uses :func:`benchmarkTrotterCutoffMatrix()` to execute the matrix exponentials.

    Specifically, let
    
    .. math::
        A_k = -i \\frac{\\nu}{4} (\\cos(k) F_x + \\cos(2k) F_y + \\cos(4k) F_z + \\cos(8k) F_q).

    See :func:`simulationUtilities.matrixExponentialLieTrotter()` for definitions of :math:`F` operators).

    Then :func:`simulationUtilities.matrixExponentialLieTrotter()` calculates the exponential of :math:`A_k` as

    .. math::
        E_{\\tau, k} = \\exp_\\tau(A_k).
    
    Let :math:`\\tau_0` be the first element in `trotterCutoff`, ususally the largest. Then the error :math:`e_\\tau` is calculated as

    .. math::
        e_\\tau = \\frac{1}{\\#k}\\sum_{k, i, j} |(E_{\\tau, k})_{i,j} - E_{\\tau_0, k})_{i,j}|,

    where :math:`\\#k` is the number of matrices being considered in the benchmark (1e6).

    Parameters
    ----------
    archive : :class:`archive.Archive`
        Specifies where to save results and plots.
    trotterCutoff : :class:`numpy.ndarray` of :class:`numpy.int`
        An array of values of the trotter cutoff to run the matrix exponentiator at.
    normBound : `float`, optional
        An upper bound to the size of the norm of the matrices being exponentiated, since :func:`simulationUtilities.matrixExponentialLieTrotter()` works better using matrices with smaller norms. See :math:`\\nu` above. Defaults to 1.

    Returns
    -------
    benchmarkResults : :class:`benchmarkResults.BenchmarkResults`
        Contains the errors found by the benchmark.
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
    Runs the exponentiations for the trotter matrix benchmark.

    Parameters
    ----------
    normBound : `float`, optional
        An upper bound to the size of the norm of the matrices being exponentiated, since :func:`simulationUtilities.matrixExponentialLieTrotter()` works better using matrices with smaller norms. Defaults to 1.
    trotterCutoff : `int`
        The value trotter cutoff to run the matrix exponentiator at.
    result : :class:`numpy.ndarray` of :class:`numpy.cdouble`
        The results of the matrix exponentiations for this value of `trotterCutoff`.
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
    Runs a benchmark for the trotter exponentiator using the integrator.

    Specifically, let :math:`(\\psi_{f,\\tau})_{m,t}` be the calculated state of the spin system, with magnetic number (`stateIndex`) :math:`m` at time :math:`t`, simulated with a dressing of :math:`f` with a trotter cutoff of :math:`\\tau`. Let :math:`\\tau_0` be the first such trotter cutoff in `trotterCutoff` (generally the largest one). Then the error :math:`e_\\tau` calculated by this benchmark is

    .. math::
        \\begin{align*}
            e_\\tau &= \\frac{1}{\\#t\\#f}\\sum_{t,f,m} |(\\psi_{f,\\tau})_{m,t} - (\\psi_{f,\\tau_0})_{m,t}|,
        \\end{align*}

    where :math:`\\#t` is the number of coarse time samples, :math:`\\#f` is the length of `frequency`.

    Parameters
    ----------
    archive : :class:`archive.Archive`
        Specifies where to save results and plots.
    signal : `list` of :class:`testSignal.TestSignal`
        The signals being simulated in the benchmark.
    frequency : :class:`numpy.ndarray` of :class:`numpy.double`
        The dressing frequencies being simulated in the benchmark.
    trotterCutoff : :class:`numpy.ndarray` of :class:`numpy.int`
        An array of values of the trotter cutoff to run the simulations at. The accuracy of the simulation output with each of these values are then compared.

    Returns
    -------
    benchmarkResults : :class:`benchmarkResults.BenchmarkResults`
        Contains the errors found by the benchmark.
    """
    stateOutput = []
    error = []
    simulationManager = SimulationManager(signal, frequency, archive, None, stateOutput, trotterCutoff)
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
    Runs a benchmark to test error induced by raising the size of the time step in the integrator, comparing the output state.

    Specifically, let :math:`(\\psi_{f,\\mathrm{d}t})_{m,t}` be the calculated state of the spin system, with magnetic number (`stateIndex`) :math:`m` at time :math:`t`, simulated with a dressing of :math:`f` with a fine time step of :math:`\\mathrm{d}t`. Let :math:`\\mathrm{d}t_0` be the first such time step in `timeStepFine` (generally the smallest one). Then the error :math:`e_{\\mathrm{d}t}` calculated by this benchmark is

    .. math::
        \\begin{align*}
            e_{\\mathrm{d}t} &= \\frac{1}{\\#t\\#f}\\sum_{t,f,m} |(\\psi_{f,\\mathrm{d}t})_{m,t} - (\\psi_{f,\\mathrm{d}t_0})_{m,t}|,
        \\end{align*}

    where :math:`\\#t` is the number of coarse time samples, :math:`\\#f` is the length of `frequency`.

    Parameters
    ----------
    archive : :class:`archive.Archive`
        Specifies where to save results and plots.
    signalTemplate : :class:`testSignal.TestSignal`
        A description of the signal to use for the environment during the simulation. For each entry in `timeStepFine`, this template is modified so that its :attr:`testSignal.TestSignal.timeProperties.timeStepFine` is equal to that entry. All modified versions of the signal are then simulated for comparison.
    frequency : :class:`numpy.ndarray` of :class:`numpy.double`
        The dressing frequencies being simulated in the benchmark.
    timeStepFine : :class:`numpy.ndarray` of :class:`numpy.double`
        An array of time steps to run the simulations with. The accuracy of the simulation output with each of these values are then compared.

    Returns
    -------
    benchmarkResults : :class:`benchmarkResults.BenchmarkResults`
        Contains the errors found by the benchmark.
    """
    timeStepFine = np.asarray(timeStepFine)
    stateOutput = []
    error = []

    signal = []
    for timeStepFineInstance in timeStepFine:
        timeProperties = TimeProperties(signalTemplate.timeProperties.timeStepCoarse, timeStepFineInstance)
        signalInstance = TestSignal(signalTemplate.neuralPulses, signalTemplate.sinusoidalNoises, timeProperties, False)
        signal += [signalInstance]

    simulationManager = SimulationManager(signal, frequency, archive, None, stateOutput)
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

def newBenchmarkTimeStepFineFrequencyDrift(archive, signalTemplate, timeStepFines, dressingFrequency):
    """
    Runs a benchmark to test error induced by raising the size of the time step in the integrator, comparing measured frequency coefficients.

    Parameters
    ----------
    archive : :class:`archive.Archive`
        Specifies where to save results and plots.
    signalTemplate : :class:`testSignal.TestSignal`
        A description of the signal to use for the environment during the simulation. For each entry in `timeStepFines`, this template is modified so that its :attr:`testSignal.TestSignal.timeProperties.timeStepFine` is equal to that entry. All modified versions of the signal are then simulated for comparison.
    timeStepFines : :class:`numpy.ndarray` of :class:`numpy.double`
        An array of time steps to run the simulations with. The accuracy of the simulation output with each of these values are then compared.
    dressingFrequency : :class:`numpy.ndarray` of :class:`numpy.double`
        The dressing frequencies being simulated in the benchmark.
    """
    dressingFrequency = np.asarray(dressingFrequency)
    signalTemplate.getAmplitude()
    signalTemplate.getFrequencyAmplitude()

    signals = []
    for timeStepFine in timeStepFines:
        timeProperties = TimeProperties(signalTemplate.timeProperties.timeStepCoarse, timeStepFine)
        signal = TestSignal(signalTemplate.neuralPulses, signalTemplate.sinusoidalNoises, timeProperties)
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