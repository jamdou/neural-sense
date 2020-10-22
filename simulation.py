"""
.. _overviewOfSimulationMethod:

*********************************
Overview of the simulation method
*********************************

The goal here is to evaluate the value of the spin, and thus the quantum state of a 3 level atom in a coarse grained time series with step :math:`\\mathrm{D}t`. The time evolution between time indices is

.. math::
   \\begin{align*}
   \\psi(t + \\mathrm{D}t) &= U(t \\rightarrow t + \\mathrm{D}t) \\psi(t)\\\\
   \\psi(t + \\mathrm{D}t) &= U(t) \\psi(t)
   \\end{align*}

Each :math:`U(t)` is completely independent of :math:`\\psi(t_0)` or :math:`U(t_0)` for any other time value :math:`t_0`. Therefore each :math:`U(t)` can be calculated independently of each other. This is done in parallel using a GPU kernel in the function :func:`getTimeEvolutionCommutatorFree4RotatingWave()` (the highest performing variant of this solver). Afterwards, the final result of

.. math::
   \\psi(t + \\mathrm{D}t) = U(t) \\psi(t)

is calculated sequentially for each :math:`t` in the function :func:`getState()`. Afterwards, the spin at each time step is calculated in parallel in the function :func:`getSpin()`.

All magnetic signals fed into the integrator in the form of sine waves, with varying amplitude, frequency, phase, and start and end times. This can be used to simulate anything from the bias and dressing fields, to the fake neural pulses, to AC line and DC detuning noise. These sinusoids are superposed and sampled at any time step needed to for the solver. The magnetic signals are written in high level as :class:`testSignal.TestSignal` objects, and are converted to a parametrisation readable to the integrator in the form of :class:`SourceProperties` objects.

*********************
Classes and functions
*********************
"""

import numpy as np
import scipy.linalg as spla
import matplotlib.pyplot as plt
import math
from numba import cuda
import numba as nb
import time as tm
from testSignal import *
from simulationUtilities import *

#===============================================================#

# Important constants
cudaDebug = False
sqrt2 = math.sqrt(2)
sqrt3 = math.sqrt(3)
expPrecision = 5                                # Where to cut off the exp Taylor series
machineEpsilon = np.finfo(np.float64).eps*1000  # When to decide that vectors are parallel
# trotterCutoff = 52

class SourceProperties:
    """
    A list of sine wave parameters fed into the simulation code.

    The source parametrised as
    
    :math:`b_{i,x}(t) = 2 \\pi` :attr:`sourceAmplitude` :math:`_{i,x}\\sin(2 \\pi` :attr:`sourceFrequency` :math:`_{i,x}(t -` :attr:`sourceTimeEndPoints` :math:`_{i,0}) +` :attr:`sourcePhase` :math:`_{i,x})`
    
    Attributes
    ----------
    dressingRabiFrequency : `float`
        The amplitude of the dressing in units of Hz.
    sourceIndexMax : `int`
        The number of sources in the simulation.
    sourceAmplitude : :class:`numpy.ndarray` of :class:`numpy.double`, (sourceIndex, spatialIndex)
        The amplitude of the sine wave of source `sourceIndex` in direction `spatialIndex`. In units of Hz.
    sourcePhase : :class:`numpy.ndarray` of :class:`numpy.double`, (sourceIndex, spatialIndex)
        The phase offset of the sine wave of source `sourceIndex` in direction `spatialIndex`. In units of radians.
    sourceFrequency : :class:`numpy.ndarray` of :class:`numpy.double`, (sourceIndex, spatialIndex)
        The frequency of the sine wave of source `sourceIndex` in direction `spatialIndex`. In units of Hz.
    sourceTimeEndPoints : :class:`numpy.ndarray` of :class:`numpy.double`, (sourceIndex, turn on time (0) or turn off time (1))
        The times that the sine wave of source `sourceIndex` turns on and off. In units of s.
    sourceQuadraticShift :  `float`
        The constant quadratic shift of the spin 1 system, in Hz.
    sourceType : :class:`numpy.ndarray` of :class:`numpy.double`, (sourceIndex)
        A string description of what source sourceIndex physically represents. Mainly for archive purposes.
    """
    def __init__(self, signal, dressingRabiFrequency = 1000.0, quadraticShift = 0.0):
        """
        Parameters
        ----------
        signal : :class:`testSignal.TestSignal`
            An object that contains high level descriptions of the signal to be measured, as well as noise.
        dressingRabiFrequency : `float`, optional
            The amplitude of the dressing in units of Hz.
        quadraticShift :  `float`, optional
            The constant quadratic shift of the spin 1 system, in Hz.
        """
        self.dressingRabiFrequency = dressingRabiFrequency
        self.sourceIndexMax = 0
        self.sourceAmplitude = np.empty([0, 3], np.double)
        self.sourcePhase = np.empty([0, 3], np.double)
        self.sourceFrequency = np.empty([0, 3], np.double)
        self.sourceTimeEndPoints = np.empty([0, 2], np.double)
        self.sourceType = np.empty([0], object)
        self.sourceQuadraticShift = quadraticShift

        # Construct the signal from the dressing information and pulse description.
        if signal:
            self.addDressing(signal)
            for neuralPulse in signal.neuralPulses:
                self.addNeuralPulse(neuralPulse)
                # self.addNeuralPulse(neuralPulse.timeStart, neuralPulse.amplitude, neuralPulse.frequency)
            for sinusoidalNoise in signal.sinusoidalNoises:
                self.addSinusoidalNoise(sinusoidalNoise)
        
    def writeToFile(self, archive):
        """
        Saves source information to archive file.

        Parameters
        ----------
        archive : :class:`h5py.Group`
            The HDF5 archive to write to.
        """
        archiveGroup = archive.require_group("sourceProperties")
        archiveGroup["sourceAmplitude"] = self.sourceAmplitude
        archiveGroup["sourcePhase"] = self.sourcePhase
        archiveGroup["sourceFrequency"] = self.sourceFrequency
        archiveGroup["sourceTimeEndPoints"] = self.sourceTimeEndPoints
        archiveGroup["sourceType"] = np.asarray(self.sourceType, dtype='|S32')
        archiveGroup["sourceIndexMax"] = np.asarray([self.sourceIndexMax])
        archiveGroup["sourceQuadraticShift"] = np.asarray([self.sourceQuadraticShift])

    def addDressing(self, signal, biasAmplitude = 700e3):
        """
        Adds a specified bias field and dressing field to the list of sources.

        Parameters
        ----------
        signal : :class:`testSignal.TestSignal`
            The signal object. Needed to specify how long the dressing and bias should be on for.
        biasAmplitude : `float`, optional
            The strength of the dc bias field in Hz. Also the frequency of the dressing.
            If one wants to add detuning, one can do that via detuning noise in :class:`testSignal.SinusoidalNoise.newDetuningNoise()`.
        """
        # Initialise
        sourceAmplitude = np.zeros([1, 3])
        sourcePhase = np.zeros([1, 3])
        sourceFrequency = np.zeros([1, 3])
        sourceTimeEndPoints = np.zeros([1, 2])
        sourceType = np.empty([1], dtype = object)

        # Label
        sourceType[0] = "Dressing"

        # Bias
        sourceAmplitude[0, 2] = biasAmplitude
        sourcePhase[0, 2] = math.pi/2

        #Dressing
        sourceAmplitude[0, 0] = 2*self.dressingRabiFrequency
        sourceFrequency[0, 0] = biasAmplitude #self.dressingFrequency
        sourceTimeEndPoints[0, :] = 1*signal.timeProperties.timeEndPoints
        sourcePhase[0, 0] = math.pi/2

        # Add
        self.sourceAmplitude = np.concatenate((self.sourceAmplitude, sourceAmplitude))
        self.sourcePhase = np.concatenate((self.sourcePhase, sourcePhase))
        self.sourceFrequency = np.concatenate((self.sourceFrequency, sourceFrequency))
        self.sourceTimeEndPoints = np.concatenate((self.sourceTimeEndPoints, sourceTimeEndPoints))
        self.sourceType = np.concatenate((self.sourceType, sourceType))
        self.sourceIndexMax += 1

    def addNeuralPulse(self, neuralPulse):
        """
        Adds a neural pulse signal to the list of sources from a :class:`testSignal.NeuralPulse` object.

        Parameters
        ----------
        neuralPulse : :class:`testSignal.NeuralPulse`
            An object parameterising the neural pulse signal to be added to the list of sources.
        """
        # Initialise
        sourceAmplitude = np.zeros([1, 3])
        sourcePhase = np.zeros([1, 3])
        sourceFrequency = np.zeros([1, 3])
        sourceTimeEndPoints = np.zeros([1, 2])
        sourceType = np.empty([1], dtype = object)

        # Label
        sourceType[0] = "NeuralPulse"

        # Pulse
        sourceAmplitude[0, 2] = neuralPulse.amplitude
        sourceFrequency[0, 2] = neuralPulse.frequency
        # sourcePhase[0, 2] = math.pi/2
        sourceTimeEndPoints[0, :] = np.asarray([neuralPulse.timeStart, neuralPulse.timeStart + 1/neuralPulse.frequency])

        # Add
        self.sourceAmplitude = np.concatenate((self.sourceAmplitude, sourceAmplitude))
        self.sourcePhase = np.concatenate((self.sourcePhase, sourcePhase))
        self.sourceFrequency = np.concatenate((self.sourceFrequency, sourceFrequency))
        self.sourceTimeEndPoints = np.concatenate((self.sourceTimeEndPoints, sourceTimeEndPoints))
        self.sourceType = np.concatenate((self.sourceType, sourceType))
        self.sourceIndexMax += 1

    def addSinusoidalNoise(self, sinusoidalNoise):
        """
        Adds sinusoidal noise from a :class:`testSignal.SinusidalNoise` object to the list of sources.

        Parameters
        ----------
        sinusoidalNoise : :class:`testSignal.SinusidalNoise`
            The sinusoidal noise object to add to the list of sources.
        """
        # Initialise
        sourceAmplitude = np.zeros([1, 3])
        sourcePhase = np.zeros([1, 3])
        sourceFrequency = np.zeros([1, 3])
        sourceTimeEndPoints = np.zeros([1, 2])
        sourceType = np.empty([1], dtype = object)

        # Label
        sourceType[0] = "Noise"

        # Pulse
        sourceAmplitude[0, :] = sinusoidalNoise.amplitude
        sourceFrequency[0, :] = sinusoidalNoise.frequency
        sourcePhase[0, :] = sinusoidalNoise.phase
        sourceTimeEndPoints[0, :] = np.asarray([0.0, 1800.0])

        # Add
        self.sourceAmplitude = np.concatenate((self.sourceAmplitude, sourceAmplitude))
        self.sourcePhase = np.concatenate((self.sourcePhase, sourcePhase))
        self.sourceFrequency = np.concatenate((self.sourceFrequency, sourceFrequency))
        self.sourceTimeEndPoints = np.concatenate((self.sourceTimeEndPoints, sourceTimeEndPoints))
        self.sourceType = np.concatenate((self.sourceType, sourceType))
        self.sourceIndexMax += 1

class StateProperties:
    """
    The initial state fed into the simulation code.

    Attributes
    ----------
    stateInit : :class:`numpy.ndarray` of :class:`numpy.cdouble` (stateIndex)
        The state (spin wavefunction) of the system at the start of the simulation.
    """
    def __init__(self, stateInit = np.asarray([1.0, 0.0, 0.0], np.cdouble)):
        """
        Parameters
        ----------
        stateInit : :class:`numpy.ndarray` of :class:`numpy.cdouble`
            The state (spin wavefunction) of the system at the start of the simulation.
        """
        self.stateInit = stateInit

    def writeToFile(self, archive):
        """
        Parameters
        ----------
        archive : :class:`h5py.Group`
            The HDF5 archive to write to.
        """
        archiveGroup = archive.require_group("stateProperties")
        archiveGroup["stateInit"] = self.stateInit

class SimulationResults:
    """
    The output of the simulation code.

    Attributes
    ----------
    timeEvolution : :class:`numpy.ndarray` of :class:`numpy.cdouble` (timeIndex, braStateIndex, ketStateIndex)
        Time evolution operator (matrix) between the current and next timesteps, for each time sampled. See :math:`U(t)` in :ref:`overviewOfSimulationMethod`.
    state : :class:`numpy.ndarray` of :class:`numpy.cdouble` (timeIndex, stateIndex)
        The state (wavefunction) of the spin system in the lab frame, for each time sampled. See :math:`\\psi(t)` in :ref:`overviewOfSimulationMethod`.
    spin : :class:`numpy.ndarray` of :class:`numpy.double` (timeIndex, spatialIndex)
        The expected value for hyperfine spin of the spin system in the lab frame, for each time sampled. Units of :math:`\\hbar`.
    sensedFrequencyAmplitude : `float`
        The measured Fourier coefficient from the simulation.
    sensedFrequencyAmplitudeMethod : `string`
        The method used to find the measured Fourier coefficient (for archival purposes).
    """
    def __init__(self, signal):
        self.timeEvolution = np.empty([signal.timeProperties.timeIndexMax, 3, 3], np.cdouble)
        self.state = np.empty([signal.timeProperties.timeIndexMax, 3], np.cdouble)
        self.spin = np.empty([signal.timeProperties.timeIndexMax, 3], np.double)
        self.sensedFrequencyAmplitude = 0.0
        self.sensedFrequencyAmplitudeMethod = "none"

    def writeToFile(self, archive, doEverything = False):
        """
        Saves results to the hdf5 file.

        Parameters
        ----------
        archive : :class:`h5py.Group`
            The HDF5 archive to write to.
        doEverything : `boolean`, optional
            If `True`, then save all time series data to file as well as parametric data. Defaults to `False` to reduce archive file size.
        """
        archiveGroup = archive.require_group("simulationResults")
        if doEverything:
            archiveGroup["timeEvolution"] = self.timeEvolution
            archiveGroup["state"] = self.state
            archiveGroup["spin"] = self.spin
        archiveGroup["sensedFrequencyAmplitude"] = self.sensedFrequencyAmplitude
        archiveGroup["sensedFrequencyAmplitude"].attrs["method"] = self.sensedFrequencyAmplitudeMethod

class Simulation:
    """
    The data needed and algorithms to control an individual simulation.

    Attributes
    ----------
    signal : :class:`testSignal.TestSignal`
        The :class:`testSignal.TestSignal` object source signal to be measured.
    sourceProperties : :class:`SourceProperties`
        The :class:`SourceProperties` parametrised sinusoidal source object to evolve the state with.
    stateProperties : :class:`StateProperties`
        The :class:`StateProperties` initial conditions for the wavefunction of the quantum system.
    simulationResults : :class:`SimulationResults`
        A record of the results of the simulation.
    trotterCutoff : `int`
        The number of squares made by the spin 1 matrix exponentiator.
    """
    def __init__(self, signal, dressingRabiFrequency = 1e3, stateProperties = StateProperties(), trotterCutoff = 52):
        """
        Parameters
        ----------
        signal : :class:`testSignal.TestSignal`
            The :class:`testSignal.TestSignal` object source signal to be measured.
        dressingRabiFrequency : `float`, optional
            The amplitude of the dressing radiation to be applied to the system. Units of Hz.
        stateProperties : :class:`StateProperties`, optional
            The :class:`StateProperties` initial conditions for the wavefunction of the quantum system.
        trotterCutoff : `int`, optional
            The number of squares made by the spin 1 matrix exponentiator.
        """
        self.signal = signal
        self.sourceProperties = SourceProperties(self.signal, dressingRabiFrequency)
        self.stateProperties = stateProperties
        self.simulationResults = SimulationResults(self.signal)
        self.trotterCutoff = trotterCutoff

        self.evaluate()

    def evaluate(self):
        """
        Time evolves the system, and finds the spin at each coarse time step.
        """
        # Decide GPU block and grid sizes
        threadsPerBlock = 64
        blocksPerGrid = (self.signal.timeProperties.timeIndexMax + (threadsPerBlock - 1)) // threadsPerBlock

        # Run stepwise solver
        self.simulationResults.timeEvolution = cuda.device_array_like(self.simulationResults.timeEvolution)
        self.signal.timeProperties.timeCoarse = cuda.device_array_like(self.signal.timeProperties.timeCoarse)

        getTimeEvolutionCommutatorFree4RotatingWave[blocksPerGrid, threadsPerBlock](self.signal.timeProperties.timeCoarse, cuda.to_device(self.signal.timeProperties.timeEndPoints), self.signal.timeProperties.timeStepFine, self.signal.timeProperties.timeStepCoarse, self.sourceProperties.sourceIndexMax, cuda.to_device(self.sourceProperties.sourceAmplitude), cuda.to_device(self.sourceProperties.sourceFrequency), cuda.to_device(self.sourceProperties.sourcePhase), cuda.to_device(self.sourceProperties.sourceTimeEndPoints), self.sourceProperties.sourceQuadraticShift, self.simulationResults.timeEvolution, self.trotterCutoff)

        self.simulationResults.timeEvolution = self.simulationResults.timeEvolution.copy_to_host()
        self.signal.timeProperties.timeCoarse = self.signal.timeProperties.timeCoarse.copy_to_host()

        # Combine results of the stepwise solver to evaluate the timeseries for the state
        getState(self.stateProperties.stateInit, self.simulationResults.state, self.simulationResults.timeEvolution)

        # Evaluate the time series for the expected spin value
        self.simulationResults.spin = cuda.device_array_like(self.simulationResults.spin)

        getSpin[blocksPerGrid, threadsPerBlock](cuda.to_device(self.simulationResults.state), self.simulationResults.spin)

        self.simulationResults.spin = self.simulationResults.spin.copy_to_host()

    def getFrequencyAmplitudeFromDemodulation(self, demodulationTimeEndPoints = [0.09, 0.1], doPlotSpin = False):
        """
        Uses demodulation of the Faraday signal to find the measured Fourier coefficient.

        Parameters
        ----------
        demodulationTimeEndPoints : :class:`numpy.ndarray` of :class:`numpy.double` (start time (0) or end time (1)), optional
            The bounds of the interval where Faraday demodulation is used to acquire the measured frequency amplitude (Fourier coefficient) required for reconstruction
        doPlotSpin : `boolean`, optional
            If `True`, plot the full time series for the expected spin of the system.
        """

        # Look at the end of the signal only to find the displacement
        demodulationTimeEndPoints = np.asarray(demodulationTimeEndPoints)
        demodulationTimeIndexEndpoints = np.floor(demodulationTimeEndPoints / self.signal.timeProperties.timeStepCoarse)
        timeCoarse = 1*np.ascontiguousarray(self.signal.timeProperties.timeCoarse[int(demodulationTimeIndexEndpoints[0]):int(demodulationTimeIndexEndpoints[1])])
        spin = 1*np.ascontiguousarray(self.simulationResults.spin[int(demodulationTimeIndexEndpoints[0]):int(demodulationTimeIndexEndpoints[1]), 0])
        spinDemodulated = np.empty_like(spin)

        # spin = 1*self.simulationResults.spin[:, 0]
        # timeCoarse = 1*self.signal.timeProperties.timeCoarse
        # plt.figure()
        # plt.plot(timeCoarse, spin)
        # plt.plot(self.simulationResults.spin[:, :])
        # plt.plot(self.signal.timeProperties.timeCoarse, self.simulationResults.spin[:, :])

        # Decide GPU block and grid sizes
        threadsPerBlock = 128
        blocksPerGrid = (self.signal.timeProperties.timeIndexMax + (threadsPerBlock - 1)) // threadsPerBlock

        # Multiply the Faraday signal by 2cos(wt)
        getFrequencyAmplitudeFromDemodulationMultiply[blocksPerGrid, threadsPerBlock](timeCoarse, spin, spinDemodulated, self.sourceProperties.sourceAmplitude[0, 2])

        # plt.plot(timeCoarse, spinDemodulated)

        # Average the result of the multiplication (ie apply a strict low pass filter and retreive the DC value)
        self.simulationResults.sensedFrequencyAmplitude = 0.0
        self.simulationResults.sensedFrequencyAmplitudeMethod = "demodulation"
        self.simulationResults.sensedFrequencyAmplitude = getFrequencyAmplitudeFromDemodulationLowPass(self.signal.timeProperties.timeEndPoints, spinDemodulated, self.simulationResults.sensedFrequencyAmplitude)

        # plt.plot(timeCoarse, spin*0 + self.simulationResults.sensedFrequencyAmplitude)

        if doPlotSpin:
            plt.figure()
            plt.plot(self.signal.timeProperties.timeCoarse, self.simulationResults.spin[:, :])
            plt.show()

    def writeToFile(self, archive):
        """
        Saves the simulation record to hdf5.

        Parameters
        ----------
        archive : :class:`h5py.Group`
            The HDF5 archive to write to.
        """
        self.sourceProperties.writeToFile(archive)
        self.stateProperties.writeToFile(archive)
        self.simulationResults.writeToFile(archive, False)

@cuda.jit(debug = cudaDebug,  max_registers = 63)
def getTimeEvolutionCommutatorFree4RotatingWave(timeCoarse, timeEndPoints, timeStepFine, timeStepCoarse,
    sourceIndexMax, sourceAmplitude, sourceFrequency, sourcePhase, sourceTimeEndPoints, sourceQuadraticShift,
    timeEvolutionCoarse, trotterCutoff):
    """
    Find the stepwise time evolution opperator using a 2 exponential, commutator free, order 4 Magnus integrator, in a rotating frame. This method compared to the others here has the highest accuracy for a given fine time step (increasing accuracy), or equivalently, can obtain the same accuracy using larger fine time steps (reducing execution time).

    Parameters
    ----------
    timeCoarse : :class:`numpy.ndarray` of :class:`numpy.double` (timeIndex)
        A coarse grained list of time samples that the time evolution operator is found for. This is an output, so use an empty :class:`numpy.ndarray` with :func:`numpy.empty()`, or declare a :class:`numba.cuda.cudadrv.devicearray.DeviceNDArray` using :func:`numba.cuda.device_array_like()`.
    timeEndPoints : :class:`numpy.ndarray` of :class:`numpy.double` (start time (0) or end time (1))
        The time values for when the experiment is to start and finishes.
    timeStepFine : `float`
        The time step used within the integration algorithm.
    timeStepCoarse : `float`
        The time difference between each element of `timeCoarse`. Determines the sample rate of the outputs `timeCoarse` and `timeEvolutionCoarse`.
    sourceIndexMax : `int`
        The number of sources in the simulation.
    sourceAmplitude : :class:`numpy.ndarray` of :class:`numpy.double`, (sourceIndex, spatialIndex)
        The amplitude of the sine wave of source `sourceIndex` in direction `spatialIndex`. In units of Hz.
    sourcePhase : :class:`numpy.ndarray` of :class:`numpy.double`, (sourceIndex, spatialIndex)
        The phase offset of the sine wave of source `sourceIndex` in direction `spatialIndex`. In units of radians.
    sourceFrequency : :class:`numpy.ndarray` of :class:`numpy.double`, (sourceIndex, spatialIndex)
        The frequency of the sine wave of source `sourceIndex` in direction `spatialIndex`. In units of Hz.
    sourceTimeEndPoints : :class:`numpy.ndarray` of :class:`numpy.double`, (sourceIndex, turn on time (0) or turn off time (1))
        The times that the sine wave of source `sourceIndex` turns on and off. In units of s.
    sourceQuadraticShift :  `float`
        The constant quadratic shift of the spin 1 system, in Hz.
    timeEvolutionCoarse : :class:`numpy.ndarray` of :class:`numpy.cdouble` (timeIndex, braStateIndex, ketStateIndex)
        Time evolution operator (matrix) between the current and next timesteps, for each time sampled. See :math:`U(t)` in :ref:`overviewOfSimulationMethod`. This is an output, so use an empty :class:`numpy.ndarray` with :func:`numpy.empty()`, or declare a :class:`numba.cuda.cudadrv.devicearray.DeviceNDArray` using :func:`numba.cuda.device_array_like()`.
    trotterCutoff : `int`
        The number of squares made by the spin 1 matrix exponentiator.
    """

    # Declare variables
    timeEvolutionFine = cuda.local.array((3, 3), dtype = nb.complex128)
    timeEvolutionOld = cuda.local.array((3, 3), dtype = nb.complex128)
    magneticField = cuda.local.array((2, 3), dtype = nb.float64)
    weight = cuda.local.array(2, dtype = nb.float64)
    hamiltonian = cuda.local.array((3, 3), dtype = nb.complex128)
    rotatingWaveWinding = cuda.local.array(2, dtype = nb.complex128)

    # Run calculation for each coarse timestep in parallel
    timeIndex = cuda.threadIdx.x + cuda.blockIdx.x*cuda.blockDim.x
    if timeIndex < timeCoarse.size:
        timeCoarse[timeIndex] = timeEndPoints[0] + timeStepCoarse*timeIndex
        timeFine = timeCoarse[timeIndex]

        # Initialise time evolution operator to 1
        setToOne(timeEvolutionCoarse[timeIndex, :])
        rotatingWave = 2*math.pi*sourceAmplitude[0, 2]

        # For every fine step
        for timeFineIndex in range(math.floor(timeStepCoarse/timeStepFine + 0.5)):
            for timeSampleIndex in range(2):
                # 2nd order quadrature => Sample at +- 1/sqrt(3)
                timeSample = timeFine + 0.5*timeStepFine*(1 + (2*timeSampleIndex - 1)/sqrt3)
                rotatingWaveWinding[timeSampleIndex] = math.cos(rotatingWave*(timeSample - timeCoarse[timeIndex])) + 1j*math.sin(rotatingWave*(timeSample - timeCoarse[timeIndex]))
                for spacialIndex in nb.prange(3):
                    magneticField[timeSampleIndex, spacialIndex] = 0
                for sourceIndex in range(sourceIndexMax):
                    if timeSample >= sourceTimeEndPoints[sourceIndex, 0] and timeSample <= sourceTimeEndPoints[sourceIndex, 1]:
                        for spacialIndex in nb.prange(3):
                            magneticField[timeSampleIndex, spacialIndex] += 2*math.pi*sourceAmplitude[sourceIndex, spacialIndex]*math.sin(2*math.pi*sourceFrequency[sourceIndex, spacialIndex]*(timeSample - sourceTimeEndPoints[sourceIndex, 0]) + sourcePhase[sourceIndex, spacialIndex])
                            
            for exponentialIndex in range(-1, 2, 2):
                # Calculate hamiltonian from magnetic field
                weight[0] = (1.5 - exponentialIndex*sqrt3)/6
                weight[1] = (1.5 + exponentialIndex*sqrt3)/6

                hamiltonian[0, 0] = (-1j*timeStepFine)*(weight[0]*(magneticField[0, 2] + sourceQuadraticShift/3 - rotatingWave) + weight[1]*(magneticField[1, 2] + sourceQuadraticShift/3 - rotatingWave))
                hamiltonian[0, 1] = (-1j*timeStepFine)*(weight[0]*(magneticField[0, 0] - 1j*magneticField[0, 1])*rotatingWaveWinding[0] + weight[1]*(magneticField[1, 0] - 1j*magneticField[1, 1])*rotatingWaveWinding[1])/sqrt2
                hamiltonian[0, 2] = 0
                hamiltonian[1, 0] = (-1j*timeStepFine)*(weight[0]*(magneticField[0, 0] + 1j*magneticField[0, 1])/rotatingWaveWinding[0] + weight[1]*(magneticField[1, 0] + 1j*magneticField[1, 1])/rotatingWaveWinding[1])/sqrt2
                hamiltonian[1, 1] = (-1j*timeStepFine)*(weight[0] + weight[1])*(-2/3)*sourceQuadraticShift
                hamiltonian[1, 2] = hamiltonian[0, 1]
                hamiltonian[2, 0] = 0
                hamiltonian[2, 1] = hamiltonian[1, 0]
                hamiltonian[2, 2] = (-1j*timeStepFine)*(weight[0]*(-magneticField[0, 2] + sourceQuadraticShift/3 + rotatingWave) + weight[1]*(-magneticField[1, 2] + sourceQuadraticShift/3 + rotatingWave))

                # Calculate the exponential from the expansion
                matrixExponentialLieTrotter(hamiltonian, timeEvolutionFine, trotterCutoff)

                # Premultiply to the exitsing time evolution operator
                setTo(timeEvolutionCoarse[timeIndex, :], timeEvolutionOld)
                matrixMultiply(timeEvolutionFine, timeEvolutionOld, timeEvolutionCoarse[timeIndex, :])

            timeFine += timeStepFine

        # Take out of rotating frame
        rotatingWaveWinding[0] = math.cos(rotatingWave*timeStepCoarse) + 1j*math.sin(rotatingWave*timeStepCoarse)
        timeEvolutionCoarse[timeIndex, 0, 0] /= rotatingWaveWinding[0]
        timeEvolutionCoarse[timeIndex, 0, 1] /= rotatingWaveWinding[0]
        timeEvolutionCoarse[timeIndex, 0, 2] /= rotatingWaveWinding[0]

        timeEvolutionCoarse[timeIndex, 2, 0] *= rotatingWaveWinding[0]
        timeEvolutionCoarse[timeIndex, 2, 1] *= rotatingWaveWinding[0]
        timeEvolutionCoarse[timeIndex, 2, 2] *= rotatingWaveWinding[0]

@cuda.jit(debug = cudaDebug,  max_registers = 95)
def getTimeEvolutionCommutatorFree4(timeCoarse, timeEndPoints, timeStepFine, timeStepCoarse,
    sourceIndexMax, sourceAmplitude, sourceFrequency, sourcePhase, sourceTimeEndPoints,
    timeEvolutionCoarse, trotterCutoff):
    """
    Find the stepwise time evolution opperator using a 2 exponential, commutator free, order 4 Magnus integrator.
    
    Parameters
    ----------
    timeCoarse : :class:`numpy.ndarray` of :class:`numpy.double` (timeIndex)
        A coarse grained list of time samples that the time evolution operator is found for. This is an output, so use an empty :class:`numpy.ndarray` with :func:`numpy.empty()`, or declare a :class:`numba.cuda.cudadrv.devicearray.DeviceNDArray` using :func:`numba.cuda.device_array_like()`.
    timeEndPoints : :class:`numpy.ndarray` of :class:`numpy.double` (start time (0) or end time (1))
        The time values for when the experiment is to start and finishes.
    timeStepFine : `float`
        The time step used within the integration algorithm.
    timeStepCoarse : `float`
        The time difference between each element of `timeCoarse`. Determines the sample rate of the outputs `timeCoarse` and `timeEvolutionCoarse`.
    sourceIndexMax : `int`
        The number of sources in the simulation.
    sourceAmplitude : :class:`numpy.ndarray` of :class:`numpy.double`, (sourceIndex, spatialIndex)
        The amplitude of the sine wave of source `sourceIndex` in direction `spatialIndex`. In units of Hz.
    sourcePhase : :class:`numpy.ndarray` of :class:`numpy.double`, (sourceIndex, spatialIndex)
        The phase offset of the sine wave of source `sourceIndex` in direction `spatialIndex`. In units of radians.
    sourceFrequency : :class:`numpy.ndarray` of :class:`numpy.double`, (sourceIndex, spatialIndex)
        The frequency of the sine wave of source `sourceIndex` in direction `spatialIndex`. In units of Hz.
    sourceTimeEndPoints : :class:`numpy.ndarray` of :class:`numpy.double`, (sourceIndex, turn on time (0) or turn off time (1))
        The times that the sine wave of source `sourceIndex` turns on and off. In units of s.
    timeEvolutionCoarse : :class:`numpy.ndarray` of :class:`numpy.cdouble` (timeIndex, braStateIndex, ketStateIndex)
        Time evolution operator (matrix) between the current and next timesteps, for each time sampled. See :math:`U(t)` in :ref:`overviewOfSimulationMethod`. This is an output, so use an empty :class:`numpy.ndarray` with :func:`numpy.empty()`, or declare a :class:`numba.cuda.cudadrv.devicearray.DeviceNDArray` using :func:`numba.cuda.device_array_like()`.
    trotterCutoff : `int`
        The number of squares made by the spin 1 matrix exponentiator.
    """

    # Declare variables
    timeEvolutionFine = cuda.local.array((3, 3), dtype = nb.complex128)
    timeEvolutionOld = cuda.local.array((3, 3), dtype = nb.complex128)
    magneticField = cuda.local.array((2, 3), dtype = nb.float64)
    weight = cuda.local.array(2, dtype = nb.float64)
    hamiltonian = cuda.local.array((3, 3), dtype = nb.complex128)

    # Run calculation for each coarse timestep in parallel
    timeIndex = cuda.threadIdx.x + cuda.blockIdx.x*cuda.blockDim.x
    if timeIndex < timeCoarse.size:
        timeCoarse[timeIndex] = timeEndPoints[0] + timeStepCoarse*timeIndex
        timeFine = timeCoarse[timeIndex]

        # Initialise time evolution operator to 1
        setToOne(timeEvolutionCoarse[timeIndex, :])

        # For every fine step
        for timeFineIndex in range(math.floor(timeStepCoarse/timeStepFine + 0.5)):
            for timeSampleIndex in range(2):
                for spacialIndex in nb.prange(3):
                    magneticField[timeSampleIndex, spacialIndex] = 0
                for sourceIndex in range(sourceIndexMax):
                    # 2nd order quadrature => Sample at +- 1/sqrt(3)
                    timeSample = timeFine + 0.5*timeStepFine*(1 + (2*timeSampleIndex - 1)/sqrt3)
                    if timeSample >= sourceTimeEndPoints[sourceIndex, 0] and timeSample <= sourceTimeEndPoints[sourceIndex, 1]:
                        for spacialIndex in nb.prange(3):
                            magneticField[timeSampleIndex, spacialIndex] += 2*math.pi*sourceAmplitude[sourceIndex, spacialIndex]*math.sin(2*math.pi*sourceFrequency[sourceIndex, spacialIndex]*(timeSample - sourceTimeEndPoints[sourceIndex, 0]) + sourcePhase[sourceIndex, spacialIndex])
                            
            for exponentialIndex in range(-1, 2, 2):
                # Calculate hamiltonian from magnetic field
                weight[0] = (1.5 - exponentialIndex*sqrt3)/6
                weight[1] = (1.5 + exponentialIndex*sqrt3)/6

                hamiltonian[0, 0] = (-1j*timeStepFine)*(weight[0]*magneticField[0, 2] + weight[1]*magneticField[1, 2])
                hamiltonian[0, 1] = (-1j*timeStepFine)*(weight[0]*(magneticField[0, 0] - 1j*magneticField[0, 1]) + weight[1]*(magneticField[1, 0] - 1j*magneticField[1, 1]))/sqrt2
                hamiltonian[0, 2] = 0
                hamiltonian[1, 0] = (-1j*timeStepFine)*(weight[0]*(magneticField[0, 0] + 1j*magneticField[0, 1]) + weight[1]*(magneticField[1, 0] + 1j*magneticField[1, 1]))/sqrt2
                hamiltonian[1, 1] = 0
                hamiltonian[1, 2] = hamiltonian[0, 1]
                hamiltonian[2, 0] = 0
                hamiltonian[2, 1] = hamiltonian[1, 0]
                hamiltonian[2, 2] = -hamiltonian[0, 0]

                # Calculate the exponential from the expansion
                matrixExponentialLieTrotter(hamiltonian, timeEvolutionFine, trotterCutoff)

                # Premultiply to the exitsing time evolution operator
                setTo(timeEvolutionCoarse[timeIndex, :], timeEvolutionOld)
                matrixMultiply(timeEvolutionFine, timeEvolutionOld, timeEvolutionCoarse[timeIndex, :])

            timeFine += timeStepFine

@cuda.jit(debug = cudaDebug,  max_registers = 95)
def getTimeEvolutionHalfStep(timeCoarse, timeEndPoints, timeStepFine, timeStepCoarse,
    sourceIndexMax, sourceAmplitude, sourceFrequency, sourcePhase, sourceTimeEndPoints,
    timeEvolutionCoarse, trotterCutoff):
    """
    The same sampling as used in the cython code. Uses two exponentials per fine timestep.

    Parameters
    ----------
    timeCoarse : :class:`numpy.ndarray` of :class:`numpy.double` (timeIndex)
        A coarse grained list of time samples that the time evolution operator is found for. This is an output, so use an empty :class:`numpy.ndarray` with :func:`numpy.empty()`, or declare a :class:`numba.cuda.cudadrv.devicearray.DeviceNDArray` using :func:`numba.cuda.device_array_like()`.
    timeEndPoints : :class:`numpy.ndarray` of :class:`numpy.double` (start time (0) or end time (1))
        The time values for when the experiment is to start and finishes.
    timeStepFine : `float`
        The time step used within the integration algorithm.
    timeStepCoarse : `float`
        The time difference between each element of `timeCoarse`. Determines the sample rate of the outputs `timeCoarse` and `timeEvolutionCoarse`.
    sourceIndexMax : `int`
        The number of sources in the simulation.
    sourceAmplitude : :class:`numpy.ndarray` of :class:`numpy.double`, (sourceIndex, spatialIndex)
        The amplitude of the sine wave of source `sourceIndex` in direction `spatialIndex`. In units of Hz.
    sourcePhase : :class:`numpy.ndarray` of :class:`numpy.double`, (sourceIndex, spatialIndex)
        The phase offset of the sine wave of source `sourceIndex` in direction `spatialIndex`. In units of radians.
    sourceFrequency : :class:`numpy.ndarray` of :class:`numpy.double`, (sourceIndex, spatialIndex)
        The frequency of the sine wave of source `sourceIndex` in direction `spatialIndex`. In units of Hz.
    sourceTimeEndPoints : :class:`numpy.ndarray` of :class:`numpy.double`, (sourceIndex, turn on time (0) or turn off time (1))
        The times that the sine wave of source `sourceIndex` turns on and off. In units of s.
    timeEvolutionCoarse : :class:`numpy.ndarray` of :class:`numpy.cdouble` (timeIndex, braStateIndex, ketStateIndex)
        Time evolution operator (matrix) between the current and next timesteps, for each time sampled. See :math:`U(t)` in :ref:`overviewOfSimulationMethod`. This is an output, so use an empty :class:`numpy.ndarray` with :func:`numpy.empty()`, or declare a :class:`numba.cuda.cudadrv.devicearray.DeviceNDArray` using :func:`numba.cuda.device_array_like()`.
    trotterCutoff : `int`
        The number of squares made by the spin 1 matrix exponentiator.
    """

    # Declare variables
    timeEvolutionFine = cuda.local.array((3, 3), dtype = nb.complex128)
    timeEvolutionOld = cuda.local.array((3, 3), dtype = nb.complex128)
    magneticField = cuda.local.array(3, dtype = nb.float64)
    hamiltonian = cuda.local.array((3, 3), dtype = nb.complex128)

    # Run calculation for each coarse timestep in parallel
    timeIndex = cuda.threadIdx.x + cuda.blockIdx.x*cuda.blockDim.x
    if timeIndex < timeCoarse.size:
        timeCoarse[timeIndex] = timeEndPoints[0] + timeStepCoarse*timeIndex
        timeFine = timeCoarse[timeIndex]

        # Initialise time evolution operator to 1
        setToOne(timeEvolutionCoarse[timeIndex, :])

        # For every fine step
        for timeFineIndex in range(math.floor(timeStepCoarse/timeStepFine + 0.5)):
            for timeSampleIndex in range(2):
                if 1 + timeFine + timeSampleIndex*timeStepFine > 0:
                    for spacialIndex in nb.prange(3):
                        magneticField[spacialIndex] = 0
                    for sourceIndex in range(sourceIndexMax):
                        if timeFine + timeSampleIndex*timeStepFine >= sourceTimeEndPoints[sourceIndex, 0] and timeFine + timeSampleIndex*timeStepFine <= sourceTimeEndPoints[sourceIndex, 1]:
                            for spacialIndex in nb.prange(3):
                                magneticField[spacialIndex] += 2*math.pi*sourceAmplitude[sourceIndex, spacialIndex]*math.sin(2*math.pi*sourceFrequency[sourceIndex, spacialIndex]*(timeFine + timeSampleIndex*timeStepFine - sourceTimeEndPoints[sourceIndex, 0]) + sourcePhase[sourceIndex, spacialIndex])

                    # Calculate hamiltonian from magnetic field
                    hamiltonian[0, 0] = (-1j*timeStepFine/2)*magneticField[2]
                    hamiltonian[0, 1] = (-1j*timeStepFine/2)*(magneticField[0] - 1j*magneticField[1])/sqrt2
                    hamiltonian[0, 2] = 0
                    hamiltonian[1, 0] = (-1j*timeStepFine/2)*(magneticField[0] + 1j*magneticField[1])/sqrt2
                    hamiltonian[1, 1] = 0
                    hamiltonian[1, 2] = (-1j*timeStepFine/2)*(magneticField[0] - 1j*magneticField[1])/sqrt2
                    hamiltonian[2, 0] = 0
                    hamiltonian[2, 1] = (-1j*timeStepFine/2)*(magneticField[0] + 1j*magneticField[1])/sqrt2
                    hamiltonian[2, 2] = (-1j*timeStepFine/2)*(-magneticField[2])

                    matrixExponentialLieTrotter(hamiltonian, timeEvolutionFine, trotterCutoff)

                    setTo(timeEvolutionCoarse[timeIndex, :], timeEvolutionOld)
                    matrixMultiply(timeEvolutionFine, timeEvolutionOld, timeEvolutionCoarse[timeIndex, :])

            timeFine += timeStepFine

@cuda.jit(debug = cudaDebug,  max_registers = 95)
def getTimeEvolutionMidpointSample(timeCoarse, timeEndPoints, timeStepFine, timeStepCoarse,
    sourceIndexMax, sourceAmplitude, sourceFrequency, sourcePhase, sourceTimeEndPoints,
    timeEvolutionCoarse, trotterCutoff):
    """
    The most basic form of integrator, uses one exponential per time step.

    Parameters
    ----------
    timeCoarse : :class:`numpy.ndarray` of :class:`numpy.double` (timeIndex)
        A coarse grained list of time samples that the time evolution operator is found for. This is an output, so use an empty :class:`numpy.ndarray` with :func:`numpy.empty()`, or declare a :class:`numba.cuda.cudadrv.devicearray.DeviceNDArray` using :func:`numba.cuda.device_array_like()`.
    timeEndPoints : :class:`numpy.ndarray` of :class:`numpy.double` (start time (0) or end time (1))
        The time values for when the experiment is to start and finishes.
    timeStepFine : `float`
        The time step used within the integration algorithm.
    timeStepCoarse : `float`
        The time difference between each element of `timeCoarse`. Determines the sample rate of the outputs `timeCoarse` and `timeEvolutionCoarse`.
    sourceIndexMax : `int`
        The number of sources in the simulation.
    sourceAmplitude : :class:`numpy.ndarray` of :class:`numpy.double`, (sourceIndex, spatialIndex)
        The amplitude of the sine wave of source `sourceIndex` in direction `spatialIndex`. In units of Hz.
    sourcePhase : :class:`numpy.ndarray` of :class:`numpy.double`, (sourceIndex, spatialIndex)
        The phase offset of the sine wave of source `sourceIndex` in direction `spatialIndex`. In units of radians.
    sourceFrequency : :class:`numpy.ndarray` of :class:`numpy.double`, (sourceIndex, spatialIndex)
        The frequency of the sine wave of source `sourceIndex` in direction `spatialIndex`. In units of Hz.
    sourceTimeEndPoints : :class:`numpy.ndarray` of :class:`numpy.double`, (sourceIndex, turn on time (0) or turn off time (1))
        The times that the sine wave of source `sourceIndex` turns on and off. In units of s.
    timeEvolutionCoarse : :class:`numpy.ndarray` of :class:`numpy.cdouble` (timeIndex, braStateIndex, ketStateIndex)
        Time evolution operator (matrix) between the current and next timesteps, for each time sampled. See :math:`U(t)` in :ref:`overviewOfSimulationMethod`. This is an output, so use an empty :class:`numpy.ndarray` with :func:`numpy.empty()`, or declare a :class:`numba.cuda.cudadrv.devicearray.DeviceNDArray` using :func:`numba.cuda.device_array_like()`.
    trotterCutoff : `int`
        The number of squares made by the spin 1 matrix exponentiator.
    """

    # Declare variables
    timeEvolutionFine = cuda.local.array((3, 3), dtype = nb.complex128)
    timeEvolutionOld = cuda.local.array((3, 3), dtype = nb.complex128)
    magneticField = cuda.local.array(3, dtype = nb.float64)
    hamiltonian = cuda.local.array((3, 3), dtype = nb.complex128)

    # Run calculation for each coarse timestep in parallel
    timeIndex = cuda.threadIdx.x + cuda.blockIdx.x*cuda.blockDim.x
    if timeIndex < timeCoarse.size:
        timeCoarse[timeIndex] = timeEndPoints[0] + timeStepCoarse*timeIndex
        timeFine = timeCoarse[timeIndex]

        # Initialise time evolution operator to 1
        setToOne(timeEvolutionCoarse[timeIndex, :])

        # For every fine step
        for timeFineIndex in range(math.floor(timeStepCoarse/timeStepFine + 0.5)):
            # Calculate magnetic field from sources
            for spacialIndex in nb.prange(3):
                magneticField[spacialIndex] = 0
            for sourceIndex in range(sourceIndexMax):
                if timeFine + 0.5*timeStepFine >= sourceTimeEndPoints[sourceIndex, 0] and timeFine + 0.5*timeStepFine <= sourceTimeEndPoints[sourceIndex, 1]:
                    for spacialIndex in nb.prange(3):
                        magneticField[spacialIndex] += 2*math.pi*sourceAmplitude[sourceIndex, spacialIndex]*math.sin(2*math.pi*sourceFrequency[sourceIndex, spacialIndex]*(timeFine + 0.5*timeStepFine - sourceTimeEndPoints[sourceIndex, 0]) + sourcePhase[sourceIndex, spacialIndex])

            # Calculate hamiltonian from magnetic field
            hamiltonian = cuda.local.array((3, 3), dtype = nb.complex128)
            hamiltonian[0, 0] = (-1j*timeStepFine)*magneticField[2]
            hamiltonian[0, 1] = (-1j*timeStepFine)*(magneticField[0] - 1j*magneticField[1])/sqrt2
            hamiltonian[0, 2] = 0
            hamiltonian[1, 0] = (-1j*timeStepFine)*(magneticField[0] + 1j*magneticField[1])/sqrt2
            hamiltonian[1, 1] = 0
            hamiltonian[1, 2] = (-1j*timeStepFine)*(magneticField[0] - 1j*magneticField[1])/sqrt2
            hamiltonian[2, 0] = 0
            hamiltonian[2, 1] = (-1j*timeStepFine)*(magneticField[0] + 1j*magneticField[1])/sqrt2
            hamiltonian[2, 2] = (-1j*timeStepFine)*(-magneticField[2])

            matrixExponentialLieTrotter(hamiltonian, timeEvolutionFine, trotterCutoff)

            setTo(timeEvolutionCoarse[timeIndex, :], timeEvolutionOld)
            matrixMultiply(timeEvolutionFine, timeEvolutionOld, timeEvolutionCoarse[timeIndex, :])

            timeFine += timeStepFine

@nb.jit(nopython = True)
def getState(stateInit, state, timeEvolution):
    """
    Use the stepwise time evolution operators in succession to find the quantum state timeseries of the 3 level atom.

    Parameters
    ----------
    stateInit : :class:`numpy.ndarray` of :class:`numpy.cdouble`
        The state (spin wavefunction) of the system at the start of the simulation.
    state : :class:`numpy.ndarray` of :class:`numpy.cdouble` (timeIndex, stateIndex)
        The state (wavefunction) of the spin system in the lab frame, for each time sampled. See :math:`\\psi(t)` in :ref:`overviewOfSimulationMethod`. This is an output.
    timeEvolution : :class:`numpy.ndarray` of :class:`numpy.cdouble` (timeIndex, braStateIndex, ketStateIndex)
        Time evolution operator (matrix) between the current and next timesteps, for each time sampled. See :math:`U(t)` in :ref:`overviewOfSimulationMethod`.
    """
    for timeIndex in range(state.shape[0]):
        # State = time evolution * previous state
        norm = 0
        for xIndex in nb.prange(3):
            state[timeIndex, xIndex] = 0
            if timeIndex > 0:
                for zIndex in range(3):
                    state[timeIndex, xIndex] += timeEvolution[timeIndex - 1, xIndex, zIndex]*state[timeIndex - 1, zIndex]
            else:
                state[timeIndex, xIndex] += stateInit[xIndex]
            norm += (state[timeIndex, xIndex]*np.conj(state[timeIndex, xIndex])).real

        # Normalise the state in case of errors in the unitarity of the time evolution operator
        # for xIndex in nb.prange(3):
        #     state[timeIndex, xIndex] /=np.sqrt(norm)

@cuda.jit(debug = cudaDebug)
def getSpin(state, spin):
    """
    Calculate each expected spin value in parallel.

    Parameters
    ----------
    state : :class:`numpy.ndarray` of :class:`numpy.cdouble` (timeIndex, stateIndex)
        The state (wavefunction) of the spin system in the lab frame, for each time sampled. See :math:`\\psi(t)` in :ref:`overviewOfSimulationMethod`.
    spin : :class:`numpy.ndarray` of :class:`numpy.double` (timeIndex, spatialIndex)
        The expected value for hyperfine spin of the spin system in the lab frame, for each time sampled. Units of :math:`\\hbar`. This is an output, so use an empty :class:`numpy.ndarray` with :func:`numpy.empty()`, or declare a :class:`numba.cuda.cudadrv.devicearray.DeviceNDArray` using :func:`numba.cuda.device_array_like()`.
    """
    timeIndex = cuda.threadIdx.x + cuda.blockIdx.x*cuda.blockDim.x
    if timeIndex < spin.shape[0]:
        spin[timeIndex, 0] = (2*conj(state[timeIndex, 1])*(state[timeIndex, 0] + state[timeIndex, 2])/sqrt2).real
        spin[timeIndex, 1] = (2j*conj(state[timeIndex, 1])*(state[timeIndex, 0] - state[timeIndex, 2])/sqrt2).real
        spin[timeIndex, 2] = state[timeIndex, 0].real**2 + state[timeIndex, 0].imag**2 - state[timeIndex, 2].real**2 - state[timeIndex, 2].imag**2

@cuda.jit(debug = cudaDebug)
def getFrequencyAmplitudeFromDemodulationMultiply(time, spin, spinDemodulated, biasAmplitude):
    """
    Multiply each spin value by :math:`-2\\cos(2\\pi f_\\mathrm{bias, amp} t)` as part of a demodulation.

    Parameters
    ----------
    time : :class:`numpy.ndarray` of :class:`numpy.double` (timeIndex)
        The time at which each spin sample is taken.
    spin : :class:`numpy.ndarray` of :class:`numpy.double` (timeIndex, spatialIndex)
        The expected value for hyperfine spin of the spin system in the lab frame, for each time sampled. Units of :math:`\\hbar`.
    spinDemodulated : :class:`numpy.ndarray` of :class:`numpy.double` (timeIndex, spatialIndex)
        The expected value for hyperfine spin of the spin system in the lab frame, multiplied by :math:`-2\\cos(2\\pi f_\\mathrm{bias, amp} t)` for each time sampled. Units of :math:`\\hbar`. This is an output, so use an empty :class:`numpy.ndarray` with :func:`numpy.empty()`, or declare a :class:`numba.cuda.cudadrv.devicearray.DeviceNDArray` using :func:`numba.cuda.device_array_like()`.
    biasAmplitude : `float`
        The carrier frequency :math:`f_\\mathrm{bias, amp}` for which to demodulate by.
    """
    timeIndex = cuda.threadIdx.x + cuda.blockIdx.x*cuda.blockDim.x
    if timeIndex < spin.shape[0]:
        spinDemodulated[timeIndex] = -2*math.cos(2*math.pi*biasAmplitude*time[timeIndex])*spin[timeIndex]

@nb.jit(nopython = True)
def getFrequencyAmplitudeFromDemodulationLowPass(timeEndPoints, spin, sensedFrequencyAmplitude):
    """
    Average the multiplied spin to find the DC value (ie apply a low pass filter).

    Parameters
    ----------
    timeEndPoints : :class:`numpy.ndarray` of :class:`numpy.double` (start time (0) or end time (1))
        The bounds of the interval where Faraday demodulation is used to acquire the measured frequency amplitude (Fourier coefficient) required for reconstruction
    spin : :class:`numpy.ndarray` of :class:`numpy.double` (timeIndex, spatialIndex)
        The expected value for hyperfine spin of the spin system in the lab frame, multiplied by :math:`-2\\cos(2\\pi f_\\mathrm{bias, amp} t)` for each time sampled. Units of :math:`\\hbar`. The output `spinDemodulated` from :func:`getFrequencyAmplitudeFromDemodulationMultiply()`.
    sensedFrequencyAmplitude : `float`
        The measured Fourier coefficient from the simulation. This is an output.
    """
    sensedFrequencyAmplitude = 0.0
    # spin = 1/2 g T coefficient
    for timeIndex in range(spin.size):
        sensedFrequencyAmplitude += spin[timeIndex]
    sensedFrequencyAmplitude *= -1/(2*math.pi*spin.size*(timeEndPoints[1] - timeEndPoints[0]))
    return sensedFrequencyAmplitude

@cuda.jit(debug = cudaDebug)
def getFrequencyAmplitudeFromDemodulation(time, spin, spinDemodulated, biasAmplitude):
    """
    Demodulate a spin timeseries with a basic block low pass filter.

    Parameters
    ----------
    time : :class:`numpy.ndarray` of :class:`numpy.double` (timeIndex)
        The time at which each spin sample is taken.
    spin : :class:`numpy.ndarray` of :class:`numpy.double` (timeIndex, spatialIndex)
        The expected value for hyperfine spin of the spin system in the lab frame, for each time sampled. Units of :math:`\\hbar`.
    spinDemodulated : :class:`numpy.ndarray` of :class:`numpy.double` (timeIndex, spatialIndex)
        The expected value for hyperfine spin of the spin system in the lab frame, demodulated by `biasAmplitude` for each time sampled. Units of :math:`\\hbar`. This is an output, so use an empty :class:`numpy.ndarray` with :func:`numpy.empty()`, or declare a :class:`numba.cuda.cudadrv.devicearray.DeviceNDArray` using :func:`numba.cuda.device_array_like()`.
    biasAmplitude : `float`
        The carrier frequency :math:`f_\\mathrm{bias, amp}` for which to demodulate by.
    """
    timeIndex = cuda.threadIdx.x + cuda.blockIdx.x*cuda.blockDim.x
    if timeIndex < spin.shape[0]:
        spinDemodulated[timeIndex] = 0
        for timeIndexOffset in range(-50, 51):
            timeIndexUse = timeIndex + timeIndexOffset
            if timeIndexUse < 0:
                timeIndexUse = 0
            elif timeIndexUse > spin.shape[0] - 1:
                timeIndexUse = spin.shape[0] - 1
            spinDemodulated[timeIndex] -= 2*math.cos(2*math.pi*biasAmplitude*time[timeIndexUse])*spin[timeIndexUse]/101