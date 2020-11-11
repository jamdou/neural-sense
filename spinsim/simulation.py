"""
.. _overviewOfSimulationMethod:

********************
Table of integrators
********************

Below is the list of configurations for the integrators developed for this project. The highest performing are :func:`getTimeEvolutionSo()` for spin one simulations, and :func:`getTimeEvolutionSh()` for spin half simulations, but the rest are included for the sake of benchmarking.

+---------+--------------+-----------------------------------+-----------------------------------+-----------------------------------+-----------------------------------+
|                        |Cubic interpolator                                                                                         |Linear interpolator                |
+---------+--------------+-----------------------------------+-----------------------------------+-----------------------------------+-----------------------------------+
|         |              |Fourth order commutator free Magnus|Half step                          |Midpoint Sample                    |Fourth order commutator free Magnus|
+---------+--------------+-----------------------------------+-----------------------------------+-----------------------------------+-----------------------------------+
|Spin half|Rotating frame|:func:`getTimeEvolutionSh()`       |                                   |                                   |                                   |
+---------+--------------+-----------------------------------+-----------------------------------+-----------------------------------+-----------------------------------+
|Spin one |Lab frame     |:func:`getTimeEvolutionSoCf4LfCi()`|:func:`getTimeEvolutionSoHaLfCi()` |:func:`getTimeEvolutionSoMsLfCi()` |                                   |
+---------+--------------+-----------------------------------+-----------------------------------+-----------------------------------+-----------------------------------+
|         |Rotating frame|:func:`getTimeEvolutionSo()`       |                                   |                                   |:func:`getTimeEvolutionSoCf4RfLi()`|
+---------+--------------+-----------------------------------+-----------------------------------+-----------------------------------+-----------------------------------+

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

*********
Reference
*********
"""

import numpy as np
import scipy.linalg as spla
import matplotlib.pyplot as plt
import math, cmath
from numba import cuda
import numba as nb
import time as tm
from testSignal import *
from . import simulationUtilities as utilities
from enum import Enum, auto

#===============================================================#

# Important constants
cudaDebug = False
sqrt2 = math.sqrt(2)
sqrt3 = math.sqrt(3)
expPrecision = 5                                # Where to cut off the exp Taylor series
machineEpsilon = np.finfo(np.float64).eps*1000  # When to decide that vectors are parallel
# trotterCutoff = 52

interpolate = utilities.interpolateSourceCubic

class SourceProperties:
    """
    A list of sine wave parameters fed into the simulation code.

    The source is parametrised as
    
    .. math::
        \\begin{align*}
            b_{i,x}(t) &= 2 \\pi f_{\\textrm{amp},i,x}\\sin(2 \\pi f_{i,x}(t -\\tau_{i,0}) + \\phi_{i,x})\\\\
            r_x(t) &= \\sum_i b_{i,x}(t)
        \\end{align*}
    
    Attributes
    ----------
    dressingRabiFrequency : `float`
        The amplitude of the dressing in units of Hz.
    sourceIndexMax : `int`
        The number of sources in the simulation.
    sourceAmplitude : :class:`numpy.ndarray` of :class:`numpy.double`, (sourceIndex, spatialIndex)
        The amplitude of the sine wave of source `sourceIndex` in direction `spatialIndex`. See :math:`f_\\textrm{amp}` above. In units of Hz.
    sourcePhase : :class:`numpy.ndarray` of :class:`numpy.double`, (sourceIndex, spatialIndex)
        The phase offset of the sine wave of source `sourceIndex` in direction `spatialIndex`. See :math:`\\phi` above. In units of radians.
    sourceFrequency : :class:`numpy.ndarray` of :class:`numpy.double`, (sourceIndex, spatialIndex)
        The frequency of the sine wave of source `sourceIndex` in direction `spatialIndex`. See :math:`f` above. In units of Hz.
    sourceTimeEndPoints : :class:`numpy.ndarray` of :class:`numpy.double`, (sourceIndex, turn on time (0) or turn off time (1))
        The times that the sine wave of source `sourceIndex` turns on and off. See :math:`\\tau` above. In units of s.
    sourceQuadraticShift :  `float`
        The constant quadratic shift of the spin 1 system, in Hz.
    sourceType : :class:`numpy.ndarray` of :class:`numpy.double`, (sourceIndex)
        A string description of what source sourceIndex physically represents. Mainly for archive purposes.
    """
    def __init__(self, signal, stateProperties, dressingRabiFrequency = 1000.0, quadraticShift = 0.0):
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

        timeSourceIndexMax = int((signal.timeProperties.timeEndPoints[1] - signal.timeProperties.timeEndPoints[0])/signal.timeProperties.timeStepSource)
        self.source = cuda.device_array((timeSourceIndexMax, 4), dtype = np.double)

        threadsPerBlock = 64
        blocksPerGrid = (timeSourceIndexMax + (threadsPerBlock - 1)) // threadsPerBlock
        evaluateDressing[blocksPerGrid, threadsPerBlock](signal.timeProperties.timeStepSource, self.sourceIndexMax, cuda.to_device(self.sourceAmplitude), cuda.to_device(self.sourceFrequency), cuda.to_device(self.sourcePhase), cuda.to_device(self.sourceTimeEndPoints), self.sourceQuadraticShift, self.source)

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
        sourceType[0] = sinusoidalNoise.type

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

class SpinQuantumNumber(Enum):
    """
    The spin quantum number of the system being simulated. 

    Parameters
    ----------
    value : `int`
        Dimension of the hilbert space the states belong to.
    label : `string`
        What to write in the HDF5 archive file.
    """

    def __init__(self, value, dimension, utilitySet, label):
        super().__init__()
        self._value_ = value
        self.dimension = dimension
        self.utilitySet = utilitySet
        self.label = label

    HALF = (1/2, 2, utilities.spinHalf, "half")
    """
    For two level systems.
    """

    ONE = (1, 3, utilities.spinOne, "one")
    """
    For three level systems.
    """

class StateProperties:
    """
    The initial state fed into the simulation code.

    Attributes
    ----------
    spinQuantumNumber: :class:`SpinQuantumNumber(Enum)`
        The spin quantum number of the system being simulated. Determines the dimension of the hilbert space of the state, the algorithms used in the simulation, etc.
    stateInit : :class:`numpy.ndarray` of :class:`numpy.cdouble` (stateIndex)
        The state (spin wavefunction) of the system at the start of the simulation.
    """
    def __init__(self, spinQuantumNumber = SpinQuantumNumber.HALF, stateInit = None):
        """
        Parameters
        ----------
        spinQuantumNumber: :class:`SpinQuantumNumber(Enum)`
            The spin quantum number of the system being simulated. Determines the dimension of the hilbert space of the state, the algorithms used in the simulation, etc.
        stateInit : :class:`numpy.ndarray` of :class:`numpy.cdouble`
            The state (spin wavefunction) of the system at the start of the simulation.
        """
        self.spinQuantumNumber = spinQuantumNumber
        if stateInit:
            self.stateInit = np.asarray(stateInit, np.cdouble)
        else:
            if self.spinQuantumNumber == SpinQuantumNumber.HALF:
                self.stateInit = np.asarray([1, 0], np.cdouble)
            else:
                self.stateInit = np.asarray([1, 0, 0], np.cdouble)

    def writeToFile(self, archive):
        """
        Parameters
        ----------
        archive : :class:`h5py.Group`
            The HDF5 archive to write to.
        """
        archiveGroup = archive.require_group("stateProperties")
        archiveGroup["stateInit"] = self.stateInit
        archiveGroup["spinQuantumNumber"] = np.asarray(self.spinQuantumNumber.label, dtype='|S32')

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
    def __init__(self, signal, stateProperties):
        """
        Parameters
        ----------
        signal : :class:`testSignal.TestSignal`
            Defines the sampling time for the simulation results.
        stateProperties : :class:`StateProperties`
            Defines the hilbert space dimension for the simulation results.
        """
        self.timeEvolution = np.empty([signal.timeProperties.timeIndexMax, stateProperties.spinQuantumNumber.dimension, stateProperties.spinQuantumNumber.dimension], np.cdouble)
        self.state = np.empty([signal.timeProperties.timeIndexMax, stateProperties.spinQuantumNumber.dimension], np.cdouble)
        self.spin = np.empty([signal.timeProperties.timeIndexMax, 3], np.double)
        self.sensedFrequencyAmplitude = 0.0
        self.sensedFrequencyAmplitudeMethod = "none"

    def writeToFile(self, archive, doWriteEverything = False):
        """
        Saves results to the hdf5 file.

        Parameters
        ----------
        archive : :class:`h5py.Group`
            The HDF5 archive to write to.
        doWriteEverything : `boolean`, optional
            If `True`, then save all time series data to file as well as parametric data. Defaults to `False` to reduce archive file size.
        """
        archiveGroup = archive.require_group("simulationResults")
        if doWriteEverything:
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
    def __init__(self, signal, dressingRabiFrequency = 1e3, stateProperties = None, trotterCutoff = 28):
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
        self.stateProperties = stateProperties
        if not self.stateProperties:
            self.stateProperties = StateProperties()
        self.sourceProperties = SourceProperties(self.signal, self.stateProperties, dressingRabiFrequency)
        self.simulationResults = SimulationResults(self.signal, self.stateProperties)
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

        getTimeEvolution = timeEvolverFactory(self.stateProperties.spinQuantumNumber)
        getTimeEvolution[blocksPerGrid, threadsPerBlock](self.signal.timeProperties.timeCoarse, cuda.to_device(self.signal.timeProperties.timeEndPoints), self.signal.timeProperties.timeStepFine, self.signal.timeProperties.timeStepCoarse, self.signal.timeProperties.timeStepSource, cuda.to_device(self.sourceProperties.source), self.simulationResults.timeEvolution)

        # if self.stateProperties.spinQuantumNumber == SpinQuantumNumber.ONE:
        #     getTimeEvolutionSo[blocksPerGrid, threadsPerBlock](self.signal.timeProperties.timeCoarse, cuda.to_device(self.signal.timeProperties.timeEndPoints), self.signal.timeProperties.timeStepFine, self.signal.timeProperties.timeStepCoarse, self.signal.timeProperties.timeStepSource, cuda.to_device(self.sourceProperties.source), self.simulationResults.timeEvolution, self.trotterCutoff)

        # elif self.stateProperties.spinQuantumNumber == SpinQuantumNumber.HALF:
        #     getTimeEvolutionSh[blocksPerGrid, threadsPerBlock](self.signal.timeProperties.timeCoarse, cuda.to_device(self.signal.timeProperties.timeEndPoints), self.signal.timeProperties.timeStepFine, self.signal.timeProperties.timeStepCoarse, self.signal.timeProperties.timeStepSource, cuda.to_device(self.sourceProperties.source), self.simulationResults.timeEvolution, self.trotterCutoff)

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

        if self.stateProperties.spinQuantumNumber == SpinQuantumNumber.HALF:
            self.simulationResults.sensedFrequencyAmplitude *= 2

        # plt.plot(timeCoarse, spin*0 + self.simulationResults.sensedFrequencyAmplitude)

        if doPlotSpin:
            plt.figure()
            plt.plot(self.signal.timeProperties.timeCoarse, self.simulationResults.spin[:, :])
            plt.show()

    def writeToFile(self, archive, doWriteEverything = False):
        """
        Saves the simulation record to hdf5.

        Parameters
        ----------
        archive : :class:`h5py.Group`
            The HDF5 archive to write to.
        doWriteEverything : `boolean`, optional
            If `True`, then save all time series data to file as well as parametric data. Defaults to `False` to reduce archive file size.
        """
        self.sourceProperties.writeToFile(archive)
        self.stateProperties.writeToFile(archive)
        self.simulationResults.writeToFile(archive, doWriteEverything)

class IntegrationMethod(Enum):
    def __init__(self, value, description):
        super().__init__()
        self._value_ = value
        self.description = description

    MAGNUS_CF_4 = ("magnusCF4", "Commutator free fourth order Magnus based")
    """
    Commutator free fourth order Magnus based.
    """

    MIDPOINT_SAMPLE = ("midpointSample", "Single sample")
    """
    Naive integration method.
    """

    HALF_STEP = ("halfStep", "Naive double sample")
    """
    Integration method from AtomicPy.
    """

class ExponentiationMethod(Enum):
    def __init__(self, value, index):
        super().__init__()
        self._value_ = value
        self.index = index

    ANALYTIC = ("lieTrotter", 0)
    """
    Analytic expression for spin half systems only.
    """

    LIE_TROTTER = ("lieTrotter", 1)
    """
    Approximation using the Lie Trotter theorem.
    """

def timeEvolverFactory(spinQuantumNumber, useRotatingFrame = True, integrationMethod = IntegrationMethod.MAGNUS_CF_4, exponentiationMethod = ExponentiationMethod.LIE_TROTTER, trotterCutoff = 28):
    """
    Makes a time evolver just for you.
    """
    dimension = spinQuantumNumber.dimension
    lieDimension = dimension + 1
    utilitySet = spinQuantumNumber.utilitySet

    if integrationMethod == IntegrationMethod.MAGNUS_CF_4:
        sampleIndexMax = 3
    elif integrationMethod == IntegrationMethod.HALF_STEP:
        sampleIndexMax = 3
    elif integrationMethod == IntegrationMethod.MIDPOINT_SAMPLE:
        sampleIndexMax = 1
    sampleIndexEnd = sampleIndexMax - 1

    exponentiationMethodIndex = exponentiationMethod.index
    if exponentiationMethod == ExponentiationMethod.ANALYTIC and spinQuantumNumber != SpinQuantumNumber.HALF:
        print("\033[31mspinsim warning!!!\nAttempting to use an analytic exponentiation method outside of spin half. Switching to a Lie Trotter method.\033[0m")
        exponentiationMethod = ExponentiationMethod.LIE_TROTTER
        exponentiationMethodIndex = 1

    @cuda.jit(device = True, inline = True)
    def appendExponentiation(sourceSample, timeEvolutionFine, timeEvolutionCoarse):
        timeEvolutionOld = cuda.local.array((dimension, dimension), dtype = nb.complex128)

        # Calculate the exponential
        if exponentiationMethodIndex == 1:
            utilitySet.matrixExponentialLieTrotter(sourceSample, timeEvolutionFine, trotterCutoff)
        elif exponentiationMethodIndex == 0:
            utilities.spinHalf.matrixExponentialAnalytic(sourceSample, timeEvolutionFine)

        # Premultiply to the exitsing time evolution operator
        utilitySet.setTo(timeEvolutionCoarse, timeEvolutionOld)
        utilitySet.matrixMultiply(timeEvolutionFine, timeEvolutionOld, timeEvolutionCoarse)

    @cuda.jit("(float64[:], float64, complex128)", device = True, inline = True)
    def transformFrameSpinOneRotating(sourceSample, rotatingWave, rotatingWaveWinding):
        X = (sourceSample[0] + 1j*sourceSample[1])/rotatingWaveWinding
        sourceSample[0] = X.real
        sourceSample[1] = X.imag
        sourceSample[2] = sourceSample[2] - rotatingWave

    @cuda.jit("(float64[:], float64, complex128)", device = True, inline = True)
    def transformFrameSpinHalfRotating(sourceSample, rotatingWave, rotatingWaveWinding):
        X = (sourceSample[0] + 1j*sourceSample[1])/(rotatingWaveWinding**2)
        sourceSample[0] = X.real
        sourceSample[1] = X.imag
        sourceSample[2] = sourceSample[2] - 2*rotatingWave

    @cuda.jit("(float64[:], float64, complex128)", device = True, inline = True)
    def transformFrameLab(sourceSample, rotatingWave, rotatingWaveWinding):
        return

    if useRotatingFrame:
        if dimension == 3:
            transformFrame = transformFrameSpinOneRotating
        else:
            transformFrame = transformFrameSpinHalfRotating
    else:
        transformFrame = transformFrameLab

    @cuda.jit(device = True, inline = True)
    def getSource(timeSample, source, timeStepSource, sourceSample):
        utilities.interpolateSourceCubic(source, timeSample, timeStepSource, sourceSample)

    @cuda.jit(device = True, inline = True)
    def getSourceIntegrationMagnusCF4(timeFine, timeCoarse, timeStepFine, sourceSample, rotatingWave, rotatingWaveWinding, source, timeStepSource):
        timeSample = ((timeFine + 0.5*timeStepFine*(1 - 1/sqrt3)) - timeCoarse)
        rotatingWaveWinding[0] = math.cos(math.tau*rotatingWave*timeSample) + 1j*math.sin(math.tau*rotatingWave*timeSample)
        timeSample += timeCoarse
        getSource(timeSample, source, timeStepSource, sourceSample[0, :])

        timeSample = ((timeFine + 0.5*timeStepFine*(1 + 1/sqrt3)) - timeCoarse)
        rotatingWaveWinding[1] = math.cos(math.tau*rotatingWave*timeSample) + 1j*math.sin(math.tau*rotatingWave*timeSample)
        timeSample += timeCoarse
        getSource(timeSample, source, timeStepSource, sourceSample[1, :])

    @cuda.jit(device = True, inline = True)
    def getSourceIntegrationHalfStep(timeFine, timeCoarse, timeStepFine, sourceSample, rotatingWave, rotatingWaveWinding, source, timeStepSource):
        timeSample = timeFine - timeCoarse
        rotatingWaveWinding[0] = math.cos(math.tau*rotatingWave*timeSample) + 1j*math.sin(math.tau*rotatingWave*timeSample)
        timeSample += timeCoarse
        getSource(timeSample, source, timeStepSource, sourceSample[0, :])

        timeSample = timeFine + timeStepFine - timeCoarse
        rotatingWaveWinding[1] = math.cos(math.tau*rotatingWave*timeSample) + 1j*math.sin(math.tau*rotatingWave*timeSample)
        timeSample += timeCoarse
        getSource(timeSample, source, timeStepSource, sourceSample[1, :])

    @cuda.jit(device = True, inline = True)
    def getSourceIntegrationMidpoint(timeFine, timeCoarse, timeStepFine, sourceSample, rotatingWave, rotatingWaveWinding, source, timeStepSource):
        timeSample = timeFine + 0.5*timeStepFine - timeCoarse
        rotatingWaveWinding[0] = math.cos(math.tau*rotatingWave*timeSample) + 1j*math.sin(math.tau*rotatingWave*timeSample)
        timeSample += timeCoarse
        getSource(timeSample, source, timeStepSource, sourceSample[0, :])

    @cuda.jit("(complex128[:, :], complex128[:, :], float64[:, :], float64, float64, complex128[:])", device = True, inline = True)
    def appendExponentiationIntegrationMagnusCF4(timeEvolutionFine, timeEvolutionCoarse, sourceSample, timeStepFine, rotatingWave, rotatingWaveWinding):
        transformFrame(sourceSample[0, :], rotatingWave, rotatingWaveWinding[0])
        transformFrame(sourceSample[1, :], rotatingWave, rotatingWaveWinding[1])

        w0 = (1.5 + sqrt3)/6
        w1 = (1.5 - sqrt3)/6
        
        sourceSample[2, 0] = math.tau*timeStepFine*(w0*sourceSample[0, 0] + w1*sourceSample[1, 0])
        sourceSample[2, 1] = math.tau*timeStepFine*(w0*sourceSample[0, 1] + w1*sourceSample[1, 1])
        sourceSample[2, 2] = math.tau*timeStepFine*(w0*sourceSample[0, 2] + w1*sourceSample[1, 2])
        if dimension > 2:
            sourceSample[2, 3] = math.tau*timeStepFine*(w0*sourceSample[0, 3] + w1*sourceSample[1, 3])

        appendExponentiation(sourceSample[2, :], timeEvolutionFine, timeEvolutionCoarse)

        sourceSample[2, 0] = math.tau*timeStepFine*(w1*sourceSample[0, 0] + w0*sourceSample[1, 0])
        sourceSample[2, 1] = math.tau*timeStepFine*(w1*sourceSample[0, 1] + w0*sourceSample[1, 1])
        sourceSample[2, 2] = math.tau*timeStepFine*(w1*sourceSample[0, 2] + w0*sourceSample[1, 2])
        if dimension > 2:
            sourceSample[2, 3] = math.tau*timeStepFine*(w1*sourceSample[0, 3] + w0*sourceSample[1, 3])

        appendExponentiation(sourceSample[2, :], timeEvolutionFine, timeEvolutionCoarse)

    @cuda.jit("(complex128[:, :], complex128[:, :], float64[:, :], float64, float64, complex128[:])", device = True, inline = True)
    def appendExponentiationIntegrationHalfStep(timeEvolutionFine, timeEvolutionCoarse, sourceSample, timeStepFine, rotatingWave, rotatingWaveWinding):
        transformFrame(sourceSample[0, :], rotatingWave, rotatingWaveWinding[0])
        transformFrame(sourceSample[1, :], rotatingWave, rotatingWaveWinding[1])
        
        sourceSample[2, 0] = math.tau*timeStepFine*sourceSample[0, 0]/2
        sourceSample[2, 1] = math.tau*timeStepFine*sourceSample[0, 1]/2
        sourceSample[2, 2] = math.tau*timeStepFine*sourceSample[0, 2]/2
        if dimension > 2:
            sourceSample[2, 3] = math.tau*timeStepFine*sourceSample[0, 3]/2

        appendExponentiation(sourceSample[2, :], timeEvolutionFine, timeEvolutionCoarse)

        sourceSample[2, 0] = math.tau*timeStepFine*sourceSample[1, 0]/2
        sourceSample[2, 1] = math.tau*timeStepFine*sourceSample[1, 1]/2
        sourceSample[2, 2] = math.tau*timeStepFine*sourceSample[1, 2]/2
        if dimension > 2:
            sourceSample[2, 3] = math.tau*timeStepFine*sourceSample[1, 3]/2

        appendExponentiation(sourceSample[2, :], timeEvolutionFine, timeEvolutionCoarse)

    @cuda.jit("(complex128[:, :], complex128[:, :], float64[:, :], float64, float64, complex128[:])", device = True, inline = True)
    def appendExponentiationIntegrationMidpoint(timeEvolutionFine, timeEvolutionCoarse, sourceSample, timeStepFine, rotatingWave, rotatingWaveWinding):
        transformFrame(sourceSample[0, :], rotatingWave, rotatingWaveWinding[0])
        
        sourceSample[0, 0] = math.tau*timeStepFine*sourceSample[0, 0]
        sourceSample[0, 1] = math.tau*timeStepFine*sourceSample[0, 1]
        sourceSample[0, 2] = math.tau*timeStepFine*sourceSample[0, 2]
        if dimension > 2:
            sourceSample[0, 3] = math.tau*timeStepFine*sourceSample[0, 3]

        appendExponentiation(sourceSample[0, :], timeEvolutionFine, timeEvolutionCoarse)

    if integrationMethod == IntegrationMethod.MAGNUS_CF_4:
        getSourceIntegration = getSourceIntegrationMagnusCF4
        appendExponentiationIntegration = appendExponentiationIntegrationMagnusCF4
    elif integrationMethod == IntegrationMethod.HALF_STEP:
        getSourceIntegration = getSourceIntegrationHalfStep
        appendExponentiationIntegration = appendExponentiationIntegrationHalfStep
    elif integrationMethod == IntegrationMethod.MIDPOINT_SAMPLE:
        getSourceIntegration = getSourceIntegrationMidpoint
        appendExponentiationIntegration = appendExponentiationIntegrationMidpoint

    @cuda.jit(debug = False,  max_registers = 63)
    def getTimeEvolution(timeCoarse, timeEndPoints, timeStepFine, timeStepCoarse,
        timeStepSource, source,
        timeEvolutionCoarse):
        # Declare variables
        timeEvolutionFine = cuda.local.array((dimension, dimension), dtype = nb.complex128)

        sourceSample = cuda.local.array((sampleIndexMax, lieDimension), dtype = nb.float64)
        rotatingWaveWinding = cuda.local.array(sampleIndexEnd, dtype = nb.complex128)

        # Run calculation for each coarse timestep in parallel
        timeIndex = cuda.threadIdx.x + cuda.blockIdx.x*cuda.blockDim.x
        if timeIndex < timeCoarse.size:
            timeCoarse[timeIndex] = timeEndPoints[0] + timeStepCoarse*timeIndex
            timeFine = timeCoarse[timeIndex]

            # Initialise time evolution operator to 1
            utilitySet.setToOne(timeEvolutionCoarse[timeIndex, :])
            sourceSample[0, 2] = 0
            if useRotatingFrame:
                timeSample = timeCoarse[timeIndex] + timeStepCoarse/2
                getSource(timeSample, source, timeStepSource, sourceSample[0, :])
            rotatingWave = sourceSample[0, 2]

            # For every fine step
            for timeFineIndex in range(math.floor(timeStepCoarse/timeStepFine + 0.5)):
                getSourceIntegration(timeFine, timeCoarse[timeIndex], timeStepFine, sourceSample, rotatingWave, rotatingWaveWinding, source, timeStepSource)
                appendExponentiationIntegration(timeEvolutionFine, timeEvolutionCoarse[timeIndex, :], sourceSample, timeStepFine, rotatingWave, rotatingWaveWinding)

                timeFine += timeStepFine

            if useRotatingFrame:
                # Take out of rotating frame
                rotatingWaveWinding[0] = math.cos(math.tau*rotatingWave*timeStepCoarse) + 1j*math.sin(math.tau*rotatingWave*timeStepCoarse)

                timeEvolutionCoarse[timeIndex, 0, 0] /= rotatingWaveWinding[0]
                timeEvolutionCoarse[timeIndex, 0, 1] /= rotatingWaveWinding[0]
                if dimension > 2:
                    timeEvolutionCoarse[timeIndex, 0, 2] /= rotatingWaveWinding[0]

                    timeEvolutionCoarse[timeIndex, 2, 0] *= rotatingWaveWinding[0]
                    timeEvolutionCoarse[timeIndex, 2, 1] *= rotatingWaveWinding[0]
                    timeEvolutionCoarse[timeIndex, 2, 2] *= rotatingWaveWinding[0]
                else:
                    timeEvolutionCoarse[timeIndex, 1, 0] *= rotatingWaveWinding[0]
                    timeEvolutionCoarse[timeIndex, 1, 1] *= rotatingWaveWinding[0]
    return getTimeEvolution

@cuda.jit(debug = cudaDebug,  max_registers = 63)
def getTimeEvolutionSo(timeCoarse, timeEndPoints, timeStepFine, timeStepCoarse,
    timeStepSource, source,
    timeEvolutionCoarse, trotterCutoff):
    """
    Find the stepwise time evolution opperator.

    =========================== =================================== ====
    Property                    Value                               Code
    =========================== =================================== ====
    Spin                        One                                 So
    Integration method          Fourth order commutator free Magnus Cf4
    Frame                       Rotating                            Rf
    Source interpolation method Cubic (Catmull-Rom)                 Ci
    Matrix exponentiator        Lie Trotter based (Spin 1)          
    =========================== =================================== ====

    Parameters
    ----------
    timeCoarse : :class:`numpy.ndarray` of :class:`numpy.double` (timeIndex)
        A coarse grained list of time samples that the time evolution operator is found for. In units of s. This is an output, so use an empty :class:`numpy.ndarray` with :func:`numpy.empty()`, or declare a :class:`numba.cuda.cudadrv.devicearray.DeviceNDArray` using :func:`numba.cuda.device_array_like()`.
    timeEndPoints : :class:`numpy.ndarray` of :class:`numpy.double` (start time (0) or end time (1))
        The time values for when the experiment is to start and finishes. In units of s.
    timeStepFine : `float`
        The time step used within the integration algorithm. In units of s.
    timeStepCoarse : `float`
        The time difference between each element of `timeCoarse`. In units of s. Determines the sample rate of the outputs `timeCoarse` and `timeEvolutionCoarse`.
    timeStepSource : `float`
        The time difference between each element of `source`. In units of s.
    source : :class:`numpy.ndarray` of :class:`numpy.cdouble` (timeSourceIndex, spacialIndex)
        The strength of the source of the spin system. Fourth component is quadratic shift.
    timeEvolutionCoarse : :class:`numpy.ndarray` of :class:`numpy.cdouble` (timeIndex, braStateIndex, ketStateIndex)
        Time evolution operator (matrix) between the current and next timesteps, for each time sampled. See :math:`U(t)` in :ref:`overviewOfSimulationMethod`. This is an output, so use an empty :class:`numpy.ndarray` with :func:`numpy.empty()`, or declare a :class:`numba.cuda.cudadrv.devicearray.DeviceNDArray` using :func:`numba.cuda.device_array_like()`.
    trotterCutoff : `int`
        The number of squares made by the spin 1 matrix exponentiator.
    """
    # Declare variables
    timeEvolutionFine = cuda.local.array((3, 3), dtype = nb.complex128)
    timeEvolutionOld = cuda.local.array((3, 3), dtype = nb.complex128)

    sourceSample = cuda.local.array((2, 4), dtype = nb.float64)
    weight = cuda.local.array(2, dtype = nb.float64)
    rotatingWaveWinding = cuda.local.array(2, dtype = nb.complex128)

    # Run calculation for each coarse timestep in parallel
    timeIndex = cuda.threadIdx.x + cuda.blockIdx.x*cuda.blockDim.x
    if timeIndex < timeCoarse.size:
        timeCoarse[timeIndex] = timeEndPoints[0] + timeStepCoarse*timeIndex
        timeFine = timeCoarse[timeIndex]

        # Initialise time evolution operator to 1
        utilities.spinOne.setToOne(timeEvolutionCoarse[timeIndex, :])
        timeSample = timeCoarse[timeIndex] + timeStepCoarse/2
        # utilities.interpolateSourceCubic
        interpolate(source, timeSample, timeStepSource, sourceSample[0, :])
        rotatingWave = math.tau*sourceSample[0, 2]

        # For every fine step
        for timeFineIndex in range(math.floor(timeStepCoarse/timeStepFine + 0.5)):
            for timeSampleIndex in range(2):
                # 2nd order quadrature => Sample at +- 1/sqrt(3)
                timeSample = ((timeFine + 0.5*timeStepFine*(1 + (2*timeSampleIndex - 1)/sqrt3)) - timeCoarse[timeIndex])
                rotatingWaveWinding[timeSampleIndex] = math.cos(rotatingWave*timeSample) + 1j*math.sin(rotatingWave*timeSample)

                timeSample = (timeSample + timeCoarse[timeIndex])
                utilities.interpolateSourceCubic(source, timeSample, timeStepSource, sourceSample[timeSampleIndex, :])
                            
            for exponentialIndex in range(-1, 2, 2):
                weight[0] = (1.5 - exponentialIndex*sqrt3)/6
                weight[1] = (1.5 + exponentialIndex*sqrt3)/6

                X = math.tau*timeStepFine*(weight[0]*(sourceSample[0, 0] + 1j*sourceSample[0, 1])/rotatingWaveWinding[0] + weight[1]*(sourceSample[1, 0] + 1j*sourceSample[1, 1])/rotatingWaveWinding[1])
                z = math.tau*timeStepFine*(weight[0]*(sourceSample[0, 2] - rotatingWave/math.tau) + weight[1]*(sourceSample[1, 2] - rotatingWave/math.tau))
                q = math.tau*timeStepFine*(weight[0]*sourceSample[0, 3] + weight[1]*sourceSample[1, 3])

                # Calculate the exponential from the expansion
                utilities.spinOne.matrixExponentialLieTrotter(X.real, X.imag, z, q, timeEvolutionFine, trotterCutoff)

                # Premultiply to the exitsing time evolution operator
                utilities.spinOne.setTo(timeEvolutionCoarse[timeIndex, :], timeEvolutionOld)
                utilities.spinOne.matrixMultiply(timeEvolutionFine, timeEvolutionOld, timeEvolutionCoarse[timeIndex, :])

            timeFine += timeStepFine

        # Take out of rotating frame
        rotatingWaveWinding[0] = math.cos(rotatingWave*timeStepCoarse) + 1j*math.sin(rotatingWave*timeStepCoarse)

        timeEvolutionCoarse[timeIndex, 0, 0] /= rotatingWaveWinding[0]
        timeEvolutionCoarse[timeIndex, 0, 1] /= rotatingWaveWinding[0]
        timeEvolutionCoarse[timeIndex, 0, 2] /= rotatingWaveWinding[0]

        timeEvolutionCoarse[timeIndex, 2, 0] *= rotatingWaveWinding[0]
        timeEvolutionCoarse[timeIndex, 2, 1] *= rotatingWaveWinding[0]
        timeEvolutionCoarse[timeIndex, 2, 2] *= rotatingWaveWinding[0]

@cuda.jit(debug = cudaDebug,  max_registers = 63)
def getTimeEvolutionSh(timeCoarse, timeEndPoints, timeStepFine, timeStepCoarse,
    timeStepSource, source,
    timeEvolutionCoarse, trotterCutoff):
    """
    Find the stepwise time evolution opperator.

    =========================== =================================== ====
    Property                    Value                               Code
    =========================== =================================== ====
    Spin                        Half                                So
    Integration method          Fourth order commutator free Magnus Cf4
    Frame                       Rotating                            Rf
    Source interpolation method Cubic (Catmull-Rom)                 Ci
    Matrix exponentiator        Analytic (spin half)
    =========================== =================================== ====

    Parameters
    ----------
    timeCoarse : :class:`numpy.ndarray` of :class:`numpy.double` (timeIndex)
        A coarse grained list of time samples that the time evolution operator is found for. In units of s. This is an output, so use an empty :class:`numpy.ndarray` with :func:`numpy.empty()`, or declare a :class:`numba.cuda.cudadrv.devicearray.DeviceNDArray` using :func:`numba.cuda.device_array_like()`.
    timeEndPoints : :class:`numpy.ndarray` of :class:`numpy.double` (start time (0) or end time (1))
        The time values for when the experiment is to start and finishes. In units of s.
    timeStepFine : `float`
        The time step used within the integration algorithm. In units of s.
    timeStepCoarse : `float`
        The time difference between each element of `timeCoarse`. In units of s. Determines the sample rate of the outputs `timeCoarse` and `timeEvolutionCoarse`.
    timeStepSource : `float`
        The time difference between each element of `source`. In units of s.
    source : :class:`numpy.ndarray` of :class:`numpy.cdouble` (timeSourceIndex, spacialIndex)
        The strength of the source of the spin system. Fourth component is quadratic shift.
    timeEvolutionCoarse : :class:`numpy.ndarray` of :class:`numpy.cdouble` (timeIndex, braStateIndex, ketStateIndex)
        Time evolution operator (matrix) between the current and next timesteps, for each time sampled. See :math:`U(t)` in :ref:`overviewOfSimulationMethod`. This is an output, so use an empty :class:`numpy.ndarray` with :func:`numpy.empty()`, or declare a :class:`numba.cuda.cudadrv.devicearray.DeviceNDArray` using :func:`numba.cuda.device_array_like()`.
    trotterCutoff : `int`
        The number of squares made by the spin 1 matrix exponentiator.
    """

    # Declare variables
    timeEvolutionFine = cuda.local.array((2, 2), dtype = nb.complex128)
    timeEvolutionOld = cuda.local.array((2, 2), dtype = nb.complex128)

    sourceSample = cuda.local.array((2, 3), dtype = nb.float64)
    weight = cuda.local.array(2, dtype = nb.float64)
    rotatingWaveWinding = cuda.local.array(2, dtype = nb.complex128)

    # Run calculation for each coarse timestep in parallel
    timeIndex = cuda.threadIdx.x + cuda.blockIdx.x*cuda.blockDim.x
    if timeIndex < timeCoarse.size:
        timeCoarse[timeIndex] = timeEndPoints[0] + timeStepCoarse*timeIndex
        timeFine = timeCoarse[timeIndex]

        # Initialise time evolution operator to 1
        utilities.spinHalf.setToOne(timeEvolutionCoarse[timeIndex, :])
        timeSample = timeCoarse[timeIndex] + timeStepCoarse/2
        utilities.interpolateSourceCubic(source, timeSample, timeStepSource, sourceSample[0, :])
        rotatingWave = math.tau*sourceSample[0, 2]/2

        # For every fine step
        for timeFineIndex in range(math.floor(timeStepCoarse/timeStepFine + 0.5)):
            for timeSampleIndex in range(2):
                # 2nd order quadrature => Sample at +- 1/sqrt(3)
                timeSample = ((timeFine + 0.5*timeStepFine*(1 + (2*timeSampleIndex - 1)/sqrt3)) - timeCoarse[timeIndex])
                rotatingWaveWinding[timeSampleIndex] = math.cos(rotatingWave*timeSample) + 1j*math.sin(rotatingWave*timeSample)

                timeSample = (timeSample + timeCoarse[timeIndex])
                utilities.interpolateSourceCubic(source, timeSample, timeStepSource, sourceSample[timeSampleIndex, :])
                            
            for exponentialIndex in range(-1, 2, 2):
                weight[0] = (1.5 - exponentialIndex*sqrt3)/6
                weight[1] = (1.5 + exponentialIndex*sqrt3)/6

                X = math.tau*timeStepFine*(weight[0]*(sourceSample[0, 0] + 1j*sourceSample[0, 1])/(rotatingWaveWinding[0]**2) + weight[1]*(sourceSample[1, 0] + 1j*sourceSample[1, 1])/(rotatingWaveWinding[1]**2))
                z = math.tau*timeStepFine*(weight[0]*(sourceSample[0, 2] - 2*rotatingWave/math.tau) + weight[1]*(sourceSample[1, 2] - 2*rotatingWave/math.tau))

                # Calculate the exponential from the expansion
                utilities.spinHalf.matrixExponentialAnalytic(X.real, X.imag, z, timeEvolutionFine)

                # Premultiply to the exitsing time evolution operator
                utilities.spinHalf.setTo(timeEvolutionCoarse[timeIndex, :], timeEvolutionOld)
                utilities.spinHalf.matrixMultiply(timeEvolutionFine, timeEvolutionOld, timeEvolutionCoarse[timeIndex, :])

            timeFine += timeStepFine

        # Take out of rotating frame
        rotatingWaveWinding[0] = math.cos(rotatingWave*timeStepCoarse) + 1j*math.sin(rotatingWave*timeStepCoarse)

        timeEvolutionCoarse[timeIndex, 0, 0] /= rotatingWaveWinding[0]
        timeEvolutionCoarse[timeIndex, 0, 1] /= rotatingWaveWinding[0]

        timeEvolutionCoarse[timeIndex, 1, 0] *= rotatingWaveWinding[0]
        timeEvolutionCoarse[timeIndex, 1, 1] *= rotatingWaveWinding[0]

@cuda.jit(debug = cudaDebug,  max_registers = 63)
def getTimeEvolutionSoCf4RfLi(timeCoarse, timeEndPoints, timeStepFine, timeStepCoarse,
    timeStepSource, source,
    timeEvolutionCoarse, trotterCutoff):
    """
    Find the stepwise time evolution opperator.

    =========================== =================================== ====
    Property                    Value                               Code
    =========================== =================================== ====
    Spin                        One                                 So
    Integration method          Fourth order commutator free Magnus Cf4
    Frame                       Rotating                            Rf
    Source interpolation method Linear                              Li
    Matrix exponentiator        Lie Trotter based (Spin 1)          
    =========================== =================================== ====

    Parameters
    ----------
    timeCoarse : :class:`numpy.ndarray` of :class:`numpy.double` (timeIndex)
        A coarse grained list of time samples that the time evolution operator is found for. In units of s. This is an output, so use an empty :class:`numpy.ndarray` with :func:`numpy.empty()`, or declare a :class:`numba.cuda.cudadrv.devicearray.DeviceNDArray` using :func:`numba.cuda.device_array_like()`.
    timeEndPoints : :class:`numpy.ndarray` of :class:`numpy.double` (start time (0) or end time (1))
        The time values for when the experiment is to start and finishes. In units of s.
    timeStepFine : `float`
        The time step used within the integration algorithm. In units of s.
    timeStepCoarse : `float`
        The time difference between each element of `timeCoarse`. In units of s. Determines the sample rate of the outputs `timeCoarse` and `timeEvolutionCoarse`.
    timeStepSource : `float`
        The time difference between each element of `source`. In units of s.
    source : :class:`numpy.ndarray` of :class:`numpy.cdouble` (timeSourceIndex, spacialIndex)
        The strength of the source of the spin system. Fourth component is quadratic shift.
    timeEvolutionCoarse : :class:`numpy.ndarray` of :class:`numpy.cdouble` (timeIndex, braStateIndex, ketStateIndex)
        Time evolution operator (matrix) between the current and next timesteps, for each time sampled. See :math:`U(t)` in :ref:`overviewOfSimulationMethod`. This is an output, so use an empty :class:`numpy.ndarray` with :func:`numpy.empty()`, or declare a :class:`numba.cuda.cudadrv.devicearray.DeviceNDArray` using :func:`numba.cuda.device_array_like()`.
    trotterCutoff : `int`
        The number of squares made by the spin 1 matrix exponentiator.
    """

    # Declare variables
    timeEvolutionFine = cuda.local.array((3, 3), dtype = nb.complex128)
    timeEvolutionOld = cuda.local.array((3, 3), dtype = nb.complex128)

    sourceSample = cuda.local.array((2, 4), dtype = nb.float64)
    weight = cuda.local.array(2, dtype = nb.float64)
    rotatingWaveWinding = cuda.local.array(2, dtype = nb.complex128)

    # Run calculation for each coarse timestep in parallel
    timeIndex = cuda.threadIdx.x + cuda.blockIdx.x*cuda.blockDim.x
    if timeIndex < timeCoarse.size:
        timeCoarse[timeIndex] = timeEndPoints[0] + timeStepCoarse*timeIndex
        timeFine = timeCoarse[timeIndex]

        # Initialise time evolution operator to 1
        utilities.spinOne.setToOne(timeEvolutionCoarse[timeIndex, :])
        timeSample = timeCoarse[timeIndex] + timeStepCoarse/2
        utilities.interpolateSourceLinear(source, timeSample, timeStepSource, sourceSample[0, :])
        rotatingWave = math.tau*sourceSample[0, 2]

        # For every fine step
        for timeFineIndex in range(math.floor(timeStepCoarse/timeStepFine + 0.5)):
            for timeSampleIndex in range(2):
                # 2nd order quadrature => Sample at +- 1/sqrt(3)
                timeSample = ((timeFine + 0.5*timeStepFine*(1 + (2*timeSampleIndex - 1)/sqrt3)) - timeCoarse[timeIndex])
                rotatingWaveWinding[timeSampleIndex] = math.cos(rotatingWave*timeSample) + 1j*math.sin(rotatingWave*timeSample)

                timeSample = (timeSample + timeCoarse[timeIndex])
                utilities.interpolateSourceLinear(source, timeSample, timeStepSource, sourceSample[timeSampleIndex, :])
                            
            for exponentialIndex in range(-1, 2, 2):
                weight[0] = (1.5 - exponentialIndex*sqrt3)/6
                weight[1] = (1.5 + exponentialIndex*sqrt3)/6

                X = math.tau*timeStepFine*(weight[0]*(sourceSample[0, 0] + 1j*sourceSample[0, 1])/rotatingWaveWinding[0] + weight[1]*(sourceSample[1, 0] + 1j*sourceSample[1, 1])/rotatingWaveWinding[1])
                z = math.tau*timeStepFine*(weight[0]*(sourceSample[0, 2] - rotatingWave/math.tau) + weight[1]*(sourceSample[1, 2] - rotatingWave/math.tau))
                q = math.tau*timeStepFine*(weight[0]*sourceSample[0, 3] + weight[1]*sourceSample[1, 3])

                # Calculate the exponential from the expansion
                utilities.spinOne.matrixExponentialLieTrotter(X.real, X.imag, z, q, timeEvolutionFine, trotterCutoff)

                # Premultiply to the exitsing time evolution operator
                utilities.spinOne.setTo(timeEvolutionCoarse[timeIndex, :], timeEvolutionOld)
                utilities.spinOne.matrixMultiply(timeEvolutionFine, timeEvolutionOld, timeEvolutionCoarse[timeIndex, :])

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
def getTimeEvolutionSoCf4LfCi(timeCoarse, timeEndPoints, timeStepFine, timeStepCoarse,
    timeStepSource, source,
    timeEvolutionCoarse, trotterCutoff):
    """
    Find the stepwise time evolution opperator.

    =========================== =================================== ====
    Property                    Value                               Code
    =========================== =================================== ====
    Spin                        One                                 So
    Integration method          Fourth order commutator free Magnus Cf4
    Frame                       Lab                                 Lf
    Source interpolation method Cubic (Catmull-Rom)                 Ci
    Matrix exponentiator        Lie Trotter based (Spin 1)          
    =========================== =================================== ====
    
    Parameters
    ----------
    timeCoarse : :class:`numpy.ndarray` of :class:`numpy.double` (timeIndex)
        A coarse grained list of time samples that the time evolution operator is found for. In units of s. This is an output, so use an empty :class:`numpy.ndarray` with :func:`numpy.empty()`, or declare a :class:`numba.cuda.cudadrv.devicearray.DeviceNDArray` using :func:`numba.cuda.device_array_like()`.
    timeEndPoints : :class:`numpy.ndarray` of :class:`numpy.double` (start time (0) or end time (1))
        The time values for when the experiment is to start and finishes. In units of s.
    timeStepFine : `float`
        The time step used within the integration algorithm. In units of s.
    timeStepCoarse : `float`
        The time difference between each element of `timeCoarse`. In units of s. Determines the sample rate of the outputs `timeCoarse` and `timeEvolutionCoarse`.
    timeStepSource : `float`
        The time difference between each element of `source`. In units of s.
    source : :class:`numpy.ndarray` of :class:`numpy.cdouble` (timeSourceIndex, spacialIndex)
        The strength of the source of the spin system. Fourth component is quadratic shift.
    timeEvolutionCoarse : :class:`numpy.ndarray` of :class:`numpy.cdouble` (timeIndex, braStateIndex, ketStateIndex)
        Time evolution operator (matrix) between the current and next timesteps, for each time sampled. See :math:`U(t)` in :ref:`overviewOfSimulationMethod`. This is an output, so use an empty :class:`numpy.ndarray` with :func:`numpy.empty()`, or declare a :class:`numba.cuda.cudadrv.devicearray.DeviceNDArray` using :func:`numba.cuda.device_array_like()`.
    trotterCutoff : `int`
        The number of squares made by the spin 1 matrix exponentiator.
    """

    # Declare variables
    timeEvolutionFine = cuda.local.array((3, 3), dtype = nb.complex128)
    timeEvolutionOld = cuda.local.array((3, 3), dtype = nb.complex128)
    sourceSample = cuda.local.array((2, 4), dtype = nb.float64)
    weight = cuda.local.array(2, dtype = nb.float64)

    # Run calculation for each coarse timestep in parallel
    timeIndex = cuda.threadIdx.x + cuda.blockIdx.x*cuda.blockDim.x
    if timeIndex < timeCoarse.size:
        timeCoarse[timeIndex] = timeEndPoints[0] + timeStepCoarse*timeIndex
        timeFine = timeCoarse[timeIndex]

        # Initialise time evolution operator to 1
        utilities.spinOne.setToOne(timeEvolutionCoarse[timeIndex, :])

        # For every fine step
        for timeFineIndex in range(math.floor(timeStepCoarse/timeStepFine + 0.5)):
            for timeSampleIndex in range(2):
                # 2nd order quadrature => Sample at +- 1/sqrt(3)
                timeSample = ((timeFine + 0.5*timeStepFine*(1 + (2*timeSampleIndex - 1)/sqrt3)))
                utilities.interpolateSourceCubic(source, timeSample, timeStepSource, sourceSample[timeSampleIndex, :])
                            
            for exponentialIndex in range(-1, 2, 2):
                weight[0] = (1.5 - exponentialIndex*sqrt3)/6
                weight[1] = (1.5 + exponentialIndex*sqrt3)/6

                X = math.tau*timeStepFine*(weight[0]*(sourceSample[0, 0] + 1j*sourceSample[0, 1]) + weight[1]*(sourceSample[1, 0] + 1j*sourceSample[1, 1]))
                z = math.tau*timeStepFine*(weight[0]*(sourceSample[0, 2]) + weight[1]*(sourceSample[1, 2]))
                q = math.tau*timeStepFine*(weight[0]*sourceSample[0, 3] + weight[1]*sourceSample[1, 3])

                # Calculate the exponential from the expansion
                utilities.spinOne.matrixExponentialLieTrotter(X.real, X.imag, z, q, timeEvolutionFine, trotterCutoff)

                # Premultiply to the exitsing time evolution operator
                utilities.spinOne.setTo(timeEvolutionCoarse[timeIndex, :], timeEvolutionOld)
                utilities.spinOne.matrixMultiply(timeEvolutionFine, timeEvolutionOld, timeEvolutionCoarse[timeIndex, :])

            timeFine += timeStepFine

@cuda.jit(debug = cudaDebug,  max_registers = 95)
def getTimeEvolutionSoHaLfCi(timeCoarse, timeEndPoints, timeStepFine, timeStepCoarse,
    timeStepSource, source,
    timeEvolutionCoarse, trotterCutoff):
    """
    Find the stepwise time evolution opperator.

    =========================== =================================== ====
    Property                    Value                               Code
    =========================== =================================== ====
    Spin                        One                                 So
    Integration method          Halfstep sample at edge of interval Ha
    Frame                       Lab                                 Lf
    Source interpolation method Cubic (Catmull-Rom)                 Ci
    Matrix exponentiator        Lie Trotter based (Spin 1)          
    =========================== =================================== ====

    The same sampling as used in the previous cython code.

    Parameters
    ----------
    timeCoarse : :class:`numpy.ndarray` of :class:`numpy.double` (timeIndex)
        A coarse grained list of time samples that the time evolution operator is found for. In units of s. This is an output, so use an empty :class:`numpy.ndarray` with :func:`numpy.empty()`, or declare a :class:`numba.cuda.cudadrv.devicearray.DeviceNDArray` using :func:`numba.cuda.device_array_like()`.
    timeEndPoints : :class:`numpy.ndarray` of :class:`numpy.double` (start time (0) or end time (1))
        The time values for when the experiment is to start and finishes. In units of s.
    timeStepFine : `float`
        The time step used within the integration algorithm. In units of s.
    timeStepCoarse : `float`
        The time difference between each element of `timeCoarse`. In units of s. Determines the sample rate of the outputs `timeCoarse` and `timeEvolutionCoarse`.
    timeStepSource : `float`
        The time difference between each element of `source`. In units of s.
    source : :class:`numpy.ndarray` of :class:`numpy.cdouble` (timeSourceIndex, spacialIndex)
        The strength of the source of the spin system. Fourth component is quadratic shift.
    timeEvolutionCoarse : :class:`numpy.ndarray` of :class:`numpy.cdouble` (timeIndex, braStateIndex, ketStateIndex)
        Time evolution operator (matrix) between the current and next timesteps, for each time sampled. See :math:`U(t)` in :ref:`overviewOfSimulationMethod`. This is an output, so use an empty :class:`numpy.ndarray` with :func:`numpy.empty()`, or declare a :class:`numba.cuda.cudadrv.devicearray.DeviceNDArray` using :func:`numba.cuda.device_array_like()`.
    trotterCutoff : `int`
        The number of squares made by the spin 1 matrix exponentiator.
    """

    # Declare variables
    timeEvolutionFine = cuda.local.array((3, 3), dtype = nb.complex128)
    timeEvolutionOld = cuda.local.array((3, 3), dtype = nb.complex128)
    sourceSample = cuda.local.array(4, dtype = nb.float64)

    # Run calculation for each coarse timestep in parallel
    timeIndex = cuda.threadIdx.x + cuda.blockIdx.x*cuda.blockDim.x
    if timeIndex < timeCoarse.size:
        timeCoarse[timeIndex] = timeEndPoints[0] + timeStepCoarse*timeIndex
        timeFine = timeCoarse[timeIndex]

        # Initialise time evolution operator to 1
        utilities.spinOne.setToOne(timeEvolutionCoarse[timeIndex, :])

        # For every fine step
        for timeFineIndex in range(math.floor(timeStepCoarse/timeStepFine + 0.5)):
            for timeSampleIndex in range(2):
                timeSample = timeFine + timeSampleIndex*timeStepFine
                utilities.interpolateSourceCubic(source, timeSample, timeStepSource, sourceSample)

                X = 0.5*math.tau*timeStepFine*(sourceSample[0] + 1j*sourceSample[1])
                z = 0.5*math.tau*timeStepFine*sourceSample[2]
                q = 0.5*math.tau*timeStepFine*sourceSample[3]

                # Calculate the exponential from the expansion
                utilities.spinOne.matrixExponentialLieTrotter(X.real, X.imag, z, q, timeEvolutionFine, trotterCutoff)

                # Premultiply to the exitsing time evolution operator
                utilities.spinOne.setTo(timeEvolutionCoarse[timeIndex, :], timeEvolutionOld)
                utilities.spinOne.matrixMultiply(timeEvolutionFine, timeEvolutionOld, timeEvolutionCoarse[timeIndex, :])

            timeFine += timeStepFine

@cuda.jit(debug = cudaDebug,  max_registers = 95)
def getTimeEvolutionSoMsLfCi(timeCoarse, timeEndPoints, timeStepFine, timeStepCoarse,
    timeStepSource, source,
    timeEvolutionCoarse, trotterCutoff):
    """
    Find the stepwise time evolution opperator.

    =========================== =================================== ====
    Property                    Value                               Code
    =========================== =================================== ====
    Spin                        One                                 So
    Integration method          Single midpoint sample              Ms
    Frame                       Lab                                 Lf
    Source interpolation method Cubic (Catmull-Rom)                 Ci
    Matrix exponentiator        Lie Trotter based (Spin 1)          
    =========================== =================================== ====

    Parameters
    ----------
    timeCoarse : :class:`numpy.ndarray` of :class:`numpy.double` (timeIndex)
        A coarse grained list of time samples that the time evolution operator is found for. In units of s. This is an output, so use an empty :class:`numpy.ndarray` with :func:`numpy.empty()`, or declare a :class:`numba.cuda.cudadrv.devicearray.DeviceNDArray` using :func:`numba.cuda.device_array_like()`.
    timeEndPoints : :class:`numpy.ndarray` of :class:`numpy.double` (start time (0) or end time (1))
        The time values for when the experiment is to start and finishes. In units of s.
    timeStepFine : `float`
        The time step used within the integration algorithm. In units of s.
    timeStepCoarse : `float`
        The time difference between each element of `timeCoarse`. In units of s. Determines the sample rate of the outputs `timeCoarse` and `timeEvolutionCoarse`.
    timeStepSource : `float`
        The time difference between each element of `source`. In units of s.
    source : :class:`numpy.ndarray` of :class:`numpy.cdouble` (timeSourceIndex, spacialIndex)
        The strength of the source of the spin system. Fourth component is quadratic shift.
    timeEvolutionCoarse : :class:`numpy.ndarray` of :class:`numpy.cdouble` (timeIndex, braStateIndex, ketStateIndex)
        Time evolution operator (matrix) between the current and next timesteps, for each time sampled. See :math:`U(t)` in :ref:`overviewOfSimulationMethod`. This is an output, so use an empty :class:`numpy.ndarray` with :func:`numpy.empty()`, or declare a :class:`numba.cuda.cudadrv.devicearray.DeviceNDArray` using :func:`numba.cuda.device_array_like()`.
    trotterCutoff : `int`
        The number of squares made by the spin 1 matrix exponentiator.
    """

    # Declare variables
    timeEvolutionFine = cuda.local.array((3, 3), dtype = nb.complex128)
    timeEvolutionOld = cuda.local.array((3, 3), dtype = nb.complex128)
    sourceSample = cuda.local.array(3, dtype = nb.float64)

    # Run calculation for each coarse timestep in parallel
    timeIndex = cuda.threadIdx.x + cuda.blockIdx.x*cuda.blockDim.x
    if timeIndex < timeCoarse.size:
        timeCoarse[timeIndex] = timeEndPoints[0] + timeStepCoarse*timeIndex
        timeFine = timeCoarse[timeIndex]

        # Initialise time evolution operator to 1
        utilities.spinOne.setToOne(timeEvolutionCoarse[timeIndex, :])

        # For every fine step
        for timeFineIndex in range(math.floor(timeStepCoarse/timeStepFine + 0.5)):
            timeSample = timeFine
            utilities.interpolateSourceCubic(source, timeSample, timeStepSource, sourceSample)

            X = math.tau*timeStepFine*(sourceSample[0] + 1j*sourceSample[1])
            z = math.tau*timeStepFine*sourceSample[2]
            q = math.tau*timeStepFine*sourceSample[3]

            # Calculate the exponential from the expansion
            utilities.spinOne.matrixExponentialLieTrotter(X.real, X.imag, z, q, timeEvolutionFine, trotterCutoff)

            # Premultiply to the exitsing time evolution operator
            utilities.spinOne.setTo(timeEvolutionCoarse[timeIndex, :], timeEvolutionOld)
            utilities.spinOne.matrixMultiply(timeEvolutionFine, timeEvolutionOld, timeEvolutionCoarse[timeIndex, :])

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
        for xIndex in nb.prange(state.shape[1]):
            state[timeIndex, xIndex] = 0
            if timeIndex > 0:
                for zIndex in range(state.shape[1]):
                    state[timeIndex, xIndex] += timeEvolution[timeIndex - 1, xIndex, zIndex]*state[timeIndex - 1, zIndex]
            else:
                state[timeIndex, xIndex] += stateInit[xIndex]

@cuda.jit(debug = cudaDebug)
def getSpin(state, spin):
    """
    Calculate each expected spin value in parallel.

    For spin half:

    .. math::
        \\begin{align*}
            \\langle F\\rangle(t) = \\begin{pmatrix}
                \\Re(\\psi_{+\\frac{1}{2}}(t)\\psi_{-\\frac{1}{2}}(t)^*)\\\\
                -\\Im(\\psi_{+\\frac{1}{2}}(t)\\psi_{-\\frac{1}{2}}(t)^*)\\\\
                \\frac{1}{2}(|\\psi_{+\\frac{1}{2}}(t)|^2 - |\\psi_{-\\frac{1}{2}}(t)|^2)
            \\end{pmatrix}
        \\end{align*}

    For spin one:

    .. math::
        \\begin{align*}
            \\langle F\\rangle(t) = \\begin{pmatrix}
                \\Re(\\sqrt{2}\\psi_{0}(t)^*(\\psi_{+1}(t) + \\psi_{-1}(t))\\\\
                -\\Im(\\sqrt{2}\\psi_{0}(t)^*(\\psi_{+1}(t) - \\psi_{-1}(t))\\\\
                |\\psi_{+1}(t)|^2 - |\\psi_{-1}(t)|^2
            \\end{pmatrix}
        \\end{align*}

    Parameters
    ----------
    state : :class:`numpy.ndarray` of :class:`numpy.cdouble` (timeIndex, stateIndex)
        The state (wavefunction) of the spin system in the lab frame, for each time sampled. See :math:`\\psi(t)` in :ref:`overviewOfSimulationMethod`.
    spin : :class:`numpy.ndarray` of :class:`numpy.double` (timeIndex, spatialIndex)
        The expected value for hyperfine spin of the spin system in the lab frame, for each time sampled. Units of :math:`\\hbar`. This is an output, so use an empty :class:`numpy.ndarray` with :func:`numpy.empty()`, or declare a :class:`numba.cuda.cudadrv.devicearray.DeviceNDArray` using :func:`numba.cuda.device_array_like()`.
    """
    timeIndex = cuda.threadIdx.x + cuda.blockIdx.x*cuda.blockDim.x
    if timeIndex < spin.shape[0]:
        if state.shape[1] == 2:
            spin[timeIndex, 0] = (state[timeIndex, 0]*utilities.scalar.conj(state[timeIndex, 1])).real
            spin[timeIndex, 1] = (1j*state[timeIndex, 0]*utilities.scalar.conj(state[timeIndex, 1])).real
            spin[timeIndex, 2] = 0.5*(state[timeIndex, 0].real**2 + state[timeIndex, 0].imag**2 - state[timeIndex, 1].real**2 - state[timeIndex, 1].imag**2)
        else:
            spin[timeIndex, 0] = (2*utilities.scalar.conj(state[timeIndex, 1])*(state[timeIndex, 0] + state[timeIndex, 2])/sqrt2).real
            spin[timeIndex, 1] = (2j*utilities.scalar.conj(state[timeIndex, 1])*(state[timeIndex, 0] - state[timeIndex, 2])/sqrt2).real
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

@cuda.jit(debug = cudaDebug)
def evaluateDressing(timeStepSource, sourceIndexMax, sourceAmplitude, sourceFrequency, sourcePhase, sourceTimeEndPoints, sourceQuadraticShift, source):
    timeSourceIndex = cuda.threadIdx.x + cuda.blockIdx.x*cuda.blockDim.x
    if timeSourceIndex < source.shape[0]:
        timeSource = timeSourceIndex*timeStepSource
        for sourceIndex in range(sourceIndexMax):
            for spacialIndex in range(source.shape[1]):
                source[timeSourceIndex, spacialIndex] += \
                    (timeSource >= sourceTimeEndPoints[sourceIndex, 0])*\
                    (timeSource <= sourceTimeEndPoints[sourceIndex, 1])*\
                    sourceAmplitude[sourceIndex, spacialIndex]*\
                    math.sin(\
                        math.tau*sourceFrequency[sourceIndex, spacialIndex]*\
                        (timeSource - sourceTimeEndPoints[sourceIndex, 0]) +\
                        sourcePhase[sourceIndex, spacialIndex]\
                    )
            source[timeSourceIndex, 3] = sourceQuadraticShift