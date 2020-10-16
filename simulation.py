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

"""
=== Overview of the simulation method ===

The goal here is to evaluate the value of the spin, and thus
the quantum state of a 3 level atom in a coarse grained time
series with step Dt. The time evolution between time indices is

y(t + Dt) = U(t -> t + Dt) y(t)
y(t + Dt) = U(t) y(t)

Each U(t) is completely independent of y(t0) or U(t0) for any
other time value t0. Therefore each U(t) can be calculated
independently of each other. This is done in parallel using a GPU
kernel. Afterwards, the final result of

y(t + Dt) = U(t) y(t)

is calculated sequentially for each t. Afterwards, the spin at
each timestep is calculated in parallel.

(So far) all magnetic signals fed into the integrator in the form
of sine waves, with varying amplitude, frequency, phase, and
start and end times. This can be used to simulate anything from
the bias and dressing fields, to the fake neural pulses, to
AC line and DC detuning noise. These sinusoids are superposed
and sampled at any time step needed to for the solver.
"""

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
    """
    def __init__(self, signal, dressingRabiFrequency = 1000, quadraticShift = 0.0):
        self.dressingRabiFrequency = dressingRabiFrequency
        self.sourceIndexMax = 0                             # The number of sources
        self.sourceAmplitude = np.empty([0, 3])             # Sine wave amplitude (Hz) [source index, xyz direction]
        self.sourcePhase = np.empty([0, 3])                 # Sine wave phase offset, starts from 0 (rad) [source index, xyz direction]
        self.sourceFrequency = np.empty([0, 3])             # Sine wave frequency, separate for each direction (Hz) [source index, xyz direction]
        self.sourceTimeEndPoints = np.empty([0, 2])         # Sine wave start and end times (s) [source index, start time / end time]
        self.sourceType = np.empty([0], dtype = object)     # Description of what the source physically represents, for archive (string) [source index]
        self.sourceQuadraticShift = quadraticShift          # 3 state quadratic shift (Hz)

        # Construct the signal from the dressing information and pulse description.
        if signal:
            self.addDressing(signal, dressingRabiFrequency)
            for neuralPulse in signal.neuralPulses:
                self.addNeuralPulse(neuralPulse.timeStart, neuralPulse.amplitude, neuralPulse.frequency)
        
    def writeToFile(self, archive):
        """
        Saves source information to archive file
        """
        archiveGroup = archive.require_group("sourceProperties")
        archiveGroup["sourceAmplitude"] = self.sourceAmplitude
        archiveGroup["sourcePhase"] = self.sourcePhase
        archiveGroup["sourceFrequency"] = self.sourceFrequency
        archiveGroup["sourceTimeEndPoints"] = self.sourceTimeEndPoints
        archiveGroup["sourceType"] = np.asarray(self.sourceType, dtype='|S32')
        archiveGroup["sourceIndexMax"] = np.asarray([self.sourceIndexMax])
        archiveGroup["sourceQuadraticShift"] = np.asarray([self.sourceQuadraticShift])

    def addDressing(self, signal, dressingRabiFrequency = 1e3, dressingFrequency = 700e3, biasAmplitude = 700e3):
        """
        Adds bias and dressing to list of sources
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
        sourceAmplitude[0, 0] = 2*dressingRabiFrequency
        sourceFrequency[0, 0] = dressingFrequency
        sourceTimeEndPoints[0, :] = 1*signal.timeProperties.timeEndPoints
        sourcePhase[0, 0] = math.pi/2

        # Add
        self.sourceAmplitude = np.concatenate((self.sourceAmplitude, sourceAmplitude))
        self.sourcePhase = np.concatenate((self.sourcePhase, sourcePhase))
        self.sourceFrequency = np.concatenate((self.sourceFrequency, sourceFrequency))
        self.sourceTimeEndPoints = np.concatenate((self.sourceTimeEndPoints, sourceTimeEndPoints))
        self.sourceType = np.concatenate((self.sourceType, sourceType))
        self.sourceIndexMax += 1

    def addNeuralPulse(self, neuralPulseTimeStart, neuralPulseAmplitude = 100.0, neuralPulseFrequency = 1e3):
        """
        Adds a neural pulse signal to the list of sources
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
        sourceAmplitude[0, 2] = neuralPulseAmplitude
        sourceFrequency[0, 2] = neuralPulseFrequency
        # sourcePhase[0, 2] = math.pi/2
        sourceTimeEndPoints[0, :] = np.asarray([neuralPulseTimeStart, neuralPulseTimeStart + 1/neuralPulseFrequency])

        # Add
        self.sourceAmplitude = np.concatenate((self.sourceAmplitude, sourceAmplitude))
        self.sourcePhase = np.concatenate((self.sourcePhase, sourcePhase))
        self.sourceFrequency = np.concatenate((self.sourceFrequency, sourceFrequency))
        self.sourceTimeEndPoints = np.concatenate((self.sourceTimeEndPoints, sourceTimeEndPoints))
        self.sourceType = np.concatenate((self.sourceType, sourceType))
        self.sourceIndexMax += 1

class StateProperties:
    """
    The initial state fed into the simulation code
    """
    def __init__(self, stateInit = np.asarray([1, 0, 0])):
        self.stateInit = stateInit

    def writeToFile(self, archive):
        archiveGroup = archive.require_group("stateProperties")
        archiveGroup["stateInit"] = self.stateInit

class SimulationResults:
    """
    The output of the simulation code
    """
    def __init__(self, signal):
        self.timeEvolution = np.empty([signal.timeProperties.timeIndexMax, 3, 3], np.cdouble)   # Time evolution operator between the current and next timesteps (SU3) [time index, matrix [y, x]]
        self.state = np.empty([signal.timeProperties.timeIndexMax, 3], np.cdouble)              # Lab frame quantum state (C3 unit vector) [time index, spin projection index]
        self.spin = np.empty([signal.timeProperties.timeIndexMax, 3], np.double)                # Expected hyperfine spin (hbar) [time index, xyz direction]
        self.sensedFrequencyAmplitude = 0.0                                                     # Measured sine coefficient (Hz)
        self.sensedFrequencyAmplitudeMethod = "none"                                            # Whether demodulation or demapping was used (string)

    def writeToFile(self, archive, doEverything = False):
        """
        Saves results to the hdf5 file
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
    The data needed and algorithms to control an individual simulation
    """
    def __init__(self, signal, dressingRabiFrequency = 1e3, stateProperties = StateProperties(), trotterCutoff = 52):
        self.signal = signal
        self.sourceProperties = SourceProperties(self.signal, dressingRabiFrequency)
        self.stateProperties = stateProperties
        self.simulationResults = SimulationResults(self.signal)
        self.trotterCutoff = trotterCutoff

        self.evaluate()

    def evaluate(self):
        """
        Time evolve the system, and find the spin at each coarse time step
        """
        # Decide GPU block and grid sizes
        threadsPerBlock = 64
        blocksPerGrid = (self.signal.timeProperties.timeIndexMax + (threadsPerBlock - 1)) // threadsPerBlock

        # Run stepwise solver
        # getTimeEvolutionMidpointSample[blocksPerGrid, threadsPerBlock](self.signal.timeProperties.timeCoarse, self.signal.timeProperties.timeEndPoints, self.signal.timeProperties.timeStepFine, self.signal.timeProperties.timeStepCoarse, self.sourceProperties.sourceIndexMax, self.sourceProperties.sourceAmplitude, self.sourceProperties.sourceFrequency, self.sourceProperties.sourcePhase, self.sourceProperties.sourceTimeEndPoints, self.simulationResults.timeEvolution, self.trotterCutoff)
        getTimeEvolutionCommutatorFree4RotatingWave[blocksPerGrid, threadsPerBlock](self.signal.timeProperties.timeCoarse, self.signal.timeProperties.timeEndPoints, self.signal.timeProperties.timeStepFine, self.signal.timeProperties.timeStepCoarse, self.sourceProperties.sourceIndexMax, self.sourceProperties.sourceAmplitude, self.sourceProperties.sourceFrequency, self.sourceProperties.sourcePhase, self.sourceProperties.sourceTimeEndPoints, self.sourceProperties.sourceQuadraticShift, self.simulationResults.timeEvolution, self.trotterCutoff)

        # Combine results of the stepwise solver to evaluate the timeseries for the state
        getState(self.stateProperties.stateInit, self.simulationResults.state, self.simulationResults.timeEvolution)

        # Evaluate the time series for the expected spin value
        getSpin[blocksPerGrid, threadsPerBlock](self.simulationResults.state, self.simulationResults.spin)

    def getFrequencyAmplitudeFromDemodulation(self, demodulationTimeEndPoints = [0.09, 0.1], doPlotSpin = False):
        """
        Use demodulation of the Faraday signal to find the measured Fourier coefficient
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
        Save the simulation record to hdf5
        """
        self.sourceProperties.writeToFile(archive)
        self.stateProperties.writeToFile(archive)
        self.simulationResults.writeToFile(archive, False)

@cuda.jit(debug = cudaDebug,  max_registers = 63)
def getTimeEvolutionCommutatorFree4RotatingWave(timeCoarse, timeEndPoints, timeStepFine, timeStepCoarse,
    sourceIndexMax, sourceAmplitude, sourceFrequency, sourcePhase, sourceTimeEndPoints, sourceQuadraticShift,
    timeEvolutionCoarse, trotterCutoff):
    """
    Find the stepwise time evolution opperator using a 2 exponential commutator free order 4 Magnus integrator, in a rotating frame.
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
    Find the stepwise time evolution opperator using a 2 exponential commutator free order 4 Magnus integrator.
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
    The same sampling as used in the cython code. Uses two exponentials per fine timestep
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
    The most basic form of integrator, uses one exponential per time step
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
    Use the stepwise time evolution operators in succession to find the quantum state timeseries of the 3 level atom
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
    Calculate each expected spin value in parallel
    """
    timeIndex = cuda.threadIdx.x + cuda.blockIdx.x*cuda.blockDim.x
    if timeIndex < spin.shape[0]:
        spin[timeIndex, 0] = (2*conj(state[timeIndex, 1])*(state[timeIndex, 0] + state[timeIndex, 2])/sqrt2).real
        spin[timeIndex, 1] = (2j*conj(state[timeIndex, 1])*(state[timeIndex, 0] - state[timeIndex, 2])/sqrt2).real
        spin[timeIndex, 2] = state[timeIndex, 0].real**2 + state[timeIndex, 0].imag**2 - state[timeIndex, 2].real**2 - state[timeIndex, 2].imag**2

@cuda.jit(debug = cudaDebug)
def getFrequencyAmplitudeFromDemodulationMultiply(time, spin, spinDemodulated, biasAmplitude):
    """
    Multiply each spin value by -2cos(wt) as part of a demodulation
    """
    timeIndex = cuda.threadIdx.x + cuda.blockIdx.x*cuda.blockDim.x
    if timeIndex < spin.shape[0]:
        spinDemodulated[timeIndex] = -2*math.cos(2*math.pi*biasAmplitude*time[timeIndex])*spin[timeIndex]

@nb.jit(nopython = True)
def getFrequencyAmplitudeFromDemodulationLowPass(timeEndPoints, spin, sensedFrequencyAmplitude):
    """
    Average the multiplied spin to find the DC value (ie apply a low pass filter)
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
    Demodulate a spin timeseries with a basic block low pass filter
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