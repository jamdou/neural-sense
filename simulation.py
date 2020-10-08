import numpy as np
import scipy.linalg as spla
import matplotlib.pyplot as plt
import math
from numba import cuda
import numba as nb
import time as tm
from testSignal import *

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
    def __init__(self, signal, dressingRabiFrequency):
        self.dressingRabiFrequency = dressingRabiFrequency
        self.sourceIndexMax = 0                             # The number of sources
        self.sourceAmplitude = np.empty([0, 3])             # Sine wave amplitude (Hz) [source index, xyz direction]
        self.sourcePhase = np.empty([0, 3])                 # Sine wave phase offset, starts from 0 (rad) [source index, xyz direction]
        self.sourceFrequency = np.empty([0, 3])             # Sine wave frequency, separate for each direction (Hz) [source index, xyz direction]
        self.sourceTimeEndPoints = np.empty([0, 2])         # Sine wave start and end times (s) [source index, start time / end time]
        self.sourceType = np.empty([0], dtype = object)     # Description of what the source physically represents, for archive (string) [source index]

        # Construct the signal from the dressing information and pulse description.
        if signal:
            self.addDressing(signal, dressingRabiFrequency)
            for neuralPulse in signal.neuralPulses:
                self.addNeuralPulse(neuralPulse.timeStart, neuralPulse.amplitude, neuralPulse.frequency)
        
    def writeToFile(self, archive):
        """
        Saves source information to archive file
        """
        archiveGroup = archive.create_group("sourceProperties")
        archiveGroup["sourceAmplitude"] = self.sourceAmplitude
        archiveGroup["sourcePhase"] = self.sourcePhase
        archiveGroup["sourceFrequency"] = self.sourceFrequency
        archiveGroup["sourceTimeEndPoints"] = self.sourceTimeEndPoints
        archiveGroup["sourceType"] = np.asarray(self.sourceType, dtype='|S32')
        archiveGroup["sourceIndexMax"] = np.asarray([self.sourceIndexMax])

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
        archiveGroup = archive.create_group("stateProperties")
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
        archiveGroup = archive.create_group("simulationResults")
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
        getTimeEvolutionCommutatorFree4[blocksPerGrid, threadsPerBlock](
            self.signal.timeProperties.timeCoarse, self.signal.timeProperties.timeEndPoints, self.signal.timeProperties.timeStepFine, self.signal.timeProperties.timeStepCoarse, self.sourceProperties.sourceIndexMax, self.sourceProperties.sourceAmplitude, self.sourceProperties.sourceFrequency, self.sourceProperties.sourcePhase, self.sourceProperties.sourceTimeEndPoints, self.simulationResults.timeEvolution, self.trotterCutoff)

        # Combine results of the stepwise solver to evaluate the timeseries for the state
        getState(self.stateProperties.stateInit, self.simulationResults.state, self.simulationResults.timeEvolution)

        # Evaluate the time series for the expected spin value
        getSpin[blocksPerGrid, threadsPerBlock](self.simulationResults.state, self.simulationResults.spin)

    def getFrequencyAmplitudeFromDemodulation(self, demodulationTimeEndPoints = [0.09, 0.1]):
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
        # plt.show()

    def writeToFile(self, archive):
        """
        Save the simulation record to hdf5
        """
        self.sourceProperties.writeToFile(archive)
        self.stateProperties.writeToFile(archive)
        self.simulationResults.writeToFile(archive, False)

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
        for timeFineIndex in range(timeStepCoarse/timeStepFine):
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
    timeEvolutionCoarse):
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
        for timeFineIndex in range(timeStepCoarse/timeStepFine):

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

                    matrixExponentialLieTrotter(hamiltonian, timeEvolutionFine)

                    setTo(timeEvolutionCoarse[timeIndex, :], timeEvolutionOld)
                    matrixMultiply(timeEvolutionFine, timeEvolutionOld, timeEvolutionCoarse[timeIndex, :])

            timeFine += timeStepFine

@cuda.jit(debug = cudaDebug,  max_registers = 31)
def getTimeEvolutionMidpointSample(timeCoarse, timeEndPoints, timeStepFine, timeStepCoarse,
    sourceIndexMax, sourceAmplitude, sourceFrequency, sourcePhase, sourceTimeEndPoints,
    timeEvolutionCoarse):
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
        for timeFineIndex in range(timeStepCoarse/timeStepFine):
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

            matrixExponentialCrossProduct(hamiltonian, timeEvolutionFine)

            setTo(timeEvolutionCoarse[timeIndex, :], timeEvolutionOld)
            matrixMultiply(timeEvolutionFine, timeEvolutionOld, timeEvolutionCoarse[timeIndex, :])

            timeFine += timeStepFine

@cuda.jit(device = True, debug = cudaDebug)
def matrixExponentialCrossProduct(exponent, result):
    """
    Calculate a matrix exponential by diagonalising using the cross product
    """
    diagonal = cuda.local.array(3, dtype = nb.float64)          # Eigenvalues
    winding = cuda.local.array(3, dtype = nb.complex128)        # e^i eigenvalues
    rotation = cuda.local.array((3, 3), dtype = nb.complex128)  # Eigenvectors
    shifted = cuda.local.array((3, 3), dtype = nb.complex128)   # Temp, for eigenvalue algorithm
    scaled = cuda.local.array((3, 3), dtype = nb.complex128)    # Temp, for eigenvalue algorithm

    setToOne(rotation)

    # Find eigenvalues
    # Based off a trigonometric solution to the cubic characteristic equation
    # First shift and scale the matrix to a nice position
    shift = 1j*(exponent[0, 0] + exponent[1, 1] + exponent[2, 2])/3
    for xIndex in range(3):
        for yIndex in range(3):
            shifted[yIndex, xIndex] = 1j*exponent[yIndex, xIndex]
            if yIndex == xIndex:
                shifted[yIndex, xIndex] -= shift
    matrixMultiply(shifted, shifted, scaled)
    scale = math.sqrt((scaled[0, 0] + scaled[1, 1] + scaled[2, 2]).real/6)
    if scale > 0:
        for xIndex in range(3):
            for yIndex in range(3):
                scaled[yIndex, xIndex] = shifted[yIndex, xIndex]/scale
    # Now find the determinant of the shifted and scaled matrix
    detScaled = 0
    for xIndex in range(3):
        detPartialPlus = 1
        detPartialMinus = 1
        for yIndex in range(3):
            detPartialPlus *= scaled[yIndex, (xIndex + yIndex)%3]
            detPartialMinus *= scaled[2 - yIndex, (xIndex + yIndex)%3]
        detScaled += (detPartialPlus - detPartialMinus).real
    # The eigenvalues of the matrix are related to the determinant of the scaled matrix
    for diagonalIndex in range(3):
        diagonal[diagonalIndex] = scale*2*math.cos((1/3)*math.acos(detScaled/2) + (2/3)*math.pi*diagonalIndex)

    # First eigenvector
    # Eigenvector y of eigenvalue s of A are in the kernel of (A - s 1).
    # Something something first isomorphism theorem =>
    # y is parallel to all vectors in the coimage of (A - s 1), ie the image of (A - s 1)* (hermitian adjoint,
    # sorry, can't type a dagger, will use maths notation), ie the column space of (A - s 1)*. Love me some linear
    # algebra. Can find such a vector by cross producting vectors in this coimage.

    # Find (A - s 1)*
    for xIndex in range(3):
        for yIndex in range(3):
            shifted[yIndex, xIndex] = 1j*exponent[yIndex, xIndex]
            if yIndex == xIndex:
                shifted[yIndex, xIndex] -= diagonal[0]
    adjoint(shifted, scaled)
    
    # Normalise vectors in the coimage of (A - s 1)
    hasAlreadyFoundIt = False
    for xIndex in range(3):
        norm = norm2(scaled[:, xIndex])
        if norm > machineEpsilon:
            for yIndex in range(3):
                scaled[yIndex, xIndex] /= norm
        else:
            # If the column of (A - s 1)* has size 0, then e1 is an eigenvector.
            # Not more work needs to be done to find it.
            for yIndex in range(3):
                if xIndex == yIndex:
                    rotation[yIndex, 0] = 1
                else:
                    rotation[yIndex, 0] = 0
                hasAlreadyFoundIt = True

    if not hasAlreadyFoundIt:
        # Find the cross product of vectors in the coimage
        cross(scaled[:, 0], scaled[:, 1], rotation[:, 0])
        norm = norm2(rotation[:, 0])
        if norm > machineEpsilon:
            # If these vectors cross to something non-zero then we've found one
            for xIndex in range(3):
                rotation[xIndex, 0] /= norm
        else:
            # Otherwise these are parallel, and we should cross other vectors to see if that helps
            cross(scaled[:, 1], scaled[:, 2], rotation[:, 0])
            norm = norm2(rotation[:, 0])
            if norm > machineEpsilon:
                for xIndex in range(3):
                    rotation[xIndex, 0] /= norm
            else:
                # If it's still zero the we should have found it already, so panic.
                print("RIIIIP")

    # Second eigenvector
    for xIndex in range(3):
        for yIndex in range(3):
            shifted[yIndex, xIndex] = 1j*exponent[yIndex, xIndex]
            if yIndex == xIndex:
                shifted[yIndex, xIndex] -= diagonal[1]
    adjoint(shifted, scaled)
    hasAlreadyFoundIt = False
    for xIndex in range(3):
        norm = norm2(scaled[:, xIndex])
        if norm > machineEpsilon:
            for yIndex in range(3):
                scaled[yIndex, xIndex] /= norm
        else:
            for yIndex in range(3):
                if xIndex == yIndex:
                    rotation[yIndex, 1] = 1
                else:
                    rotation[yIndex, 1] = 0
                hasAlreadyFoundIt = True
    # If not part of same eigenspace as the first one the proceed as normal,
    # otherwise find a vector orthogonal to the first eigenvector
    if math.fabs(diagonal[0] - diagonal[1]) > machineEpsilon:
        if not hasAlreadyFoundIt:
            cross(scaled[:, 0], scaled[:, 1], rotation[:, 1])
            norm = norm2(rotation[:, 1])
            if norm > machineEpsilon:
                for xIndex in range(3):
                    rotation[xIndex, 1] /= norm
            else:
                cross(scaled[:, 1], scaled[:, 2], rotation[:, 1])
                norm = norm2(rotation[:, 1])
                if norm > machineEpsilon:
                    for xIndex in range(3):
                        rotation[xIndex, 1] /= norm
                else:
                    print("RIIIIP")
    else:
        cross(scaled[:, 0], rotation[:, 0], rotation[:, 1])
        norm = norm2(rotation[:, 1])
        if norm > machineEpsilon:
            for xIndex in range(3):
                rotation[xIndex, 1] /= norm
        else:
            cross(scaled[:, 1], rotation[:, 0], rotation[:, 1])
            norm = norm2(rotation[:, 1])
            if norm > machineEpsilon:
                for xIndex in range(3):
                    rotation[xIndex, 1] /= norm
            else:
                print("RIIIIP")

    # Third eigenvector
    # The third eigenvector is orthogonal to the first two
    cross(rotation[:, 0], rotation[:, 1], rotation[:, 2])

    # winding = exp(-j w)
    for xIndex in range(3):
        winding[xIndex] = math.cos(diagonal[xIndex]) - 1j*math.sin(diagonal[xIndex])
    # U = exp( Q* diag(-j w) Q ) = Q* diag(exp(-j w)) Q
    for xIndex in range(3):
        for yIndex in range(3):
            result[yIndex, xIndex] = 0
            for zIndex in range(3):
                result[yIndex, xIndex] += rotation[yIndex, zIndex]*winding[zIndex]*conj(rotation[xIndex, zIndex])

@cuda.jit(device = True, debug = cudaDebug)
def matrixExponentialLieTrotter(exponent, result, trotterCutoff = 52):
    x = (1j*exponent[1, 0]).real*sqrt2
    y = (1j*exponent[1, 0]).imag*sqrt2
    z = (1j*(exponent[0, 0] + 0.5*exponent[1, 1])).real
    q = (-1.5*1j*exponent[1, 1]).real

    hyperCubeAmount = 2*math.ceil(trotterCutoff/4 + math.log(math.fabs(x) + math.fabs(y) + math.fabs(z) + math.fabs(q))/(4*math.log(2.0)))
    precision = 4**hyperCubeAmount

    x /= precision
    y /= precision
    z /= precision
    q /= precision

    cx = math.cos(x)
    sx = math.sin(x)
    cy = math.cos(y)
    sy = math.sin(y)

    cisz = math.cos(z + q/3) - 1j*math.sin(z + q/3)
    result[0, 0] = 0.5*cisz*(cx + cy - 1j*sx*sy)
    result[1, 0] = cisz*(-1j*sx + cx*sy)/sqrt2
    result[2, 0] = 0.5*cisz*(cx - cy - 1j*sx*sy)

    cisz = math.cos(2*q/3) + 1j*math.sin(2*q/3)
    result[0, 1] = cisz*(-sy - 1j*cy*sx)/sqrt2
    result[1, 1] = cisz*cx*cy
    result[2, 1] = cisz*(sy - 1j*cy*sx)/sqrt2

    cisz = math.cos(z - q/3) + 1j*math.sin(z - q/3)
    result[0, 2] = 0.5*cisz*(cx - cy + 1j*sx*sy)
    result[1, 2] = cisz*(-1j*sx - cx*sy)/sqrt2
    result[2, 2] = 0.5*cisz*(cx + cy + 1j*sx*sy)

    for powerIndex in range(hyperCubeAmount):
        matrixMultiply(result, result, exponent)
        matrixMultiply(exponent, exponent, result)

@cuda.jit(device = True, debug = cudaDebug)
def matrixExponentialTaylor(exponent, result):
    """
    Calculate a matrix exponential using a Taylor series
    """
    T = cuda.local.array((3, 3), dtype = nb.complex128)
    TOld = cuda.local.array((3, 3), dtype = nb.complex128)
    setToOne(T)
    setToOne(result)

    # exp(A) = 1 + A + A^2/2 + ...
    for taylorIndex in range(expPrecision):
        # TOld = T
        for xIndex in nb.prange(3):
            for yIndex in nb.prange(3):
                TOld[yIndex, xIndex] = T[yIndex, xIndex]
        # T = TOld*A/n
        for xIndex in nb.prange(3):
            for yIndex in nb.prange(3):
                T[yIndex, xIndex] = 0
                for zIndex in range(3):
                    T[yIndex, xIndex] += (TOld[yIndex, zIndex]*exponent[zIndex, xIndex])/(taylorIndex + 1)
        # E = E + T
        for xIndex in nb.prange(3):
            for yIndex in nb.prange(3):
                result[yIndex, xIndex] += T[yIndex, xIndex]

@cuda.jit(device = True, debug = cudaDebug)
def matrixExponentialQL(exponent, result):
    """
    This is based on a method in the cython - I couldn't get it to work in either this or cython, but it isn't necessary.
    """
    diagonal = cuda.local.array(3, dtype = nb.float64)
    offDiagonal = cuda.local.array(2, dtype = nb.float64)
    winding = cuda.local.array(3, dtype = nb.complex128)
    rotation = cuda.local.array((3, 3), dtype = nb.complex128)

    setToOne(result)

    # initialise
    #       ( e^ i phi                  ) ( Bz            Bphi/sqrt2             ) ( e^ i phi                  )
    # -i    (             1             ) ( Bphi/sqrt2                Bphi/sqrt2 ) (             1             )
    #       (                 e^ -i phi ) (               Bphi/sqrt2  -Bz        ) (                 e^ -i phi )

    diagonal[0] = -exponent[0, 0].imag
    diagonal[1] = -exponent[1, 1].imag
    diagonal[2] = -exponent[2, 2].imag
    offDiagonal[0] = complexAbs(exponent[1, 0])
    offDiagonal[1] = complexAbs(exponent[2, 1])
    # offDiagonal[2] = 0

    setToOne(rotation)
    if offDiagonal[0] > 0:
        rotation[0, 0] = 1j*exponent[1, 0]/offDiagonal[0]
    if offDiagonal[1] > 0:
        rotation[2, 2] = 1j*exponent[2, 1]/offDiagonal[1]

    for offDiagonalIndex in range(0, 2):
        iterationIndex = 0
        while True:
            # If floating point arithmetic can't tell the difference between the size of the off diagonals and zero, then we're good and can stop
            for offDiagonalCompareIndex in range(offDiagonalIndex, 2):
                orderOfMagnitude = math.fabs(diagonal[offDiagonalCompareIndex]) + math.fabs(diagonal[offDiagonalCompareIndex + 1])
                if math.fabs(offDiagonal[offDiagonalCompareIndex]) + orderOfMagnitude == orderOfMagnitude:
                    break
                # if math.fabs(offDiagonal[offDiagonalCompareIndex])/orderOfMagnitude < 1e-6:
                #     break
            if offDiagonalCompareIndex == offDiagonalIndex:
                break
            
            iterationIndex += 1
            if iterationIndex > 60:
                print("yote")
                break

            temporaryG = 0.5*(diagonal[offDiagonalIndex + 1] - diagonal[offDiagonalIndex])/offDiagonal[offDiagonalIndex]
            size = math.sqrt(temporaryG**2 + 1.0)
            if temporaryG > 0:
                temporaryG = diagonal[offDiagonalCompareIndex] - diagonal[offDiagonalIndex] + offDiagonal[offDiagonalIndex]/(temporaryG + size)
            else:
                temporaryG = diagonal[offDiagonalCompareIndex] - diagonal[offDiagonalIndex] + offDiagonal[offDiagonalIndex]/(temporaryG - size)

            diagonaliseSine = 1.0
            diagonaliseCosine = 1.0
            temporaryP = 0.0
            temporaryB = 0.0
            for offDiagonalCalculationIndex in range(offDiagonalCompareIndex - 1, offDiagonalIndex - 1, -1):
                temporaryF = diagonaliseSine*offDiagonal[offDiagonalCalculationIndex]
                temporaryB = diagonaliseCosine*offDiagonal[offDiagonalCalculationIndex]
                if math.fabs(temporaryF) > math.fabs(temporaryG):
                    diagonaliseCosine = temporaryG/temporaryF
                    size = math.sqrt(diagonaliseCosine**2 + 1.0)
                    offDiagonal[offDiagonalCalculationIndex + 1] = temporaryF*size
                    diagonaliseSine = 1.0/size
                    diagonaliseCosine *= diagonaliseSine
                else:
                    diagonaliseSine = temporaryF/temporaryG
                    size = math.sqrt(diagonaliseSine**2 + 1.0)
                    offDiagonal[offDiagonalCalculationIndex + 1] = temporaryG*size
                    diagonaliseCosine = 1.0/size
                    diagonaliseSine *= diagonaliseCosine
            temporaryG = diagonal[offDiagonalCalculationIndex + 1] - temporaryP
            size = (diagonal[offDiagonalCalculationIndex] - temporaryG)*diagonaliseSine + 2.0*diagonaliseCosine*temporaryB
            temporaryP = diagonaliseSine*size
            diagonal[offDiagonalCalculationIndex + 1] = temporaryG + temporaryP
            temporaryG = diagonaliseCosine*size - temporaryB

            for rotationIndex in range(0, 3):
                rotationPrevious = rotation[rotationIndex, offDiagonalCalculationIndex + 1]
                rotation[rotationIndex, offDiagonalCalculationIndex + 1] = diagonaliseSine*rotation[rotationIndex, offDiagonalCalculationIndex] + diagonaliseCosine*rotationPrevious
                rotation[rotationIndex, offDiagonalCalculationIndex] = diagonaliseCosine*rotation[rotationIndex, offDiagonalCalculationIndex] - diagonaliseSine*rotationPrevious

            diagonal[offDiagonalIndex] -= temporaryP
            offDiagonal[offDiagonalIndex] = temporaryG
            offDiagonal[offDiagonalCompareIndex] = 0.0
    # winding = exp(-j w)
    for xIndex in range(3):
        winding[xIndex] = math.cos(diagonal[xIndex]) - 1j*math.sin(diagonal[xIndex])
    if cuda.threadIdx.x + cuda.blockIdx.x*cuda.blockDim.x == 0:
        print(diagonal[0])
    # U = exp( Q* diag(-j w) Q ) = Q* diag(exp(-j w)) Q
    for xIndex in range(3):
        for yIndex in range(3):
            result[yIndex, xIndex] = 0
            for zIndex in range(3):
                result[yIndex, xIndex] += winding[zIndex]*conj(rotation[xIndex, zIndex])*rotation[yIndex, zIndex]
    return

@cuda.jit(device = True, debug = cudaDebug)
def conj(z):
    """
    Conjugate of a complex number in cuda. For some reason doesn't support a built in one so I had to write this.
    """
    return (z.real - 1j*z.imag)

@cuda.jit(device = True, debug = cudaDebug)
def complexAbs(z):
    """
    The absolute value of a complex number
    """
    return math.sqrt(z.real**2 + z.imag**2)

@cuda.jit(device = True, debug = cudaDebug)
def norm2(z):
    """
    The 2 norm of a vector in C3
    """
    return math.sqrt(z[0].real**2 + z[0].imag**2 + z[1].real**2 + z[1].imag**2 + z[2].real**2 + z[2].imag**2)

@cuda.jit(device = True, debug = cudaDebug)
def cross(left, right, result):
    """
    The cross product of two vectors in C3. Note: I'm using the maths convection here, taking the
    conjugate of the real cross product, since I want this to produce an orthogonal vector.
    """
    for xIndex in range(3):
        result[xIndex] = conj(left[(xIndex + 1)%3]*right[(xIndex + 2)%3] - left[(xIndex + 2)%3]*right[(xIndex + 1)%3])

@cuda.jit(device = True, debug = cudaDebug)
def inner(left, right):
    """
    The inner (maths convention dot) product between two vectors in C3. Note the left vector is conjugated.
    Thus the inner product of two orthogonal vectors is 0.
    """
    return conj(left[0])*right[0] + conj(left[1])*right[1] + conj(left[2])*right[2]

@cuda.jit(device = True, debug = cudaDebug)
def setTo(operator, result):
    """
    result = operator, for two matrices in C3x3.
    """
    for xIndex in range(3):
        for yIndex in range(3):
            result[yIndex, xIndex] = operator[yIndex, xIndex]

@cuda.jit(device = True, debug = cudaDebug)
def setToOne(operator):
    """
    Make operator the C3x3 identity matrix (ie, 1)
    """
    for xIndex in nb.prange(3):
        for yIndex in nb.prange(3):
            if xIndex == yIndex:
                operator[yIndex, xIndex] = 1
            else:
                operator[yIndex, xIndex] = 0

@cuda.jit(device = True, debug = cudaDebug)
def setToZero(operator):
    """
    Make operator the C3x3 zero matrix
    """
    for xIndex in nb.prange(3):
        for yIndex in nb.prange(3):
            operator[yIndex, xIndex] = 0

@cuda.jit(device = True, debug = cudaDebug)
def matrixMultiply(left, right, result):
    """
    Multiply C3x3 matrices left and right together, to be returned in result.
    """
    for xIndex in range(3):
        for yIndex in range(3):
            result[yIndex, xIndex] = left[yIndex, 0]*right[0, xIndex] + left[yIndex, 1]*right[1, xIndex] + left[yIndex, 2]*right[2, xIndex]

@cuda.jit(device = True, debug = cudaDebug)
def adjoint(operator, result):
    """
    Take the hermitian adjoint (ie dagger, ie dual, ie conjugate transpose) of a C3x3 matrix
    """
    for xIndex in range(3):
        for yIndex in range(3):
            result[yIndex, xIndex] = conj(operator[xIndex, yIndex])

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