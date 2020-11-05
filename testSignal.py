"""
Classes to store information about magnetic signals being simulated and reconstructed, as well as timing.
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from numba import cuda
import numba as nb

cudaDebug = False

class TimeProperties:
    """
    Grouped details about time needed for simulations and reconstructions.

    Attributes
    ----------
    timeStepCoarse : `float`
        The time difference between each element of `timeCoarse`. In units of s. Determines the sample rate of the outputs `timeCoarse` and `timeEvolutionCoarse`.
    timeStepFine : `float`
        The time step used within the integration algorithm. In units of s.
    timeEndPoints : :class:`numpy.ndarray` of :class:`numpy.double` (start time (0) or end time (1))
        The time values for when the experiment is to start and finishes. In units of s.
    timeIndexMax : `int`
        The number of times to be sampled. That is, the size of `timeCoarse`.
    timeCoarse : :class:`numpy.ndarray` of :class:`numpy.double` (timeIndex)
        A coarse grained list of time samples that the signal is defined for. In units of s.
    """
    def __init__(self, timeStepCoarse = 5e-7, timeStepFine = 1e-7, timeEndPoints = [0, 1e-1]):
        """
        Parameters
        ----------
        timeStepCoarse : `float`, optional
            The time difference between each element of `timeCoarse`. In units of s. Determines the sample rate of the outputs `timeCoarse` and `timeEvolutionCoarse`. Defaults to 500ns.
        timeStepFine : `float`, optional
            The time step used within the integration algorithm. In units of s. Defaults to 100ns.
        timeEndPoints : :class:`numpy.ndarray` of :class:`numpy.double` (start time (0) or end time (1))
            The time values for when the experiment is to start and finishes. In units of s. Defaults to between 0s and 100ms.
        """
        self.timeStepCoarse = timeStepCoarse
        self.timeStepFine = timeStepFine
        self.timeEndPoints = np.asarray(timeEndPoints, np.double)
        self.timeIndexMax = int((self.timeEndPoints[1] - self.timeEndPoints[0])/self.timeStepCoarse)
        self.timeCoarse = np.empty(self.timeIndexMax)

    def writeToFile(self, archive):
        """
        Saves the time properties to a hdf5 file.

        Parameters
        ----------
        archive : :class:`h5py.Group`
            The HDF5 archive to write to.
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
    A simple parametrised description of a neural magnetic pulse, being approximated as a half sine pulse.

    Attributes
    ----------
    timeStart : `float`
        The point in time where the neural pulse first becomes non-zero. In units of s.
    amplitude : `float`
        The peak amplitude of the neural pulse. In units of Hz.
    frequency : `float`
        The mean frequency of the neural pulse. In units of Hz.
    """
    def __init__(self, timeStart = 0.0, amplitude = 10.0, frequency = 1e3):
        """
        Parameters
        ----------
        timeStart : `float`, optional
            The point in time where the neural pulse first becomes non-zero. In units of s. Defaults to 0.
        amplitude : `float`, optional
            The peak amplitude of the neural pulse. In units of Hz. Defaults to 10Hz.
        frequency : `float`, optional
            The mean frequency of the neural pulse. In units of Hz. Defaults to 1kHz.
        """
        self.timeStart = timeStart
        self.amplitude = amplitude
        self.frequency = frequency

    def writeToFile(self, archive):
        """
        Writes the pulse details to a hdf5 file.

        Parameters
        ----------
        archive : :class:`h5py.Group`
            The HDF5 archive to write to.
        """
        archive["timeStart"] = self.timeStart
        archive["amplitude"] = self.amplitude
        archive["frequency"] = self.frequency

class SinusoidalNoise:
    """
    For describing 50Hz and detuning noise.

    Attributes
    ----------
    amplitude : :class:`numpy.ndarray` of :class:`numpy.double`, (spatialIndex)
        The amplitude of each spatial component of the noise as a 3D vector. In units of Hz.
    frequency : :class:`numpy.ndarray` of :class:`numpy.double`, (spatialIndex)
        The frequency of each spatial component of the noise as a 3D vector. In units of Hz.
    phase : :class:`numpy.ndarray` of :class:`numpy.double`, (spatialIndex)
        The frequency of each spatial component of the noise as a 3D vector. In units of rad.
    type : `string`
        A label for the kind of noise this represents. For archival purposes.
    """
    def __init__(self, amplitude = [0, 0, 0], frequency = [50, 50, 50], phase = [math.pi/2, math.pi/2, math.pi/2], type = "SinusoidalNoise"):
        """
        Parameters
        ----------
        amplitude : :class:`numpy.ndarray` of :class:`numpy.double`, (spatialIndex)
        The amplitude of each spatial component of the noise as a 3D vector. In units of Hz. Defaults to 0Hz in each direction.
        frequency : :class:`numpy.ndarray` of :class:`numpy.double`, (spatialIndex)
            The frequency of each spatial component of the noise as a 3D vector. In units of Hz. Defaults to 50Hz in each direction.
        phase : :class:`numpy.ndarray` of :class:`numpy.double`, (spatialIndex)
            The frequency of each spatial component of the noise as a 3D vector. In units of rad. Defaults to :math:`\\frac{\\pi}{2}` rad in each direction.
        type : `string`, optional
        A label for the kind of noise this represents. For archival purposes. Defaults to `"SinusoidalNoise"`.
        """
        self.amplitude = np.asarray(amplitude, dtype=np.double)
        self.frequency = np.asarray(frequency, dtype=np.double)

        self.phase = np.asarray(phase, dtype=np.double)
        self.type = type

    @staticmethod
    def newDetuningNoise(zAmplitude = 1.0):
        """
        A shorthand constructor for detuning noise (ie dc noise along the z direction).
        
        .. note::
            Inverts the sign of the amplitude to obey the regular convention of

            .. math::
                \\begin{align*}
                    f_\\textrm{dressing} &= f_\\textrm{amp, bias} + d\\\\
                    \\omega_\\textrm{dressing} &= \\omega_\\textrm{amp, bias} + \\Delta,
                \\end{align*}

            with :math:`d` and :math:`\\Delta` the detuning in Hz and rad/s respectively.

            That is, a dc offset of :math:`-d` along the z direction will give a detuning of :math:`d` under this convention.

        Parameters
        ----------
        zAmplitude : `float`, optional
            The amplitude of the detuning. See :math:`d` above. Measured in Hz. Defaults to 1Hz.

        Returns
        -------
        sinusoidalNoise : :class:`SinusoidalNoise`
            A parametrisation of the detuning noise.
        """
        return SinusoidalNoise([0, 0, -zAmplitude], [0, 0, 0], [math.pi/2, math.pi/2, math.pi/2], "DetuningNoise")

    @staticmethod
    def newLineNoise(amplitude = [100.0, 0.0, 0.0], phase = [0.0, 0.0, 0.0]):
        """
        A shorthand constructor for line noise.

        .. note::
            Uses the Australian standard of 50Hz line noise.

        Parameters
        ----------
        amplitude : :class:`numpy.ndarray` of :class:`numpy.double`, (spatialIndex)
            The amplitude of each spatial component of the noise as a 3D vector. In units of Hz. Defaults to 100Hz in the x direction only.
        phase : :class:`numpy.ndarray` of :class:`numpy.double`, (spatialIndex)
            The frequency of each spatial component of the noise as a 3D vector. In units of rad. Defaults to 0rad in each direction.

        Returns
        -------
        sinusoidalNoise : :class:`SinusoidalNoise`
            A parametrisation of the line noise.
        """
        return SinusoidalNoise(amplitude, [50, 50, 50], phase, "LineNoise")

    def writeToFile(self, archive):
        """
        Writes the pulse details to a hdf5 file.

        Parameters
        ----------
        archive : :class:`h5py.Group`
            The HDF5 archive to write to.
        """
        archive["amplitude"] = self.amplitude
        archive["frequency"] = self.frequency
        archive["phase"] = self.phase
        archive["type"] = np.asarray(self.type, dtype='|S32')
        
class TestSignal:
    """
    Details for a simulated neural pulse sequence and noise (no control signals).

    Here the convention is used of measuring signal amplitudes in terms of the energy splitting they give to the spin system. This is seen in the lab as oscillations (by the frequency equivalent to this energy given by Plank's constant, :math:`f = \\frac{E}{h}=\\frac{E}{2\\pi\\hbar}`). For this reason all amplitudes are in units of Hz.

    Fourier sine coefficients :math:`s(f) = s_f` of the time domain signal :math:`r(t) = r_t`, are defined by

    .. math::
        \\begin{align*}
            s(f) &= \\frac{1}{T}\\int_0^T \\sin(2\\pi f t) r(t) \\mathrm{d}t \\\\
            s_f &= \\sum_t \\frac{\\mathrm{d}t}{T}\\sin(2\\pi f t) r_t \\\\
            s_f &= S_{f,t} r_t, \\textrm{ using Einstein notation, with} \\\\
            S_{f,t} &= \\left( \\frac{\\mathrm{d}t}{T}\\sin(2\\pi f t) \\right)_{f,t}, \\textrm{ as the sine fourier transform.}
        \\end{align*}

    Note, the sine Fourier Transform has singular values of :math:`\\sqrt{\\frac{\\mathrm{d}t}{2T}}`, and the Moore-Penrose psudoinverse of :math:`S_{f,t}` is

    .. math::
        \\begin{align*}
            \\left(S_{f,t}^+\\right)_{t, f} &= \\left( \\frac{2T}{\\mathrm{d}t} S_{f, t} \\right)_{t, f}
        \\end{align*}
    
    Attributes
    ----------
    timeProperties : :class:`TimeProperties`
        Details describing time for the signal.
    neuralPulses : `list` of :class:`NeuralPulse`
        List of individual neural pulses that make up the signal.
    sinusoidalNoises : `list` of :class:`SinusoidalNoise`
        List of individual noise sources that add to the environment of the spin system.
    amplitude : :class:`numpy.ndarray` of :class:`numpy.double` (timeIndex)
        Time series description of the signal. In units of Hz. Only evaluated when :func:`getAmplitude()` is called. See :math:`r(t)` above.
    frequency : :class:`numpy.ndarray` of :class:`numpy.double` (frequencyIndex)
        List of frequencies corresponding to the Fourier transform of the pulse sequence. In units of Hz.
    frequencyAmplitude : :class:`numpy.ndarray` of :class:`numpy.double` (frequencyIndex)
        The sine Fourier transform of the pulse sequence. In units of Hz. Only evaluated when :func:`getFrequencyAmplitude()` is called. See :math:`s(f)` above.
    """
    def __init__(self, neuralPulses = [NeuralPulse()], sinusoidalNoises = [], timeProperties = TimeProperties(), doEvaluate = True):
        """
        Parameters
        ----------
        neuralPulses : `list` of :class:`NeuralPulse`, optional
            List of individual neural pulses that make up the signal. Defaults to a single basic neural pulse.
        sinusoidalNoises : `list` of :class:`SinusoidalNoise`, optional
            List of individual noise sources that add to the environment of the spin system. Defaults to no noise.
        timeProperties : :class:`TimeProperties`, optional
            Details describing time for the signal.
        doEvaluate : `boolean`
            If `True`, calls functions :func:`getAmplitude()` and :func:`getFrequencyAmplitude()` to evaluate `amplitude` and `frequencyAmplitude`. Otherwise leaves these arrays empty, saving computational time. Defaults to `True`.
        """
        self.timeProperties = timeProperties
        self.neuralPulses = neuralPulses
        self.sinusoidalNoises = sinusoidalNoises
        self.amplitude = np.empty_like(timeProperties.timeCoarse, dtype = np.double)
        self.frequency = np.empty_like(timeProperties.timeCoarse, dtype = np.double)
        self.frequencyAmplitude = np.empty_like(self.frequency)     # The sine Fourier transform of the pulse sequence (Hz) [frequency index]
        if doEvaluate:
            self.getAmplitude()
            self.getFrequencyAmplitude()

    def getAmplitude(self):
        """
        Evaluates the timeseries representation of the neural pulse sequence, filling in the results to :attr:`amplitude`. Uses the numba cuda kernel :func:`testSignal.getAmplitude` for evaluation.
        """
        # Unwrap the neural pulse objects
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
        getAmplitude[blocksPerGrid, threadsPerBlock](self.timeProperties.timeCoarse, self.timeProperties.timeEndPoints, self.timeProperties.timeStepCoarse, timeStart, amplitude, frequency, self.amplitude)

    def getFrequencyAmplitude(self):
        """
        Takes a sine Fourier transform of the pulse sequence using a dot product, filling in the results to :attr:`frequencyAmplitude`. See the description in :class:`TestSignal`. Uses the numba cuda kernel :func:`testSignal.getFrequencyAmplitude` for evaluation.
        """
        # GPU control variables
        threadsPerBlock = 128
        blocksPerGrid = (self.timeProperties.timeIndexMax + (threadsPerBlock - 1)) // threadsPerBlock
        # Run GPU code
        getFrequencyAmplitude[blocksPerGrid, threadsPerBlock](self.timeProperties.timeEndPoints, self.timeProperties.timeCoarse, self.timeProperties.timeStepCoarse, self.amplitude, self.frequency, self.frequencyAmplitude)

    def plotFrequencyAmplitude(self, archive = None):
        """
        Plots the sine Fourier transform of the pulse sequence.

        Parameters
        ----------
        archive : :class:`archive.Archive`, optional
            If specified, saves the plots in `pdf` and `png` format to the path specified by :attr:`Archive.plotPath`.
        """
        plt.plot(self.frequency, self.frequencyAmplitude, "+--")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude (Hz)")
        plt.xlim([0, 2000])
        plt.grid()
        if archive:
            plt.title(archive.executionTimeString + "Fourier Transform Frequency Amplitude")
            plt.savefig(archive.plotPath + "fourierTransformFrequencyAmplitude.pdf")
            plt.savefig(archive.plotPath + "fourierTransformFrequencyAmplitude.png")
        plt.show()
        
    def writeToFile(self, archive):
        """
        Writes the pulse sequence description to a hdf5 file.

        Parameters
        ----------
        archive : :class:`h5py.Group`
            The HDF5 archive to write to.
        """
        archiveGroup = archive.require_group("testSignal")
        archiveGroup["amplitude"] = self.amplitude
        archiveGroup["frequency"] = self.frequency
        archiveGroup["frequencyAmplitude"] = self.frequencyAmplitude

        self.timeProperties.writeToFile(archiveGroup)
        
        archiveGroupNeuralPulsesGroup = archiveGroup.require_group("neuralPulses")
        for neuralPulseIndex, neuralPulse in enumerate(self.neuralPulses):
            archiveGroupNeuralPulsesGroupIndex = archiveGroupNeuralPulsesGroup.require_group(str(neuralPulseIndex))
            neuralPulse.writeToFile(archiveGroupNeuralPulsesGroupIndex)
        archiveGroupSinusoidalNoiseGroup = archiveGroup.require_group("sinusoidalNoise")

        for sinusoidalNoiseIndex, sinusoidalNoise in enumerate(self.sinusoidalNoises):
            archiveGroupSinusoidalNoiseGroupIndex = archiveGroupSinusoidalNoiseGroup.require_group(str(sinusoidalNoiseIndex))
            sinusoidalNoise.writeToFile(archiveGroupSinusoidalNoiseGroupIndex)

@cuda.jit(debug = cudaDebug)
def getAmplitude(
    timeCoarse, timeEndPoints, timeStepCoarse,
    timeStart, amplitude, frequency,
    timeAmplitude):
    """
    Evaluates the timeseries for a pulse sequence.

    Parameters
    ----------
    timeCoarse : :class:`numpy.ndarray` of :class:`numpy.double` (timeIndex)
        A coarse grained list of time samples that the signal is defined for. In units of s.
    timeEndPoints : :class:`numpy.ndarray` of :class:`numpy.double` (start time (0) or end time (1))
        The time values for when the experiment is to start and finishes. In units of s.
    timeStepCoarse : `float`
        The time difference between each element of `timeCoarse`. In units of s. Determines the sample rate of the outputs `timeCoarse`.
    timeStart : :class:`numpy.ndarray` of :class:`numpy.double` (neuralPulseIndex)
        The point in time where the neural pulse `neuralPulseIndex` first becomes non-zero. In units of s.
    amplitude : :class:`numpy.ndarray` of :class:`numpy.double` (neuralPulseIndex)
        The peak amplitude of the neural pulse `neuralPulseIndex`. In units of Hz.
    frequency : :class:`numpy.ndarray` of :class:`numpy.double` (neuralPulseIndex)
        The mean frequency of the neural pulse `neuralPulseIndex`. In units of Hz.
    timeAmplitude : :class:`numpy.ndarray` of :class:`numpy.double` (timeIndex)
        Time series description of the signal. In units of Hz. This is an output, so use an empty :class:`numpy.ndarray` with :func:`numpy.empty()`, or declare a :class:`numba.cuda.cudadrv.devicearray.DeviceNDArray` using :func:`numba.cuda.device_array_like()`.
    """
    timeIndex = cuda.threadIdx.x + cuda.blockIdx.x*cuda.blockDim.x  # The time array index the current thread is working on
    if timeIndex < timeCoarse.size:                                 # Check we haven't overflowed the array
        # Evaluate time
        timeCoarse[timeIndex] = timeIndex*timeStepCoarse

        # Evaluate signal
        timeAmplitude[timeIndex] = 0
        for neuralPulseIndex in range(timeStart.size):
            if timeCoarse[timeIndex] > timeStart[neuralPulseIndex] and timeCoarse[timeIndex] < timeStart[neuralPulseIndex] + 1/frequency[neuralPulseIndex]:
                timeAmplitude[timeIndex] += amplitude[neuralPulseIndex]*math.sin(2*math.pi*frequency[neuralPulseIndex]*(timeCoarse[timeIndex] - timeStart[neuralPulseIndex]))

@cuda.jit(debug = cudaDebug, max_registers = 31)
def getFrequencyAmplitude(timeEndPoints, timeCoarse, timeStepCoarse, timeAmplitude, frequency, frequencyAmplitude):
    """
    Takes the sine Fourier transform of the pulse sequence, given a time series.

    Parameters
    ----------
    timeCoarse : :class:`numpy.ndarray` of :class:`numpy.double` (timeIndex)
        A coarse grained list of time samples that the signal is defined for. In units of s.
    timeEndPoints : :class:`numpy.ndarray` of :class:`numpy.double` (start time (0) or end time (1))
        The time values for when the experiment is to start and finishes. In units of s.
    timeStepCoarse : `float`
        The time difference between each element of `timeCoarse`. In units of s. Determines the sample rate of the outputs `timeCoarse`.
    timeAmplitude : :class:`numpy.ndarray` of :class:`numpy.double` (timeIndex)
        Time series description of the signal. In units of Hz.
    frequency : :class:`numpy.ndarray` of :class:`numpy.double` (frequencyIndex)
        List of frequencies corresponding to the Fourier transform of the pulse sequence. In units of Hz. This is an output, so use an empty :class:`numpy.ndarray` with :func:`numpy.empty()`, or declare a :class:`numba.cuda.cudadrv.devicearray.DeviceNDArray` using :func:`numba.cuda.device_array_like()`
    frequencyAmplitude : :class:`numpy.ndarray` of :class:`numpy.double` (frequencyIndex)
        The sine Fourier transform of the pulse sequence. In units of Hz. This is an output, so use an empty :class:`numpy.ndarray` with :func:`numpy.empty()`, or declare a :class:`numba.cuda.cudadrv.devicearray.DeviceNDArray` using :func:`numba.cuda.device_array_like()`
    """

    # Max registers set to 31 => my GPU is at full capacity. Might be worth removing or increasing if too slow.

    frequencyIndex = cuda.threadIdx.x + cuda.blockIdx.x*cuda.blockDim.x # The frequency array index the current thread is working on
    if frequencyIndex < frequencyAmplitude.size:                        # Check we haven't overflowed the array
        frequencyStep = 0.125/(timeEndPoints[1] - timeEndPoints[0])
        frequency[frequencyIndex] = frequencyStep*frequencyIndex        # The frequency this thread is calculating the coefficient for
        frequencyAmplitudeTemporary = 0                                 # It's apparently more optimal to repeatedly write to a register (this) than the output array in memory. Though honestly can't see the improvement
        frequencyTemporary = frequency[frequencyIndex]                  # And read from a register rather than memory
        for timeIndex in nb.prange(timeCoarse.size):
            frequencyAmplitudeTemporary += timeAmplitude[timeIndex]*math.sin(2*math.pi*timeCoarse[timeIndex]*frequencyTemporary)    # Dot product
        frequencyAmplitude[frequencyIndex] = frequencyAmplitudeTemporary*timeStepCoarse/(timeEndPoints[1] - timeEndPoints[0])       # Scale to make an integral
