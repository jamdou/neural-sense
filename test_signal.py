"""
Classes to store information about magnetic signals being simulated and reconstructed, as well as timing.
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from numba import cuda
import numba as nb

cuda_debug = False

class TimeProperties:
    """
    Grouped details about time needed for simulations and reconstructions.

    Attributes
    ----------
    time_step_coarse : `float`
        The time difference between each element of `time_coarse`. In units of s. Determines the sample rate of the outputs `time_coarse` and `time_evolution_coarse`.
    time_step_fine : `float`
        The time step used within the integration algorithm. In units of s.
    time_end_points : :class:`numpy.ndarray` of :class:`numpy.double` (start time (0) or end time (1))
        The time values for when the experiment is to start and finishes. In units of s.
    time_index_max : :obj:`int`
        The number of times to be sampled. That is, the size of `time_coarse`.
    time_coarse : :class:`numpy.ndarray` of :class:`numpy.double` (time_index)
        A coarse grained list of time samples that the signal is defined for. In units of s.
    """
    def __init__(self, time_step_coarse = 5e-7, time_step_fine = 1e-7, time_step_source = 1e-7, time_end_points = [0, 1e-1]):
        """
        Parameters
        ----------
        time_step_coarse : `float`, optional
            The time difference between each element of `time_coarse`. In units of s. Determines the sample rate of the outputs `time_coarse` and `time_evolution_coarse`. Defaults to 500ns.
        time_step_fine : `float`, optional
            The time step used within the integration algorithm. In units of s. Defaults to 100ns.
        time_end_points : :class:`numpy.ndarray` of :class:`numpy.double` (start time (0) or end time (1))
            The time values for when the experiment is to start and finishes. In units of s. Defaults to between 0s and 100ms.
        """
        self.time_step_coarse = time_step_coarse
        self.time_step_fine = time_step_fine
        self.time_step_source = time_step_source
        self.time_end_points = np.asarray(time_end_points, np.double)
        self.time_index_max = int((self.time_end_points[1] - self.time_end_points[0])/self.time_step_coarse)
        self.time_coarse = np.empty(self.time_index_max)
        # self.time_source = np.arange(self.time_end_points[0], self.time_end_points[1], self.time_step_source, np.double)

    def write_to_file(self, archive):
        """
        Saves the time properties to a hdf5 file.

        Parameters
        ----------
        archive : :class:`h5py.Group`
            The HDF5 archive to write to.
        """
        archive["time_properties"] = self.time_coarse
        archive["time_properties"].attrs["time_step_coarse"] = self.time_step_coarse
        archive["time_properties"].attrs["time_step_fine"] = self.time_step_fine
        archive["time_properties"].attrs["time_end_points"] = self.time_end_points
        archive["time_properties"].attrs["time_index_max"] = self.time_index_max

    def read_from_file(self, archive):
        """
        Loads the time properties from a hdf5 file
        """
        self.time_coarse = archive["time_properties"]
        self.time_step_coarse = archive["time_properties"].attrs["time_step_coarse"]
        self.time_step_fine = archive["time_properties"].attrs["time_step_fine"]
        self.time_end_points = archive["time_properties"].attrs["time_end_points"]
        self.time_index_max = archive["time_properties"].attrs["time_index_max"]

class NeuralPulse:
    """
    A simple parametrised description of a neural magnetic pulse, being approximated as a half sine pulse.

    Attributes
    ----------
    time_start : `float`
        The point in time where the neural pulse first becomes non-zero. In units of s.
    amplitude : `float`
        The peak amplitude of the neural pulse. In units of Hz.
    frequency : `float`
        The mean frequency of the neural pulse. In units of Hz.
    """
    def __init__(self, time_start = 0.0, amplitude = 10.0, frequency = 1e3):
        """
        Parameters
        ----------
        time_start : `float`, optional
            The point in time where the neural pulse first becomes non-zero. In units of s. Defaults to 0.
        amplitude : `float`, optional
            The peak amplitude of the neural pulse. In units of Hz. Defaults to 10Hz.
        frequency : `float`, optional
            The mean frequency of the neural pulse. In units of Hz. Defaults to 1k_hz.
        """
        self.time_start = time_start
        self.amplitude = amplitude
        self.frequency = frequency

    def write_to_file(self, archive):
        """
        Writes the pulse details to a hdf5 file.

        Parameters
        ----------
        archive : :class:`h5py.Group`
            The HDF5 archive to write to.
        """
        archive["time_start"] = self.time_start
        archive["amplitude"] = self.amplitude
        archive["frequency"] = self.frequency

class SinusoidalNoise:
    """
    For describing 50Hz and detuning noise.

    Attributes
    ----------
    amplitude : :class:`numpy.ndarray` of :class:`numpy.double`, (spatial_index)
        The amplitude of each spatial component of the noise as a 3D vector. In units of Hz.
    frequency : :class:`numpy.ndarray` of :class:`numpy.double`, (spatial_index)
        The frequency of each spatial component of the noise as a 3D vector. In units of Hz.
    phase : :class:`numpy.ndarray` of :class:`numpy.double`, (spatial_index)
        The frequency of each spatial component of the noise as a 3D vector. In units of rad.
    type : :obj:`str`
        A label for the kind of noise this represents. For archival purposes.
    """
    def __init__(self, amplitude = [0, 0, 0], frequency = [50, 50, 50], phase = [math.pi/2, math.pi/2, math.pi/2], type = "SinusoidalNoise"):
        """
        Parameters
        ----------
        amplitude : :class:`numpy.ndarray` of :class:`numpy.double`, (spatial_index)
        The amplitude of each spatial component of the noise as a 3D vector. In units of Hz. Defaults to 0Hz in each direction.
        frequency : :class:`numpy.ndarray` of :class:`numpy.double`, (spatial_index)
            The frequency of each spatial component of the noise as a 3D vector. In units of Hz. Defaults to 50Hz in each direction.
        phase : :class:`numpy.ndarray` of :class:`numpy.double`, (spatial_index)
            The frequency of each spatial component of the noise as a 3D vector. In units of rad. Defaults to :math:`\\frac{\\pi}{2}` rad in each direction.
        type : :obj:`str`, optional
        A label for the kind of noise this represents. For archival purposes. Defaults to `"SinusoidalNoise"`.
        """
        self.amplitude = np.asarray(amplitude, dtype=np.double)
        self.frequency = np.asarray(frequency, dtype=np.double)

        self.phase = np.asarray(phase, dtype=np.double)
        self.type = type

    @staticmethod
    def new_detuning_noise(z_amplitude = 1.0):
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
        z_amplitude : `float`, optional
            The amplitude of the detuning. See :math:`d` above. Measured in Hz. Defaults to 1Hz.

        Returns
        -------
        sinusoidal_noise : :class:`SinusoidalNoise`
            A parametrisation of the detuning noise.
        """
        return SinusoidalNoise([0, 0, -z_amplitude], [0, 0, 0], [math.pi/2, math.pi/2, math.pi/2], "Detuning_noise")

    @staticmethod
    def new_line_noise(amplitude = [100.0, 0.0, 0.0], phase = [0.0, 0.0, 0.0]):
        """
        A shorthand constructor for line noise.

        .. note::
            Uses the Australian standard of 50Hz line noise.

        Parameters
        ----------
        amplitude : :class:`numpy.ndarray` of :class:`numpy.double`, (spatial_index)
            The amplitude of each spatial component of the noise as a 3D vector. In units of Hz. Defaults to 100Hz in the x direction only.
        phase : :class:`numpy.ndarray` of :class:`numpy.double`, (spatial_index)
            The frequency of each spatial component of the noise as a 3D vector. In units of rad. Defaults to 0rad in each direction.

        Returns
        -------
        sinusoidal_noise : :class:`SinusoidalNoise`
            A parametrisation of the line noise.
        """
        return SinusoidalNoise(amplitude, [50, 50, 50], phase, "Line_noise")

    def write_to_file(self, archive):
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
    time_properties : :class:`TimeProperties`
        Details describing time for the signal.
    neural_pulses : `list` of :class:`NeuralPulse`
        List of individual neural pulses that make up the signal.
    sinusoidal_noises : `list` of :class:`SinusoidalNoise`
        List of individual noise sources that add to the environment of the spin system.
    amplitude : :class:`numpy.ndarray` of :class:`numpy.double` (time_index)
        Time series description of the signal. In units of Hz. Only evaluated when :func:`get_amplitude()` is called. See :math:`r(t)` above.
    frequency : :class:`numpy.ndarray` of :class:`numpy.double` (frequency_index)
        List of frequencies corresponding to the Fourier transform of the pulse sequence. In units of Hz.
    frequency_amplitude : :class:`numpy.ndarray` of :class:`numpy.double` (frequency_index)
        The sine Fourier transform of the pulse sequence. In units of Hz. Only evaluated when :func:`get_frequency_amplitude()` is called. See :math:`s(f)` above.
    """
    def __init__(self, neural_pulses = [NeuralPulse()], sinusoidal_noises = [], time_properties = TimeProperties(), do_evaluate = True):
        """
        Parameters
        ----------
        neural_pulses : `list` of :class:`NeuralPulse`, optional
            List of individual neural pulses that make up the signal. Defaults to a single basic neural pulse.
        sinusoidal_noises : `list` of :class:`SinusoidalNoise`, optional
            List of individual noise sources that add to the environment of the spin system. Defaults to no noise.
        time_properties : :class:`TimeProperties`, optional
            Details describing time for the signal.
        do_evaluate : :obj:`bool`
            If :obj:`True`, calls functions :func:`get_amplitude()` and :func:`get_frequency_amplitude()` to evaluate `amplitude` and `frequency_amplitude`. Otherwise leaves these arrays empty, saving computational time. Defaults to :obj:`True`.
        """
        self.time_properties = time_properties
        self.neural_pulses = neural_pulses
        self.sinusoidal_noises = sinusoidal_noises
        self.amplitude = np.empty_like(time_properties.time_coarse, dtype = np.double)
        self.frequency = np.empty_like(time_properties.time_coarse, dtype = np.double)
        self.frequency_amplitude = np.empty_like(self.frequency)     # The sine Fourier transform of the pulse sequence (Hz) [frequency index]
        if do_evaluate:
            self.get_amplitude()
            self.get_frequency_amplitude()

    def get_amplitude(self):
        """
        Evaluates the timeseries representation of the neural pulse sequence, filling in the results to :attr:`amplitude`. Uses the numba cuda kernel :func:`test_signal.get_amplitude` for evaluation.
        """
        # Unwrap the neural pulse objects
        time_start = []
        amplitude = []
        frequency = []
        for neural_pulse in self.neural_pulses:
            time_start += [neural_pulse.time_start]
            amplitude += [neural_pulse.amplitude]
            frequency += [neural_pulse.frequency]
        time_start = np.asarray(time_start)
        amplitude = np.asarray(amplitude)
        frequency = np.asarray(frequency)

        # GPU control variables
        threads_per_block = 128
        blocks_per_grid = (self.time_properties.time_index_max + (threads_per_block - 1)) // threads_per_block
        # Run GPU code
        get_amplitude[blocks_per_grid, threads_per_block](self.time_properties.time_coarse, self.time_properties.time_end_points, self.time_properties.time_step_coarse, time_start, amplitude, frequency, self.amplitude)

    def get_frequency_amplitude(self):
        """
        Takes a sine Fourier transform of the pulse sequence using a dot product, filling in the results to :attr:`frequency_amplitude`. See the description in :class:`TestSignal`. Uses the numba cuda kernel :func:`test_signal.get_frequency_amplitude` for evaluation.
        """
        # GPU control variables
        threads_per_block = 128
        blocks_per_grid = (self.time_properties.time_index_max + (threads_per_block - 1)) // threads_per_block
        # Run GPU code
        get_frequency_amplitude[blocks_per_grid, threads_per_block](self.time_properties.time_end_points, self.time_properties.time_coarse, self.time_properties.time_step_coarse, self.amplitude, self.frequency, self.frequency_amplitude)

    def plot_frequency_amplitude(self, archive = None):
        """
        Plots the sine Fourier transform of the pulse sequence.

        Parameters
        ----------
        archive : :class:`archive.Archive`, optional
            If specified, saves the plots in `pdf` and `png` format to the path specified by :attr:`Archive.plot_path`.
        """
        plt.plot(self.frequency, self.frequency_amplitude, "+--")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude (Hz)")
        plt.xlim([0, 2000])
        plt.grid()
        if archive:
            plt.title(archive.execution_time_string + "Fourier Transform Frequency Amplitude")
            plt.savefig(archive.plot_path + "fourierTransform_frequency_amplitude.pdf")
            plt.savefig(archive.plot_path + "fourierTransform_frequency_amplitude.png")
        plt.show()
        
    def write_to_file(self, archive):
        """
        Writes the pulse sequence description to a hdf5 file.

        Parameters
        ----------
        archive : :class:`h5py.Group`
            The HDF5 archive to write to.
        """
        archive_group = archive.require_group("test_signal")
        archive_group["amplitude"] = self.amplitude
        archive_group["frequency"] = self.frequency
        archive_group["frequency_amplitude"] = self.frequency_amplitude

        self.time_properties.write_to_file(archive_group)
        
        archive_groupNeural_pulses_group = archive_group.require_group("neural_pulses")
        for neural_pulse_index, neural_pulse in enumerate(self.neural_pulses):
            archive_groupNeuralPulses_group_index = archive_groupNeural_pulses_group.require_group(str(neural_pulse_index))
            neural_pulse.write_to_file(archive_groupNeuralPulses_group_index)
        archive_groupSinusoidalNoise_group = archive_group.require_group("sinusoidal_noise")

        for sinusoidal_noise_index, sinusoidal_noise in enumerate(self.sinusoidal_noises):
            archive_groupSinusoidalNoise_group_index = archive_groupSinusoidalNoise_group.require_group(str(sinusoidal_noise_index))
            sinusoidal_noise.write_to_file(archive_groupSinusoidalNoise_group_index)

@cuda.jit(debug = cuda_debug)
def get_amplitude(
    time_coarse, time_end_points, time_step_coarse,
    time_start, amplitude, frequency,
    time_amplitude):
    """
    Evaluates the timeseries for a pulse sequence.

    Parameters
    ----------
    time_coarse : :class:`numpy.ndarray` of :class:`numpy.double` (time_index)
        A coarse grained list of time samples that the signal is defined for. In units of s.
    time_end_points : :class:`numpy.ndarray` of :class:`numpy.double` (start time (0) or end time (1))
        The time values for when the experiment is to start and finishes. In units of s.
    time_step_coarse : `float`
        The time difference between each element of `time_coarse`. In units of s. Determines the sample rate of the outputs `time_coarse`.
    time_start : :class:`numpy.ndarray` of :class:`numpy.double` (neural_pulse_index)
        The point in time where the neural pulse `neural_pulse_index` first becomes non-zero. In units of s.
    amplitude : :class:`numpy.ndarray` of :class:`numpy.double` (neural_pulse_index)
        The peak amplitude of the neural pulse `neural_pulse_index`. In units of Hz.
    frequency : :class:`numpy.ndarray` of :class:`numpy.double` (neural_pulse_index)
        The mean frequency of the neural pulse `neural_pulse_index`. In units of Hz.
    time_amplitude : :class:`numpy.ndarray` of :class:`numpy.double` (time_index)
        Time series description of the signal. In units of Hz. This is an output, so use an empty :class:`numpy.ndarray` with :func:`numpy.empty()`, or declare a :class:`numba.cuda.cudadrv.devicearray.DeviceNDArray` using :func:`numba.cuda.device_array_like()`.
    """
    time_index = cuda.threadIdx.x + cuda.blockIdx.x*cuda.blockDim.x  # The time array index the current thread is working on
    if time_index < time_coarse.size:                                 # Check we haven't overflowed the array
        # Evaluate time
        time_coarse[time_index] = time_index*time_step_coarse

        # Evaluate signal
        time_amplitude[time_index] = 0
        for neural_pulse_index in range(time_start.size):
            if time_coarse[time_index] > time_start[neural_pulse_index] and time_coarse[time_index] < time_start[neural_pulse_index] + 1/frequency[neural_pulse_index]:
                time_amplitude[time_index] += amplitude[neural_pulse_index]*math.sin(2*math.pi*frequency[neural_pulse_index]*(time_coarse[time_index] - time_start[neural_pulse_index]))

@cuda.jit(debug = cuda_debug, max_registers = 31)
def get_frequency_amplitude(time_end_points, time_coarse, time_step_coarse, time_amplitude, frequency, frequency_amplitude):
    """
    Takes the sine Fourier transform of the pulse sequence, given a time series.

    Parameters
    ----------
    time_coarse : :class:`numpy.ndarray` of :class:`numpy.double` (time_index)
        A coarse grained list of time samples that the signal is defined for. In units of s.
    time_end_points : :class:`numpy.ndarray` of :class:`numpy.double` (start time (0) or end time (1))
        The time values for when the experiment is to start and finishes. In units of s.
    time_step_coarse : `float`
        The time difference between each element of `time_coarse`. In units of s. Determines the sample rate of the outputs `time_coarse`.
    time_amplitude : :class:`numpy.ndarray` of :class:`numpy.double` (time_index)
        Time series description of the signal. In units of Hz.
    frequency : :class:`numpy.ndarray` of :class:`numpy.double` (frequency_index)
        List of frequencies corresponding to the Fourier transform of the pulse sequence. In units of Hz. This is an output, so use an empty :class:`numpy.ndarray` with :func:`numpy.empty()`, or declare a :class:`numba.cuda.cudadrv.devicearray.DeviceNDArray` using :func:`numba.cuda.device_array_like()`
    frequency_amplitude : :class:`numpy.ndarray` of :class:`numpy.double` (frequency_index)
        The sine Fourier transform of the pulse sequence. In units of Hz. This is an output, so use an empty :class:`numpy.ndarray` with :func:`numpy.empty()`, or declare a :class:`numba.cuda.cudadrv.devicearray.DeviceNDArray` using :func:`numba.cuda.device_array_like()`
    """

    # Max registers set to 31 => my GPU is at full capacity. Might be worth removing or increasing if too slow.

    frequency_index = cuda.threadIdx.x + cuda.blockIdx.x*cuda.blockDim.x # The frequency array index the current thread is working on
    if frequency_index < frequency_amplitude.size:                        # Check we haven't overflowed the array
        frequency_step = 0.125/(time_end_points[1] - time_end_points[0])
        frequency[frequency_index] = frequency_step*frequency_index        # The frequency this thread is calculating the coefficient for
        frequency_amplitude_temporary = 0                                 # It's apparently more optimal to repeatedly write to a register (this) than the output array in memory. Though honestly can't see the improvement
        frequency_temporary = frequency[frequency_index]                  # And read from a register rather than memory
        for time_index in nb.prange(time_coarse.size):
            frequency_amplitude_temporary += time_amplitude[time_index]*math.sin(2*math.pi*time_coarse[time_index]*frequency_temporary)    # Dot product
        frequency_amplitude[frequency_index] = frequency_amplitude_temporary*time_step_coarse/(time_end_points[1] - time_end_points[0])       # Scale to make an integral
