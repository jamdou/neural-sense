"""
Classes and methods used to control the specific experiment we are trying to simulate.
"""

# from spinsim import utilities
import spinsim

import numpy as np
import matplotlib.pyplot as plt
import math, cmath
from numba import cuda
import numba as nb
import time as tm
import os
from test_signal import *
import enum

# from .benchmark.results import *

class MeasurementMethod(enum.Enum):
    FARADAY_DEMODULATION = "faraday demodulation"
    HARD_PULSE = "hard pulse"
    ADIABATIC_DEMAPPING = "adiabatic demapping"

class SimulationManager:
    """
    Controls a set of simulations running for different dressing parameters.

    Attributes
    ----------
    signal : :obj:`list` of :class:`test_signal.TestSignal`
        A list of a signal objects containing information describing the magnetic environment of the simulation, as well as timing information for the simulation. :func:`evaluate()` will run simulations for all of these values.
    trotter_cutoff : :obj:`list` of :obj:`int`
        A list of the number of squares to be used in the matrix exponentiation algorithm :func:`simulation_utilities.matrix_exponential_lie_trotter()` during the simulation. :func:`evaluate()` will run simulations for all of these values.
    frequency : :class:`numpy.ndarray` of :class:`numpy.float64`, (frequency_index)
        A list of dressing rabi frequencies for the spin system. In units of Hz. :func:`evaluate()` will run simulations for all of these values.
    frequency_amplitude : :class:`numpy.ndarray` of :class:`numpy.float64`, (simulation_index)
        The Fourier coefficient of the signal as measured from the simulation `simulation_index`. Filled in after the simulation `simulation_index` has been completed.
    archive : :class:`archive.Archive`
        The archive object to save the simulation results and parameters to.
    state_output : :obj:`list` of (`numpy.ndarray` of  `numpy.complex128`, (time_index, state_index))
        An optional :obj:`list` to directly write the state of the simulation to. Used for benchmarks. Reference can be optionally passed in on construction.
    state_properties : :class:`StateProperties`
            The :class:`StateProperties` initial conditions for the wavefunction of the quantum system.
    device : :obj:`spinsim.Device`
        The target device (hardware) that the simulation is to be run on.
    execution_time_output : :obj:`list` of :obj:`float`
        If not :obj:`None`, the execution times for each full simulation will be appended to this.
    measurement_method : :obj:`MeasurementMethod`
        Method used to retrieve frequency amplitudes.
    """
    def __init__(self, signal, frequency, archive, state_properties = None, state_output = None, trotter_cutoff = [28], device = None, execution_time_output = None, measurement_method = MeasurementMethod.HARD_PULSE):
        """
        Parameters
        ----------
        signal : :class:`test_signal.TestSignal` or :obj:`list` of :class:`test_signal.TestSignal`
            A list of a signal objects containing information describing the magnetic environment of the simulation, as well as timing information for the simulation. :func:`evaluate()` will run simulations for all of these values.
        frequency : :class:`numpy.ndarray` of :class:`numpy.float64`, (frequency_index)
            A list of dressing rabi frequencies for the spin system. In units of Hz. :func:`evaluate()` will run simulations for all of these values.
        archive : :class:`archive.Archive`
            The archive object to save the simulation results and parameters to.
        state_properties : :class:`StateProperties`
            The :class:`StateProperties` initial conditions for the wavefunction of the quantum system.
        state_output : :obj:`list` of (`numpy.ndarray` of  `numpy.complex128`, (time_index, state_index)), optional
            An optional :obj:`list` to directly write the state of the simulation to. Used for benchmarks.
        trotter_cutoff : :obj:`list` of :obj:`int`, optional
            A list of the number of squares to be used in the matrix exponentiation algorithm during the simulation. :func:`evaluate()` will run simulations for all of these values.
        device : :obj:`spinsim.Device`
            The target device (hardware) that the simulation is to be run on.
        execution_time_output : :obj:`list` of :obj:`float`
            If not :obj:`None`, the execution times for each full simulation will be appended to this.
        measurement_method : :obj:`MeasurementMethod`
            Method used to retrieve frequency amplitudes.
        """
        self.signal = signal
        if not isinstance(self.signal, list):
            self.signal = [self.signal]
        self.device = device
        if not isinstance(self.device, list):
            self.device = [self.device]
        self.trotter_cutoff = np.asarray(trotter_cutoff)
        self.frequency = np.asarray(frequency, dtype = np.float64)
        self.frequency_amplitude = np.empty(self.frequency.size*len(self.signal)*self.trotter_cutoff.size*len(self.device), dtype = np.float64)
        self.archive = archive
        self.state_output = state_output
        self.state_properties = state_properties
        self.execution_time_output = execution_time_output
        self.measurement_method = measurement_method

    def evaluate(self, do_plot = False, do_write_everything = False):
        """
        Evaluates the prepared set of simulations. Fills out the :class:`numpy.ndarray`, :attr:`frequency_amplitude`. The simulation `simulation_index` will be run with the frequency given by `frequency_index` mod :attr:`frequency.size`, the signal given by floor(`signal_index` / `frequency.size`) mod len(`signal`), and the trotter cutoff given by floor(`signal_index` / `frequency.size` / `trotter_cutoff.size`).
        Parameters
        ----------
        do_plot : :obj:`bool`, optional
            If :obj:`True`, plots time series of the expected spin values in each direction during execution.
        do_write_everything : :obj:`bool`, optional
            If :obj:`True`, then save all time series data to file as well as parametric data. Defaults to :obj:`False` to reduce archive file size.
        """
        print("\033[33mStarting simulations...\033[0m")
        execution_time_end_points = np.empty(2)
        execution_time_end_points[0] = tm.time()
        execution_time_end_points[1] = execution_time_end_points[0]
        print("Idx\tCmp\tTm\tdTm")
        archive_group_simulations = self.archive.archive_file.require_group("simulations")
        for device_index, device_instance in enumerate(self.device):
            for trotter_cutoff_index, trotter_cutoff_instance in enumerate(self.trotter_cutoff):
                for signal_index, signal_instance in enumerate(self.signal):
                    simulation = Simulation(signal_instance, self.frequency[0], self.state_properties, trotter_cutoff_instance, device_instance, measurement_method = self.measurement_method)
                    for frequency_index in range(self.frequency.size):
                        simulation_index = frequency_index + (signal_index + (trotter_cutoff_index + device_index*self.trotter_cutoff.size)*len(self.signal))*self.frequency.size
                        frequency_value = self.frequency[frequency_index]
                        simulation.evaluate(frequency_value)
                        if self.measurement_method == MeasurementMethod.FARADAY_DEMODULATION:
                            simulation.get_frequency_amplitude_from_demodulation([0.9*signal_instance.time_properties.time_end_points[1], signal_instance.time_properties.time_end_points[1]], do_plot)
                        else:
                            simulation.get_frequency_amplitude_from_projection()
                            # plt.figure()
                            # plt.plot(simulation.simulation_results.spin)
                            # plt.show()
                        # simulation.get_frequency_amplitude_from_demodulation([0.9*signal_instance.time_properties.time_end_points[1], signal_instance.time_properties.time_end_points[1]], frequency_value == 1000, self.archive)
                        simulation.write_to_file(archive_group_simulations.require_group("simulation" + str(simulation_index)), do_write_everything)
                        self.frequency_amplitude[simulation_index] = simulation.simulation_results.sensed_frequency_amplitude
                        if self.state_output is not None:
                            self.state_output += [simulation.simulation_results.state.copy()]
                        if self.execution_time_output is not None:
                            self.execution_time_output += [tm.time() - execution_time_end_points[1]]
                        print("{:4d}\t{:3.0f}%\t{:3.0f}s\t{:2.3f}s".format(simulation_index, 100*(simulation_index + 1)/(self.frequency.size*len(self.signal)*self.trotter_cutoff.size*len(self.device)), tm.time() - execution_time_end_points[0], tm.time() - execution_time_end_points[1]))
                        execution_time_end_points[1] = tm.time()
        print("\033[32mDone!\033[0m")

#===============================================================#

# Important constants
cuda_debug = False
sqrt2 = math.sqrt(2)
sqrt3 = math.sqrt(3)
exp_precision = 5                                # Where to cut off the exp Taylor series
machine_epsilon = np.finfo(np.float64).eps*1000  # When to decide that vectors are parallel
# trotter_cutoff = 52

# interpolate = utilities.interpolate_source_cubic

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
    dressing_rabi_frequency : :obj:`float`
        The amplitude of the dressing in units of Hz.
    source_index_max : :obj:`int`
        The number of sources in the simulation.
    source_amplitude : :class:`numpy.ndarray` of :class:`numpy.float64`, (source_index, spatial_index)
        The amplitude of the sine wave of source `source_index` in direction `spatial_index`. See :math:`f_\\textrm{amp}` above. In units of Hz.
    source_phase : :class:`numpy.ndarray` of :class:`numpy.float64`, (source_index, spatial_index)
        The phase offset of the sine wave of source `source_index` in direction `spatial_index`. See :math:`\\phi` above. In units of radians.
    source_frequency : :class:`numpy.ndarray` of :class:`numpy.float64`, (source_index, spatial_index)
        The frequency of the sine wave of source `source_index` in direction `spatial_index`. See :math:`f` above. In units of Hz.
    source_time_end_points : :class:`numpy.ndarray` of :class:`numpy.float64`, (source_index, turn on time (0) or turn off time (1))
        The times that the sine wave of source `source_index` turns on and off. See :math:`\\tau` above. In units of s.
    source_quadratic_shift :  :obj:`float`
        The constant quadratic shift of the spin 1 system, in Hz.
    source_type : :class:`numpy.ndarray` of :class:`numpy.float64`, (source_index)
        A string description of what source source_index physically represents. Mainly for archive purposes.
    evaluate_dressing : :obj:`callable`
        The dressing function fed into the intergator.
    """
    def __init__(self, signal, state_properties, dressing_rabi_frequency = 1000.0, quadratic_shift = 0.0, measurement_method = MeasurementMethod.FARADAY_DEMODULATION):
        """
        Parameters
        ----------
        signal : :class:`test_signal.TestSignal`
            An object that contains high level descriptions of the signal to be measured, as well as noise.
        state_properties : :class:`StateProperties`
            An object that contains the initial state and spin quatum number of the system.
        dressing_rabi_frequency : :obj:`float`, optional
            The amplitude of the dressing in units of Hz.
        quadratic_shift :  :obj:`float`, optional
            The constant quadratic shift of the spin 1 system, in Hz.
        """
        self.dressing_rabi_frequency = dressing_rabi_frequency
        self.source_index_max = 0
        self.source_amplitude = np.empty([0, 3], np.float64)
        self.source_phase = np.empty([0, 3], np.float64)
        self.source_frequency = np.empty([0, 3], np.float64)
        self.source_time_end_points = np.empty([0, 2], np.float64)
        self.source_type = np.empty([0], object)
        self.source_quadratic_shift = quadratic_shift
        self.measurement_method = measurement_method

        # Construct the signal from the dressing information and pulse description.
        if signal:
            self.add_dressing(signal)
            for neural_pulse in signal.neural_pulses:
                self.add_neural_pulse(neural_pulse)
                # self.add_neural_pulse(neural_pulse.time_start, neural_pulse.amplitude, neural_pulse.frequency)
            for sinusoidal_noise in signal.sinusoidal_noises:
                self.add_sinusoidal_noise(sinusoidal_noise)

        self.evaluate_dressing = dressing_evaluator_factory(self.source_index_max, self.source_amplitude, self.source_frequency, self.source_phase, self.source_time_end_points, self.source_quadratic_shift, measurement_method)

        # timeSource_index_max = int((signal.time_properties.time_end_points[1] - signal.time_properties.time_end_points[0])/signal.time_properties.time_step_source)
        # self.source = cuda.device_array((timeSource_index_max, 4), dtype = np.float64)

        # threads_per_block = 64
        # blocks_per_grid = (timeSource_index_max + (threads_per_block - 1)) // threads_per_block
        # evaluate_dressing[blocks_per_grid, threads_per_block](signal.time_properties.time_step_source, self.source_index_max, cuda.to_device(self.source_amplitude), cuda.to_device(self.source_frequency), cuda.to_device(self.source_phase), cuda.to_device(self.source_time_end_points), self.source_quadratic_shift, self.source)

    def write_to_file(self, archive):
        """
        Saves source information to archive file.

        Parameters
        ----------
        archive : :class:`h5py.Group`
            The HDF5 archive to write to.
        """
        archive_group = archive.require_group("source_properties")
        archive_group["source_amplitude"] = self.source_amplitude
        archive_group["source_phase"] = self.source_phase
        archive_group["source_frequency"] = self.source_frequency
        archive_group["source_time_end_points"] = self.source_time_end_points
        archive_group["source_type"] = np.asarray(self.source_type, dtype='|S32')
        archive_group["source_index_max"] = np.asarray([self.source_index_max])
        archive_group["source_quadratic_shift"] = np.asarray([self.source_quadratic_shift])

    def add_dressing(self, signal, bias_amplitude = 460e3):
        """
        Adds a specified bias field and dressing field to the list of sources.

        Parameters
        ----------
        signal : :class:`test_signal.TestSignal`
            The signal object. Needed to specify how long the dressing and bias should be on for.
        bias_amplitude : :obj:`float`, optional
            The strength of the dc bias field in Hz. Also the frequency of the dressing.
            If one wants to add detuning, one can do that via detuning noise in :class:`test_signal.SinusoidalNoise.new_detuning_noise()`.
        """
        if self.measurement_method == MeasurementMethod.HARD_PULSE:
            # Initialise
            source_amplitude = np.zeros([3, 3])
            source_phase = np.zeros([3, 3])
            source_frequency = np.zeros([3, 3])
            source_time_end_points = np.zeros([3, 2])
            source_type = np.empty([3], dtype = object)

            # Label
            source_type[0] = "Dressing"
            source_type[1] = "Readout"
            source_type[2] = "Bias"

            # Bias
            source_amplitude[2, 2] = bias_amplitude
            source_phase[2, 2] = math.pi/2

            #Dressing
            readout_amplitude = 1e3
            dressing_end_buffer = (1/readout_amplitude)/4

            source_amplitude[0, 0] = 2*self.dressing_rabi_frequency
            source_frequency[0, 0] = bias_amplitude
            source_time_end_points[0, 0] = signal.time_properties.time_end_points[0]
            source_time_end_points[0, 1] = signal.time_properties.time_end_points[1] - (1/readout_amplitude)/4 - dressing_end_buffer
            source_phase[0, 0] = math.pi/2

            source_amplitude[1, 0] = 2*readout_amplitude
            source_frequency[1, 0] = bias_amplitude
            source_time_end_points[1, 0] = source_time_end_points[0, 1]
            source_time_end_points[1, 1] = signal.time_properties.time_end_points[1] - dressing_end_buffer
            source_phase[1, 0] = math.tau*math.fmod(0.5 + bias_amplitude*(source_time_end_points[1, 0]), 1)

            source_time_end_points[2, 0] = signal.time_properties.time_end_points[0]
            source_time_end_points[2, 1] = signal.time_properties.time_end_points[1]

            self.source_index_max += 3

        else:
            # Initialise
            source_amplitude = np.zeros([1, 3])
            source_phase = np.zeros([1, 3])
            source_frequency = np.zeros([1, 3])
            source_time_end_points = np.zeros([1, 2])
            source_type = np.empty([1], dtype = object)

            # Label
            source_type[0] = "Dressing"

            # Bias
            source_amplitude[0, 2] = bias_amplitude
            source_phase[0, 2] = math.pi/2

            #Dressing
            source_amplitude[0, 0] = 2*self.dressing_rabi_frequency
            source_frequency[0, 0] = bias_amplitude
            source_time_end_points[0, :] = 1*signal.time_properties.time_end_points
            source_phase[0, 0] = math.pi/2

            self.source_index_max += 1

        # Add
        self.source_amplitude = np.concatenate((self.source_amplitude, source_amplitude))
        self.source_phase = np.concatenate((self.source_phase, source_phase))
        self.source_frequency = np.concatenate((self.source_frequency, source_frequency))
        self.source_time_end_points = np.concatenate((self.source_time_end_points, source_time_end_points))
        self.source_type = np.concatenate((self.source_type, source_type))

    def add_neural_pulse(self, neural_pulse):
        """
        Adds a neural pulse signal to the list of sources from a :class:`test_signal.NeuralPulse` object.

        Parameters
        ----------
        neural_pulse : :class:`test_signal.NeuralPulse`
            An object parameterising the neural pulse signal to be added to the list of sources.
        """
        # Initialise
        source_amplitude = np.zeros([1, 3])
        source_phase = np.zeros([1, 3])
        source_frequency = np.zeros([1, 3])
        source_time_end_points = np.zeros([1, 2])
        source_type = np.empty([1], dtype = object)

        # Label
        source_type[0] = "NeuralPulse"

        # Pulse
        source_amplitude[0, 2] = neural_pulse.amplitude
        source_frequency[0, 2] = neural_pulse.frequency
        # source_phase[0, 2] = math.pi/2
        source_time_end_points[0, :] = np.asarray([neural_pulse.time_start, neural_pulse.time_start + 1/neural_pulse.frequency])

        # Add
        self.source_amplitude = np.concatenate((self.source_amplitude, source_amplitude))
        self.source_phase = np.concatenate((self.source_phase, source_phase))
        self.source_frequency = np.concatenate((self.source_frequency, source_frequency))
        self.source_time_end_points = np.concatenate((self.source_time_end_points, source_time_end_points))
        self.source_type = np.concatenate((self.source_type, source_type))
        self.source_index_max += 1

    def add_sinusoidal_noise(self, sinusoidal_noise):
        """
        Adds sinusoidal noise from a :class:`test_signal.SinusoidalNoise` object to the list of sources.

        Parameters
        ----------
        sinusoidal_noise : :class:`test_signal.SinusoidalNoise`
            The sinusoidal noise object to add to the list of sources.
        """
        # Initialise
        source_amplitude = np.zeros([1, 3])
        source_phase = np.zeros([1, 3])
        source_frequency = np.zeros([1, 3])
        source_time_end_points = np.zeros([1, 2])
        source_type = np.empty([1], dtype = object)

        # Label
        source_type[0] = sinusoidal_noise.type

        # Pulse
        source_amplitude[0, :] = sinusoidal_noise.amplitude
        source_frequency[0, :] = sinusoidal_noise.frequency
        source_phase[0, :] = sinusoidal_noise.phase
        source_time_end_points[0, :] = np.asarray([0.0, 1800.0])

        # Add
        self.source_amplitude = np.concatenate((self.source_amplitude, source_amplitude))
        self.source_phase = np.concatenate((self.source_phase, source_phase))
        self.source_frequency = np.concatenate((self.source_frequency, source_frequency))
        self.source_time_end_points = np.concatenate((self.source_time_end_points, source_time_end_points))
        self.source_type = np.concatenate((self.source_type, source_type))
        self.source_index_max += 1

class StateProperties:
    """
    The initial state fed into the simulation code.

    Attributes
    ----------
    spin_quantum_number: :class:`spinsim.SpinQuantumNumber(Enum)`
        The spin quantum number of the system being simulated. Determines the dimension of the hilbert space of the state, the algorithms used in the simulation, etc.
    state_init : :class:`numpy.ndarray` of :class:`numpy.complex128` (state_index)
        The state (spin wavefunction) of the system at the start of the simulation.
    """
    def __init__(self, spin_quantum_number = spinsim.SpinQuantumNumber.HALF, state_init = None):
        """
        Parameters
        ----------
        spin_quantum_number: :class:`spinsim.SpinQuantumNumber(Enum)`
            The spin quantum number of the system being simulated. Determines the dimension of the hilbert space of the state, the algorithms used in the simulation, etc.
        state_init : :class:`numpy.ndarray` of :class:`numpy.complex128`
            The state (spin wavefunction) of the system at the start of the simulation.
        """
        self.spin_quantum_number = spin_quantum_number
        if state_init:
            self.state_init = np.asarray(state_init, np.complex128)
        else:
            if self.spin_quantum_number == spinsim.SpinQuantumNumber.HALF:
                self.state_init = np.asarray([1, 0], np.complex128)
            else:
                self.state_init = np.asarray([1, 0, 0], np.complex128)

    def write_to_file(self, archive):
        """
        Parameters
        ----------
        archive : :class:`h5py.Group`
            The HDF5 archive to write to.
        """
        archive_group = archive.require_group("state_properties")
        archive_group["state_init"] = self.state_init
        archive_group["spin_quantum_number"] = np.asarray(self.spin_quantum_number.label, dtype='|S32')

class SimulationResults:
    """
    The output of the simulation code.

    Attributes
    ----------
    time_evolution : :class:`numpy.ndarray` of :class:`numpy.complex128` (time_index, bra_state_index, ket_state_index)
        Time evolution operator (matrix) between the current and next timesteps, for each time sampled.
    state : :class:`numpy.ndarray` of :class:`numpy.complex128` (time_index, state_index)
        The state (wavefunction) of the spin system in the lab frame, for each time sampled.
    spin : :class:`numpy.ndarray` of :class:`numpy.float64` (time_index, spatial_index)
        The expected value for hyperfine spin of the spin system in the lab frame, for each time sampled. Units of :math:`\\hbar`.
    sensed_frequency_amplitude : :obj:`float`
        The measured Fourier coefficient from the simulation.
    sensed_frequency_amplitude_method : :obj:`str`
        The method used to find the measured Fourier coefficient (for archival purposes).
    """
    def __init__(self, signal, state_properties):
        """
        Parameters
        ----------
        signal : :class:`test_signal.TestSignal`
            Defines the sampling time for the simulation results.
        state_properties : :class:`StateProperties`
            Defines the hilbert space dimension for the simulation results.
        """
        self.time_evolution = np.empty([signal.time_properties.time_index_max, state_properties.spin_quantum_number.dimension, state_properties.spin_quantum_number.dimension], np.complex128)
        self.state = np.empty([signal.time_properties.time_index_max, state_properties.spin_quantum_number.dimension], np.complex128)
        self.spin = np.empty([signal.time_properties.time_index_max, 3], np.float64)
        self.sensed_frequency_amplitude = 0.0
        self.sensed_frequency_amplitude_method = "none"

    def write_to_file(self, archive, do_write_everything = False):
        """
        Saves results to the hdf5 file.

        Parameters
        ----------
        archive : :class:`h5py.Group`
            The HDF5 archive to write to.
        do_write_everything : :obj:`bool`, optional
            If :obj:`True`, then save all time series data to file as well as parametric data. Defaults to :obj:`False` to reduce archive file size.
        """
        archive_group = archive.require_group("simulation_results")
        if do_write_everything:
            archive_group["time_evolution"] = self.time_evolution
            archive_group["state"] = self.state
            archive_group["spin"] = self.spin
        archive_group["sensed_frequency_amplitude"] = self.sensed_frequency_amplitude
        archive_group["sensed_frequency_amplitude"].attrs["method"] = self.sensed_frequency_amplitude_method

class Simulation:
    """
    The data needed and algorithms to control an individual simulation.

    Attributes
    ----------
    signal : :class:`test_signal.TestSignal`
        The :class:`test_signal.TestSignal` object source signal to be measured.
    source_properties : :class:`SourceProperties`
        The :class:`SourceProperties` parametrised sinusoidal source object to evolve the state with.
    state_properties : :class:`StateProperties`
        The :class:`StateProperties` initial conditions for the wavefunction of the quantum system.
    simulation_results : :class:`SimulationResults`
        A record of the results of the simulation.
    trotter_cutoff : :obj:`int`
        The number of squares made by the spin 1 matrix exponentiator.
    """
    def __init__(self, signal, dressing_rabi_frequency = 1e3, state_properties = None, trotter_cutoff = 28, device = spinsim.Device.CUDA, measurement_method = MeasurementMethod.FARADAY_DEMODULATION):
        """
        Parameters
        ----------
        signal : :class:`test_signal.TestSignal`
            The :class:`test_signal.TestSignal` object source signal to be measured.
        dressing_rabi_frequency : :obj:`float`, optional
            The amplitude of the dressing radiation to be applied to the system. Units of Hz.
        state_properties : :class:`StateProperties`, optional
            The :class:`StateProperties` initial conditions for the wavefunction of the quantum system.
        trotter_cutoff : :obj:`int`, optional
            The number of squares made by the spin 1 matrix exponentiator.
        device : :obj:`spinsim.Device`
            The target device that the simulation is to be run on.
        """
        self.signal = signal
        self.state_properties = state_properties
        if not self.state_properties:
            self.state_properties = StateProperties()
        self.source_properties = SourceProperties(self.signal, self.state_properties, dressing_rabi_frequency, 0.0, measurement_method = measurement_method)
        self.simulation_results = SimulationResults(self.signal, self.state_properties)
        self.trotter_cutoff = trotter_cutoff
        self.device = device

        # self.get_time_evolution = spinsim.time_evolver_factory(self.source_properties.evaluate_dressing, self.state_properties.spin_quantum_number, trotter_cutoff = trotter_cutoff)
        self.simulator = spinsim.Simulator(self.source_properties.evaluate_dressing, self.state_properties.spin_quantum_number, self.device, trotter_cutoff = trotter_cutoff, max_registers = 63, threads_per_block = 64)

    def evaluate(self, rabi_frequency):
        """
        Time evolves the system, and finds the spin at each coarse time step.

        Parameters
        ----------
        rabi_frequency : :obj:`float`
            A value that modifies the field sampling in a simulation. Used to make rabi frequency sweeps.
        """
        results = self.simulator.evaluate(rabi_frequency, self.signal.time_properties.time_end_points[0], self.signal.time_properties.time_end_points[1], self.signal.time_properties.time_step_fine, self.signal.time_properties.time_step_coarse, self.state_properties.state_init)
        self.simulation_results.state = results.state
        self.signal.time_properties.time_coarse = results.time
        self.simulation_results.time_evolution = results.time_evolution
        self.simulation_results.spin = results.spin

    def get_frequency_amplitude_from_projection(self):
        self.simulation_results.sensed_frequency_amplitude = self.simulation_results.spin[self.simulation_results.spin.shape[0] - 1, 2]
        # print(self.simulation_results.sensed_frequency_amplitude)
        self.simulation_results.sensed_frequency_amplitude *= -1/(2*math.pi*(self.signal.time_properties.time_end_points[1] - self.signal.time_properties.time_end_points[0]))

        if self.state_properties.spin_quantum_number == spinsim.SpinQuantumNumber.HALF:
            self.simulation_results.sensed_frequency_amplitude *= 2

    def get_frequency_amplitude_from_demodulation(self, demodulationTime_end_points = [0.09, 0.1], do_plot_spin = False, archive = None):
        """
        Uses demodulation of the Faraday signal to find the measured Fourier coefficient.

        Parameters
        ----------
        demodulationTime_end_points : :class:`numpy.ndarray` of :class:`numpy.float64` (start time (0) or end time (1)), optional
            The bounds of the interval where Faraday demodulation is used to acquire the measured frequency amplitude (Fourier coefficient) required for reconstruction
        do_plot_spin : :obj:`bool`, optional
            If :obj:`True`, plot the full time series for the expected spin of the system.
        """

        # Look at the end of the signal only to find the displacement
        demodulationTime_end_points = np.asarray(demodulationTime_end_points)
        demodulationTime_index_endpoints = np.floor(demodulationTime_end_points / self.signal.time_properties.time_step_coarse)
        time_coarse = 1*np.ascontiguousarray(self.signal.time_properties.time_coarse[int(demodulationTime_index_endpoints[0]):int(demodulationTime_index_endpoints[1])])
        spin = 1*np.ascontiguousarray(self.simulation_results.spin[int(demodulationTime_index_endpoints[0]):int(demodulationTime_index_endpoints[1]), 0])
        spin_demodulated = np.empty_like(spin)

        # spin = 1*self.simulation_results.spin[:, 0]
        # time_coarse = 1*self.signal.time_properties.time_coarse
        # plt.figure()
        # plt.plot(time_coarse, spin)
        # plt.plot(self.simulation_results.spin[:, :])
        # plt.plot(self.signal.time_properties.time_coarse, self.simulation_results.spin[:, :])

        # Decide GPU block and grid sizes
        threads_per_block = 128
        blocks_per_grid = (self.signal.time_properties.time_index_max + (threads_per_block - 1)) // threads_per_block

        # Multiply the Faraday signal by 2cos(wt)
        get_frequency_amplitude_from_demodulation_multiply[blocks_per_grid, threads_per_block](time_coarse, spin, spin_demodulated, self.source_properties.source_amplitude[0, 2])

        # plt.plot(time_coarse, spin_demodulated)

        # Average the result of the multiplication (ie apply a strict low pass filter and retreive the DC value)
        self.simulation_results.sensed_frequency_amplitude = 0.0
        self.simulation_results.sensed_frequency_amplitude_method = "demodulation"
        self.simulation_results.sensed_frequency_amplitude = get_frequency_amplitude_from_demodulation_low_pass(self.signal.time_properties.time_end_points, spin_demodulated, self.simulation_results.sensed_frequency_amplitude)

        if self.state_properties.spin_quantum_number == spinsim.SpinQuantumNumber.HALF:
            self.simulation_results.sensed_frequency_amplitude *= 2

        # plt.plot(time_coarse, spin*0 + self.simulation_results.sensed_frequency_amplitude)

        if do_plot_spin:
            plt.figure()
            plt.plot(self.signal.time_properties.time_coarse, self.simulation_results.spin[:, :])
            plt.legend(("x", "y", "z"))
            plt.xlim(0e-3, 2e-3)
            plt.xlabel("Time (s)")
            plt.ylabel("Spin projection (hbar)")
            if archive:
                plt.title(archive.execution_time_string + "Spin projection")
                archive_index = 0
                while os.path.isfile(archive.plot_path + "spin_projection_" + str(archive_index) + ".png"):
                    archive_index += 1
                plt.savefig(archive.plot_path + "spin_projection_" + str(archive_index) + ".pdf")
                plt.savefig(archive.plot_path + "spin_projection_" + str(archive_index) + ".png")
            else:
                plt.show()

    def write_to_file(self, archive, do_write_everything = False):
        """
        Saves the simulation record to hdf5.

        Parameters
        ----------
        archive : :class:`h5py.Group`
            The HDF5 archive to write to.
        do_write_everything : :obj:`bool`, optional
            If :obj:`True`, then save all time series data to file as well as parametric data. Defaults to :obj:`False` to reduce archive file size.
        """
        self.source_properties.write_to_file(archive)
        self.state_properties.write_to_file(archive)
        self.simulation_results.write_to_file(archive, do_write_everything)

@cuda.jit(debug = cuda_debug)
def get_frequency_amplitude_from_demodulation_multiply(time, spin, spin_demodulated, bias_amplitude):
    """
    Multiply each spin value by :math:`-2\\cos(2\\pi f_\\mathrm{bias, amp} t)` as part of a demodulation.

    Parameters
    ----------
    time : :class:`numpy.ndarray` of :class:`numpy.float64` (time_index)
        The time at which each spin sample is taken.
    spin : :class:`numpy.ndarray` of :class:`numpy.float64` (time_index, spatial_index)
        The expected value for hyperfine spin of the spin system in the lab frame, for each time sampled. Units of :math:`\\hbar`.
    spin_demodulated : :class:`numpy.ndarray` of :class:`numpy.float64` (time_index, spatial_index)
        The expected value for hyperfine spin of the spin system in the lab frame, multiplied by :math:`-2\\cos(2\\pi f_\\mathrm{bias, amp} t)` for each time sampled. Units of :math:`\\hbar`. This is an output, so use an empty :class:`numpy.ndarray` with :func:`numpy.empty()`, or declare a :class:`numba.cuda.cudadrv.devicearray.DeviceNDArray` using :func:`numba.cuda.device_array_like()`.
    bias_amplitude : :obj:`float`
        The carrier frequency :math:`f_\\mathrm{bias, amp}` for which to demodulate by.
    """
    time_index = cuda.threadIdx.x + cuda.blockIdx.x*cuda.blockDim.x
    if time_index < spin.shape[0]:
        spin_demodulated[time_index] = -2*math.cos(2*math.pi*bias_amplitude*time[time_index])*spin[time_index]

@nb.jit(nopython = True)
def get_frequency_amplitude_from_demodulation_low_pass(time_end_points, spin, sensed_frequency_amplitude):
    """
    Average the multiplied spin to find the DC value (ie apply a low pass filter).

    Parameters
    ----------
    time_end_points : :class:`numpy.ndarray` of :class:`numpy.float64` (start time (0) or end time (1))
        The bounds of the interval where Faraday demodulation is used to acquire the measured frequency amplitude (Fourier coefficient) required for reconstruction
    spin : :class:`numpy.ndarray` of :class:`numpy.float64` (time_index, spatial_index)
        The expected value for hyperfine spin of the spin system in the lab frame, multiplied by :math:`-2\\cos(2\\pi f_\\mathrm{bias, amp} t)` for each time sampled. Units of :math:`\\hbar`. The output `spin_demodulated` from :func:`get_frequency_amplitude_from_demodulation_multiply()`.
    sensed_frequency_amplitude : :obj:`float`
        The measured Fourier coefficient from the simulation. This is an output.
    """
    sensed_frequency_amplitude = 0.0
    # spin = 1/2 g T coefficient
    for time_index in range(spin.size):
        sensed_frequency_amplitude += spin[time_index]
    sensed_frequency_amplitude *= -1/(2*math.pi*spin.size*(time_end_points[1] - time_end_points[0]))
    return sensed_frequency_amplitude

@cuda.jit(debug = cuda_debug)
def get_frequency_amplitude_from_demodulation(time, spin, spin_demodulated, bias_amplitude):
    """
    Demodulate a spin timeseries with a basic block low pass filter.

    Parameters
    ----------
    time : :class:`numpy.ndarray` of :class:`numpy.float64` (time_index)
        The time at which each spin sample is taken.
    spin : :class:`numpy.ndarray` of :class:`numpy.float64` (time_index, spatial_index)
        The expected value for hyperfine spin of the spin system in the lab frame, for each time sampled. Units of :math:`\\hbar`.
    spin_demodulated : :class:`numpy.ndarray` of :class:`numpy.float64` (time_index, spatial_index)
        The expected value for hyperfine spin of the spin system in the lab frame, demodulated by `bias_amplitude` for each time sampled. Units of :math:`\\hbar`. This is an output, so use an empty :class:`numpy.ndarray` with :func:`numpy.empty()`, or declare a :class:`numba.cuda.cudadrv.devicearray.DeviceNDArray` using :func:`numba.cuda.device_array_like()`.
    bias_amplitude : :obj:`float`
        The carrier frequency :math:`f_\\mathrm{bias, amp}` for which to demodulate by.
    """
    time_index = cuda.threadIdx.x + cuda.blockIdx.x*cuda.blockDim.x
    if time_index < spin.shape[0]:
        spin_demodulated[time_index] = 0
        for time_index_offset in range(-50, 51):
            time_index_use = time_index + time_index_offset
            if time_index_use < 0:
                time_index_use = 0
            elif time_index_use > spin.shape[0] - 1:
                time_index_use = spin.shape[0] - 1
            spin_demodulated[time_index] -= 2*math.cos(2*math.pi*bias_amplitude*time[time_index_use])*spin[time_index_use]/101

# @cuda.jit(debug = cuda_debug)
# def evaluate_dressing(time_step_source, source_index_max, source_amplitude, source_frequency, source_phase, source_time_end_points, source_quadratic_shift, source):
#     time_source_index = cuda.threadIdx.x + cuda.blockIdx.x*cuda.blockDim.x
#     if time_source_index < source.shape[0]:
#         time_source = time_source_index*time_step_source
#         for source_index in range(source_index_max):
#             for spacial_index in range(source.shape[1]):
#                 source[time_source_index, spacial_index] += \
#                     (time_source >= source_time_end_points[source_index, 0])*\
#                     (time_source <= source_time_end_points[source_index, 1])*\
#                     source_amplitude[source_index, spacial_index]*\
#                     math.sin(\
#                         math.tau*source_frequency[source_index, spacial_index]*\
#                         (time_source - source_time_end_points[source_index, 0]) +\
#                         source_phase[source_index, spacial_index]\
#                     )
#             source[time_source_index, 3] = source_quadratic_shift

def dressing_evaluator_factory(source_index_max, source_amplitude, source_frequency, source_phase, source_time_end_points, source_quadratic_shift = 0.0, measurement_method = MeasurementMethod.FARADAY_DEMODULATION):
    """
    Makes a field :obj:`callable` for :mod:`spinsim` of block pulses and false neural pulses. `field_modifier` sweeps the Rabi frequency.

    Parameters
    ----------
    source_index_max : :obj:`int`
        The number of sources in the simulation.
    source_amplitude : :class:`numpy.ndarray` of :class:`numpy.float64`, (source_index, spatial_index)
        The amplitude of the sine wave of source `source_index` in direction `spatial_index`. See :math:`f_\\textrm{amp}` above. In units of Hz.
    source_phase : :class:`numpy.ndarray` of :class:`numpy.float64`, (source_index, spatial_index)
        The phase offset of the sine wave of source `source_index` in direction `spatial_index`. See :math:`\\phi` above. In units of radians.
    source_frequency : :class:`numpy.ndarray` of :class:`numpy.float64`, (source_index, spatial_index)
        The frequency of the sine wave of source `source_index` in direction `spatial_index`. See :math:`f` above. In units of Hz.
    source_time_end_points : :class:`numpy.ndarray` of :class:`numpy.float64`, (source_index, turn on time (0) or turn off time (1))
        The times that the sine wave of source `source_index` turns on and off. See :math:`\\tau` above. In units of s.
    source_quadratic_shift :  :obj:`float`
        The constant quadratic shift of the spin 1 system, in Hz.

    Returns
    -------
    evaluate_dressing(time_sample, rabi_frequency, field_sample) : :obj:`callable`
        A field :obj:`callable` for :mod:`spinsim` of block pulses and false neural pulses. `field_modifier` sweeps the Rabi frequency.

        Parameters:

        * **time_sample** (:class:`numpy.float64`) - The time at which to sample the field. Measured in s.
        * **rabi_frequency** (:class:`numpy.float64`) - The amplitude of the dressing of the atoms. Measured in Hz.
        * **field_sample** (:class:`numpy.ndarray` of :class:`numpy.float64`) - The returned sampled field amplitude at time `time_sample`. Measured in Hz.
    """
    if measurement_method == MeasurementMethod.ADIABATIC_DEMAPPING:
        sweep_duration = 2e-2
        sweep_speed = 10
        def evaluate_dressing(time_sample, rabi_frequency, field_sample):
            field_sample[0] = 0
            field_sample[1] = 0
            field_sample[2] = 0
            for source_index in range(source_index_max):
                for spacial_index in range(field_sample.size - 1):
                    if source_index == 0 and spacial_index == 0:
                        if (time_sample >= source_time_end_points[source_index, 0]) and (time_sample < source_time_end_points[source_index, 1]):
                            if time_sample >= source_time_end_points[source_index, 1] - sweep_duration:
                                # Circular
                                sweep_attenuation = math.cos((math.pi/2)*(time_sample - (source_time_end_points[source_index, 1] - sweep_duration))/sweep_duration)
                                amplitude = rabi_frequency*sweep_attenuation
                                phase = math.tau*rabi_frequency*(2*sweep_duration/math.pi)*(1 - sweep_attenuation)

                                # # Hyperbolic
                                # sweep_attenuation = 1/math.cosh(sweep_speed*(time_sample - (source_time_end_points[source_index, 1] - sweep_duration))/sweep_duration)
                                # amplitude = rabi_frequency*sweep_attenuation
                                # phase = math.tau*rabi_frequency*(sweep_duration/sweep_speed)*math.log(sweep_attenuation)

                                # # Linear
                                # sweep_attenuation = (time_sample - (source_time_end_points[source_index, 1] - sweep_duration))/sweep_duration
                                # amplitude = rabi_frequency*(1 - sweep_attenuation)
                                # phase = math.tau*(4*sweep_duration*rabi_frequency/2)*(sweep_attenuation**2)
                            else:
                                amplitude = rabi_frequency
                                phase = 0
                            field_sample[spacial_index] += \
                                2*amplitude*\
                                math.sin(\
                                    math.tau*source_frequency[source_index, spacial_index]*\
                                    (time_sample - source_time_end_points[source_index, 0]) +\
                                    source_phase[source_index, spacial_index] +\
                                    phase\
                                )
                    else:
                        if (time_sample >= source_time_end_points[source_index, 0]) and (time_sample < source_time_end_points[source_index, 1]):
                            field_sample[spacial_index] += \
                                source_amplitude[source_index, spacial_index]*\
                                math.sin(\
                                    math.tau*source_frequency[source_index, spacial_index]*\
                                    (time_sample - source_time_end_points[source_index, 0]) +\
                                    source_phase[source_index, spacial_index]\
                                )
            if field_sample.size > 3:
                field_sample[3] = source_quadratic_shift
    else:
        def evaluate_dressing(time_sample, rabi_frequency, field_sample):
            field_sample[0] = 0
            field_sample[1] = 0
            field_sample[2] = 0
            for source_index in range(source_index_max):
                for spacial_index in range(field_sample.size - 1):
                    if (time_sample >= source_time_end_points[source_index, 0]) and (time_sample < source_time_end_points[source_index, 1]):
                        amplitude = source_amplitude[source_index, spacial_index]
                        if source_index == 0 and spacial_index == 0:
                            amplitude = 2*rabi_frequency
                        field_sample[spacial_index] += \
                            amplitude*\
                            math.sin(\
                                math.tau*source_frequency[source_index, spacial_index]*\
                                (time_sample - source_time_end_points[source_index, 0]) +\
                                source_phase[source_index, spacial_index]\
                            )
            if field_sample.size > 3:
                field_sample[3] = source_quadratic_shift
    return evaluate_dressing

class ExperimentResults:
    """
    A class that contains information from the results of a complete experiment, whether it was done via simulations, or in the lab.

    Attributes
    ----------
    frequency : :class:`numpy.ndarray` of :class:`numpy.float64`
        The dressing Rabi frequencies at which the experiments were run. In units of Hz.
    frequency_amplitude : :class:`numpy.ndarray` of :class:`numpy.float64`
        The Fourier sine coefficient of the signal being measured, as determined by the experiment. In units of Hz.
    """
    def __init__(self, frequency = None, frequency_amplitude = None):
        """
        Parameters
        ----------
        frequency : :class:`numpy.ndarray` of :class:`numpy.float64`, optional
            The dressing Rabi frequencies at which the experiments were run. In units of Hz.
        frequency_amplitude : :class:`numpy.ndarray` of :class:`numpy.float64`, optional
            The Fourier sine coefficient of the signal being measured, as determined by the experiment. In units of Hz.
        """
        self.frequency = frequency
        self.frequency_amplitude = frequency_amplitude

    @staticmethod
    def new_from_simulation_manager(simulation_manager):
        """
        A constructor that creates a new :class:`ExperimentResults` object from an already evaluated :class:`SimulationManager` object. That is, make an experiment results object based off of simulation results.

        Parameters
        ----------
        simulation_manager : :class:`SimulationManager`
            The simulation manager object to read the results of.

        Returns
        -------
        experiment_results : :class:`ExperimentResults`
            A new object containing the results of `simulation_manager`.
        """
        frequency = 1*simulation_manager.frequency
        frequency_amplitude = 1*simulation_manager.frequency_amplitude[:frequency.size]

        return ExperimentResults(frequency, frequency_amplitude)

    def plot(self, archive = None, test_signal = None):
        """
        Plots the experiment results contained in the object. Optionally saves the plots, as well as compares the results to numerically calculated values.

        Parameters
        ----------
        archive : :class:`archive.Archive`, optional
            If specified, will save the generated plot to the archive's :attr:`archive.Archive.plot_path`.
        test_signal : :class:`test_signal:TestSignal`, optional
            The signal that was being measured during the experiment. If specified, this function will plot the sine Fourier transform of the signal behind the measured coefficients of the experiment results.
        """
        plt.figure()
        if test_signal:
            plt.plot(test_signal.frequency, test_signal.frequency_amplitude, "-k")
        plt.plot(self.frequency, self.frequency_amplitude, "xr")
        if test_signal:
            plt.legend(["Fourier Transform", "Measured"])
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude (Hz)")
        plt.xlim([0, 2000])
        # plt.ylim([-0.08, 0.08])
        plt.grid()
        if archive:
            plt.title(archive.execution_time_string + "Measured Frequency Amplitude")
            plt.savefig(archive.plot_path + "measured_frequency_amplitude.pdf")
            plt.savefig(archive.plot_path + "measured_frequency_amplitude.png")
        plt.show()

    def write_to_archive(self, archive):
        """
        Saves the contents of the experiment results to an archive.

        Parameters
        ----------
        archive : :class:`archive.archive`
            The archive object to save the results to.
        """
        archive_group_experiment_results = archive.archive_file.require_group("experiment_results")
        archive_group_experiment_results["frequency"] = self.frequency
        archive_group_experiment_results["frequency_amplitude"] = self.frequency_amplitude