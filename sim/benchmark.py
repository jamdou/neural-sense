import matplotlib.pyplot as plt
import matplotlib.lines as lns
import time as tm
from numba.core import errors
import numpy as np
import numba as nb
from numba import cuda
from enum import Enum
import math
import h5py
import subprocess
import textwrap
import gc

# import qutip
# from qutip.expect import expect

import scipy.integrate
import scipy.linalg

from . import manager
import spinsim
from archive import *
import test_signal
import sim.manager

class BenchmarkType(Enum):
    """
    An enum to define the type of benchmark being done. Each gives labels and plot parameters to allow for the plotting and arching code to be modular.

    Parameters
    ----------
    _value_ : :obj:`str`
        Label for the archive.
    x_label : :obj:`str`
        Horizontal label for when plotting.
    y_label : :obj:`str`
        Vertical label for when plotting.
    title : :obj:`str`
        Title for when plotting.
    x_scale : :obj:`str`
        The type of scaling to apply to the x axis for when plotting. Either `"linear"` for a linear scale, or `"log"` for a log scale.
    """
    def __init__(self, value, x_label, y_label, title, x_scale):
        super().__init__()
        self._value_ = value
        self.x_label = x_label
        self.y_label = y_label
        self.title = title
        self.x_scale = x_scale

    NONE = (
        "none",
        "Nothing (rad)",
        "Error",
        "Nothing",
        "log"
    )
    """
    No benchmark has been defined.
    """

    TIME_STEP_SOURCE = (
        "time_step_source",
        "Source time step (s)",
        "Error",
        "Effect of source time step size on error",
        "log"
    )
    """
    The results of :func:`benchmark.manager.new_benchmark_number_of_squares()`.
    """

    number_of_squares = (
        "number_of_squares",
        "Trotter cutoff",
        "Error",
        "Effect of trotter cutoff on error",
        "linear"
    )
    """
    The results of :func:`benchmark.manager.new_benchmark_number_of_squares()`.
    """

    number_of_squares_MATRIX = (
        "number_of_squares_matrix",
        "Trotter cutoff",
        "Error",
        "Effect of trotter cutoff on error",
        "linear"
    )
    """
    The results of :func:`benchmark.manager.new_benchmark_number_of_squares_matrix()`.
    """

    TIME_STEP_FINE = (
        "time_step_fine",
        "Fine time step (s)",
        "Error",
        "Effect of fine time step size on error",
        "log"
    )
    """
    The results of :func:`benchmark.manager.new_benchmark_time_step_fine()`.
    """

    EXECUTION_TIME_ERROR = (
        "execution_time_error",
        "Execution time (s)",
        "Error",
        "Effect of execution time on error",
        "log"
    )

    TIME_STEP_FINE_EXECUTION_TIME = (
        "time_step_fine_execution_time",
        "Fine time step (s)",
        "Execution time (s)",
        "Effect of fine time step size on execution time",
        "log"
    )
    """
    The results of :func:`benchmark.manager.new_benchmark_time_step_fine()`.
    """

    TIME_STEP_FINE_FREQUENCY_DRIFT = (
        "time_step_fine_frequency_drift",
        "Fine time step (s)",
        "Frequency shift (Hz)",
        "Effect of fine time step size on frequency shift",
        "log"
    )
    """
    The results of :func:`benchmark.manager.new_benchmark_time_step_fine_frequency_drift()`.
    """

class BenchmarkResults:
    """
    A class that holds the results of an arbitrary benchmark, and has the ability to plot them.

    Attributes
    ----------
    benchmark_type : :class:`BenchmarkType`
        The benchmark that this was the result of. Also contains information used to archive and plot the results.
    parameter : :class:`numpy.ndarray`
        The value of the parameter being varied during the benchmark.
    error : :class:`numpy.ndarray`
        The error recorded during the benchmark.
    """
    def __init__(self, benchmark_type = BenchmarkType.NONE, parameter = None, error = None):
        """
        Parameters
        ----------
        benchmark_type : :class:`BenchmarkType`, optional
            The benchmark that this was the result of. Also contains information used to archive and plot the results. Defaults to :obj:`BenchmarkType.NONE`.
        parameter : :class:`numpy.ndarray`
            The value of the parameter being varied during the benchmark. Defaults to `None`.
        error : :class:`numpy.ndarray`
            The error recorded during the benchmark. Defaults to `None`.
        """
        self.benchmark_type = benchmark_type
        self.parameter = parameter
        self.error = error

    @staticmethod
    def read_from_archive(archive, benchmark_type = None):
        """
        A constructor that reads a new benchmark result from a hdf5 file.

        Parameters
        ----------
        archive : :class:`archive.Archive`
            The archive object to read the benchmark from.
        """
        archive_group_benchmark = archive.archive_file["benchmark_results"]
        if not benchmark_type:
            for benchmark_type_scan in BenchmarkType:
                if benchmark_type_scan.value in archive_group_benchmark.keys():
                    archive_group_benchmark_results = archive_group_benchmark[benchmark_type_scan.value]
                    benchmark_results = BenchmarkResults(
                        benchmark_type_scan,
                        np.asarray(archive_group_benchmark_results[benchmark_type_scan.value]),
                        np.asarray(archive_group_benchmark_results["error"])
                    )
                    return benchmark_results
        else:
            if benchmark_type.value in archive_group_benchmark.keys():
                archive_group_benchmark_results = archive_group_benchmark[benchmark_type.value]
                benchmark_results = BenchmarkResults(
                    benchmark_type,
                    np.asarray(archive_group_benchmark_results[benchmark_type.value]),
                    np.asarray(archive_group_benchmark_results["error"])
                )
                return benchmark_results

    def write_to_archive(self, archive:Archive):
        """
        Save a benchmark to a hdf5 file.

        Parameters
        ----------
        archive : :class:`archive.Archive`
            The archive object to write the benchmark to.
        """
        archive_group_benchmark_results = archive.archive_file.require_group("benchmark_results/" + self.benchmark_type.value)
        archive_group_benchmark_results[self.benchmark_type.value] = self.parameter
        archive_group_benchmark_results["error"] = self.error

    def plot(self, archive = None, do_show_plot = True):
        """
        Plots the benchmark results.

        Parameters
        ----------
        archive : :class:`archive.Archive`, optional
            If specified, will save plots to the archive's `plot_path`.
        do_show_plot : :obj:`bool`, optional
            If :obj:`True`, will attempt to show and save the plots. Can be set to false to overlay multiple archive results to be plotted later, as is done with :func:`benchmark_manager.plot_benchmark_comparison()`.
        """
        if do_show_plot:
            plt.figure()
            plt.plot(self.parameter[1:], self.error[1:], "k.-")
        else:
            plt.plot(self.parameter[1:], self.error[1:], ".-")
        plt.grid()
        plt.yscale("log")
        plt.xscale(self.benchmark_type.x_scale)
        plt.xlabel(self.benchmark_type.x_label)
        plt.ylabel(self.benchmark_type.y_label)
        if do_show_plot:
            if archive:
                archive.write_plot(self.benchmark_type.title, "benchmark_" + self.benchmark_type.value)
            plt.draw()

def plot_benchmark_comparison(archive:Archive, archive_times, legend, title):
    """
    Plots multiple benchmarks on one plot from previous archives.

    Parameters
    ----------
    archive : :class:`archive.Archive`
        Specifies the path to save the plot to.
    archive_times : :obj:`list` of :obj:`str`
        The identifiers of the archvies containing the benchmark results to be compared.
    legend : :obj:`list` of :obj:`str`
        Labels that describe what each of the benchmark result curves respresent.
    title : :obj:`str`
        What this comparison is trying to compare.
    """
    plt.figure()
    for archive_time in archive_times:
        archive_previous = Archive(archive.archive_path[:-25], "")
        archive_previous.open_archive_file(archive_time)
        benchmark_results = BenchmarkResults.read_from_archive(archive_previous)
        benchmark_results.plot(None, False)
        archive_previous.close_archive_file(False)
    plt.legend(legend)

    if archive:
        archive.write_plot(title, "benchmark_comparison")

        archive_group_benchmark_results = archive.archive_file.require_group("benchmark_results/benchmark_comparison")
        archive_group_benchmark_results["archive_times"] = np.asarray(archive_times, dtype='|S32')
        archive_group_benchmark_results["legend"] = np.asarray(legend, dtype='|S32')
        archive_group_benchmark_results["title"] = np.asarray([title], dtype='|S32')
    plt.draw()

def new_benchmark_device_aggregate(archive:Archive, archive_times):
    """
    Collects results for multiple runs of :func:`new_benchmark_device()`, and combines them into a single bar chart.

    Parameters
    ----------
    archive : :class:`archive.Archive`
        Specifies the location to save the plots to, as well as execution time.
    archive_times : :obj:`list` of :obj:`str`
        The execution times for the archvies to be read in.
    """
    device_label = []
    execution_time = np.empty(0, dtype = np.float64)
    for archive_time in archive_times:
        archive_previous = Archive(archive.archive_path[:-25], "")
        archive_previous.open_archive_file(archive_time)
        archive_group_benchmark_device = archive_previous.archive_file.require_group("benchmark_results/device")
        device_label = np.append(device_label, np.asarray(archive_group_benchmark_device["device"]))
        execution_time = np.append(execution_time, archive_group_benchmark_device["execution_time"])
        archive_previous.close_archive_file(False)

    order = np.argsort(execution_time)
    order = order[::-1]
    execution_time = execution_time[order]
    device_label = device_label[order]
    device_label = device_label.tolist()
    execution_frequency = 1/execution_time

    core_count = {
        "Core i7-6700" : 4,
        "Core i7-8750H" : 6,
        "Core i7-10850H" : 6,
        "Ryzen 9 5900X" : 12,
        "Ryzen 7 5800X" : 8,
        "Quadro K620" : 384,
        "GeForce GTX 1070" : 2048,
        "GeForce RTX 3070" : 5888,
        "GeForce RTX 3080" : 8704,
        "Quadro T1000" : 768
    }

    colour = []
    hatch = []
    device_label_reformat = []
    for device_label_index in range(len(device_label)):
        device_label[device_label_index] = device_label[device_label_index].decode('UTF-8')
        device_label_instance = device_label[device_label_index]
        if "intel" in device_label_instance.lower():
            if "i7-10" in device_label_instance.lower():
                device_label_reformat_instance = f"Core {device_label_instance[18:27].strip()}"
            else:
                device_label_reformat_instance = f"Core {device_label_instance[18:26].strip()}"
            if "multi" in device_label_instance.lower():
                device_label[device_label_index] = f"{device_label_reformat_instance}\n({core_count[device_label_reformat_instance]} core CPU)"
                hatch += [""]
            else:
                device_label[device_label_index] = f"{device_label_reformat_instance}\n(CPU, single thread)"
                hatch += [""]
                # hatch += ["//"]
            colour += ["b"]
        elif "cuda" in device_label_instance.lower():
            if "nvidia" in device_label_instance.lower():
                device_label_reformat_instance = device_label_instance[6:-4].strip()
            else:
                device_label_reformat_instance = device_label_instance[:-4].strip()
            device_label[device_label_index] = f"{device_label_reformat_instance}\n({core_count[device_label_reformat_instance]} core cuda GPU)"
            hatch += [""]
            colour += ["g"]
        else:
            device_label_reformat_instance = device_label_instance[4:18].strip()
            if "multi" in device_label_instance.lower():
                device_label[device_label_index] = f"{device_label_reformat_instance}\n({core_count[device_label_reformat_instance]} core CPU)"
                hatch += [""]
            else:
                device_label[device_label_index] = f"{device_label_reformat_instance}\n(CPU, single thread)"
                # hatch += ["//"]
                hatch += [""]
            colour += ["r"]
        device_label_reformat += [device_label_reformat_instance]

    plt.figure(figsize = [6.4, 8.0])
    plt.subplots_adjust(left=0.3, right=0.95, top=0.9, bottom=0.1)
    for device_index in range(len(device_label)):
        print(device_label[device_index])
        plt.barh(device_label[device_index], execution_frequency[device_index], color = colour[device_index], hatch = hatch[device_index])
    for device_index in range(len(device_label)):
        if device_index == len(device_label) - 1:
            plt.text(execution_frequency[device_index], device_index, f" {execution_time[device_index]*1e3:.1f} ms per simulation ", ha = "right", va = "center", color = "w")
        else:
            if execution_frequency[device_index] > 0.8*execution_frequency[len(device_label) - 1]:
                plt.text(execution_frequency[device_index], device_index, f" {execution_time[device_index]*1e3:.1f} ms ", ha = "right", va = "center", color = "w")
            else:
                plt.text(execution_frequency[device_index], device_index, f" {execution_time[device_index]*1e3:.1f} ms ", ha = "left", va = "center")
    # rect = plt.Rectangle((16, 0), 1, 0.5, fill = False)
    # rect.set_hatch("//")
    # plt.text(16, 0.25, "Limitted to single thread ", ha = "right", va = "center")
    # plt.gca().add_patch(rect)
    rect = plt.Rectangle((16, 1), 1, 0.5, color = "b")
    plt.text(16, 1.25, "Intel CPU ", ha = "right", va = "center")
    plt.gca().add_patch(rect)
    rect = plt.Rectangle((16, 2), 1, 0.5, color = "r")
    plt.text(16, 2.25, "AMD CPU ", ha = "right", va = "center")
    plt.gca().add_patch(rect)
    rect = plt.Rectangle((16, 3), 1, 0.5, color = "g")
    plt.text(16, 3.25, "Nvidia GPU ", ha = "right", va = "center")
    plt.gca().add_patch(rect)

    plt.xlabel("Execution speed (simulations per second)")
    if archive:
        archive.write_plot("Parallelisation speed for various devices", "benchmark_device_aggregate")
    plt.show()
    

def new_benchmark_device(archive:Archive, signal:test_signal.TestSignal, frequency, state_properties:sim.manager.StateProperties):
    """
    Runs a benchmark to compare (single and multicore) CPU performance to GPU performance.

    Parameters
    ----------
    archive : :class:`archive.Archive`
        Specifies the path to save the plot to, as well as the archive file to save the results to.
    signal : :obj:`list` of :class:`test_signal.TestSignal`
        The signals being simulated in the benchmark.
    frequency : :class:`numpy.ndarray` of :class:`numpy.float64`
        The dressing frequencies being simulated in the benchmark.
    state_properties : :class:`sim.StateProperties`
        Specifies the initial state and spin quantum number of the quantum system simulated in the benchmarked.
    """
    state_output = []
    execution_time_output = []
    device = [
        spinsim.Device.CPU_SINGLE,
        spinsim.Device.CPU,
        spinsim.Device.CUDA
    ]

    cpu_name = "\n".join(textwrap.wrap(subprocess.check_output(["wmic","cpu","get", "name"]).strip().decode('utf-8').split("\n")[1], width = 23))

    gpu_name = cuda.list_devices()[0].name.decode('UTF-8')
    device_label = [
        "{}\nsingle thread".format(cpu_name),
        "{}\nmulti thread".format(cpu_name),
        "{}\ncuda".format(gpu_name)
    ]

    colour = []
    for device_label_instance in device_label:
        if "intel" in device_label_instance.lower():
            colour += ["b"]
        elif "cuda" in device_label_instance.lower():
            colour += ["g"]
        else:
            colour += ["r"]

    simulation_manager = manager.SimulationManager(signal, frequency, archive, state_properties, execution_time_output = execution_time_output, device = device)
    simulation_manager.evaluate(False)

    execution_time = np.zeros(len(device), np.float64)
    execution_time_first = np.zeros(len(device), np.float64)

    for device_index in range(len(device)):
        for frequency_index in range(frequency.size):
            if frequency_index == 0:
                execution_time_first[device_index] = execution_time_output[device_index*frequency.size]
            else:
                execution_time[device_index] += execution_time_output[frequency_index + device_index*frequency.size]
    execution_time /= frequency.size - 1

    speed_up = execution_time[0]/execution_time
    speed_up_first = execution_time_first[0]/execution_time_first

    if archive:
        archive_group_benchmark_device = archive.archive_file.require_group("benchmark_results/device")
        archive_group_benchmark_device["device"] = np.asarray(device_label, dtype='|S128')
        archive_group_benchmark_device["execution_time"] = execution_time
        archive_group_benchmark_device["speed_up"] = speed_up
        archive_group_benchmark_device["execution_time_first"] = execution_time
        archive_group_benchmark_device["speed_up_first"] = speed_up_first

    plt.figure()
    plt.subplots_adjust(left=0.3, right=0.95, top=0.85, bottom=0.1)
    plt.barh(range(len(device)), execution_time[0]/execution_time, tick_label = device_label, color = colour)
    for device_index in range(len(device_label)):
        if speed_up[device_index] > 0.6*speed_up[len(device_label) - 1]:
            plt.text(speed_up[device_index], device_index, " {:.1f}x speedup \n {:.1f}ms per sim ".format(speed_up[device_index], execution_time[device_index]*1e3), ha = "right", va = "center", color = "w")
        else:
            plt.text(speed_up[device_index], device_index, " {:.1f}x speedup \n {:.1f}ms per sim ".format(speed_up[device_index], execution_time[device_index]*1e3), ha = "left", va = "center")
    
    plt.xlabel("Speedup compared to single CPU thread")
    if archive:
        archive.write_plot("Parallelisation speed up for various devices", "benchmark_device")
    plt.show()

    plt.figure()
    plt.subplots_adjust(left=0.3, right=0.95, top=0.85, bottom=0.1)
    plt.barh(range(len(device)), execution_time_first[0]/execution_time_first, tick_label = device_label, color = colour)
    plt.text(speed_up_first[0], 0, " {:.1f}x speedup \n {:.1f}ms per sim ".format(speed_up_first[0], execution_time_first[0]*1e3), ha = "left", va = "center")
    plt.text(speed_up_first[1], 1, " {:.1f}x speedup \n {:.1f}ms per sim ".format(speed_up_first[1], execution_time_first[1]*1e3), ha = "left", va = "center")
    plt.text(speed_up_first[2], 2, " {:.1f}x speedup \n {:.1f}ms per sim ".format(speed_up_first[2], execution_time_first[2]*1e3), ha = "right", va = "center", color = "w")
    plt.xlabel("Speedup compared to single CPU thread")
    if archive:
        archive.write_plot("Parallelisation speed up for various devices,\nfirst run", "benchmark_device_first")
    plt.show()

    return

def new_benchmark_number_of_squares_matrix(archive:Archive, number_of_squares, norm_bound = 1.0):
    """
    Runs a benchmark for the trotter exponentiator :func:`utilities.matrixExponential_lie_trotter()` using arbitrary matrices. Uses :func:`benchmark_number_of_squares_matrix()` to execute the matrix exponentials.

    Specifically, let
    
    .. math::
        A_k = -i \\frac{\\nu}{4} (\\cos(k) F_x + \\cos(2k) F_y + \\cos(4k) F_z + \\cos(8k) F_q).

    See :func:`utilities.matrixExponential_lie_trotter()` for definitions of :math:`F` operators).

    Then :func:`utilities.matrixExponential_lie_trotter()` calculates the exponential of :math:`A_k` as

    .. math::
        E_{\\tau, k} = \\exp_\\tau(A_k).
    
    Let :math:`\\tau_0` be the first element in `number_of_squares`, ususally the largest. Then the error :math:`e_\\tau` is calculated as

    .. math::
        e_\\tau = \\frac{1}{\\#k}\\sum_{k, i, j} |(E_{\\tau, k})_{i,j} - E_{\\tau_0, k})_{i,j}|,

    where :math:`\\#k` is the number of matrices being considered in the benchmark (1e6).

    Parameters
    ----------
    archive : :class:`archive.Archive`
        Specifies where to save results and plots.
    number_of_squares : :class:`numpy.ndarray` of :class:`numpy.int`
        An array of values of the trotter cutoff to run the matrix exponentiator at.
    norm_bound : `float`, optional
        An upper bound to the size of the norm of the matrices being exponentiated, since :func:`utilities.matrixExponential_lie_trotter()` works better using matrices with smaller norms. See :math:`\\nu` above. Defaults to 1.

    Returns
    -------
    benchmark_results : :class:`BenchmarkResults`
        Contains the errors found by the benchmark.
    """
    print("\033[33mStarting benchmark...\033[0m")

    time_index_max = int(1e5)
    result = np.empty((time_index_max, 3, 3), dtype = np.complex128)
    result_bench = np.empty((time_index_max, 3, 3), dtype = np.complex128)
    number_of_squares = np.asarray(number_of_squares)
    error = np.empty_like(number_of_squares, dtype = np.float64)

    threads_per_block = 64
    blocks_per_grid = (time_index_max + (threads_per_block - 1)) // threads_per_block
    # benchmark_number_of_squares_matrix[blocks_per_grid, threads_per_block](norm_bound, number_of_squares[0], result_bench)
    Jx = (1/math.sqrt(2))*np.asarray(
        [
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0]
        ],
        dtype = np.complex128
    )
    Jy = (1/math.sqrt(2))*np.asarray(
        [
            [ 0, -1j,   0],
            [1j,   0, -1j],
            [ 0,  1j,   0]
        ],
        dtype = np.complex128
    )
    Jz = np.asarray(
        [
            [1, 0,  0],
            [0, 0,  0],
            [0, 0, -1]
        ],
        dtype = np.complex128
    )
    Qz = (1/3)*np.asarray(
        [
            [1,  0, 0],
            [0, -2, 0],
            [0,  0, 1]
        ],
        dtype = np.complex128
    )
    print("\tStarting scipy ground truth")
    for time_index in range(time_index_max):
        matrix = -1j*(norm_bound*math.cos(1.1*time_index)*Jx + norm_bound*math.sin(1.9*time_index)*Jy + norm_bound*math.cos(4.1*time_index)*Jz + norm_bound*math.sin(8.9*time_index)*Qz)
        result_bench[time_index, :, :] = scipy.linalg.expm(matrix)
        print(f"\t{(time_index + 1)*100/time_index_max}%", end = "\t\t\t\r")
    print("\n\tDone")
    print("\tStarting spinsim")
    for number_of_squares_index in range(number_of_squares.size):
        utilities = spinsim.Utilities(spinsim.SpinQuantumNumber.ONE, spinsim.Device.CUDA, threads_per_block, number_of_squares[number_of_squares_index])
        matrix_exponential_lie_trotter = utilities.matrix_exponential_lie_trotter

        @cuda.jit
        def benchmark_number_of_squares_matrix(norm_bound, result):
            """
            Runs the exponentiations for the trotter matrix benchmark.

            Parameters
            ----------
            norm_bound : `float`, optional
                An upper bound to the size of the norm of the matrices being exponentiated, since :func:`utilities.matrixExponential_lie_trotter()` works better using matrices with smaller norms. Defaults to 1.
            number_of_squares : :obj:`int`
                The value trotter cutoff to run the matrix exponentiator at.
            result : :class:`numpy.ndarray` of :class:`numpy.complex128`
                The results of the matrix exponentiations for this value of `number_of_squares`.
            """
            source_sample = cuda.local.array(4,dtype = nb.float64)

            time_index = cuda.threadIdx.x + cuda.blockIdx.x*cuda.blockDim.x
            if time_index < result.shape[0]:
                source_sample[0] = norm_bound*math.cos(1.1*time_index)
                source_sample[1] = norm_bound*math.sin(1.9*time_index)
                source_sample[2] = norm_bound*math.cos(4.1*time_index)
                source_sample[3] = norm_bound*math.sin(8.9*time_index)

                matrix_exponential_lie_trotter(source_sample, result[time_index, :])

        benchmark_number_of_squares_matrix[blocks_per_grid, threads_per_block](norm_bound, result)
        result_difference = (result - result_bench)
        error[number_of_squares_index] = np.sqrt(np.sum(np.real(result_difference*np.conj(result_difference))))/time_index_max

        print(f"\t{(number_of_squares_index + 1)*100/number_of_squares.size}%", end = "\t\t\t\r")
    print("\n\tDone")
    # time_index_max = 1000000
    # result = np.empty((time_index_max, 3, 3), dtype = np.complex128)
    # result_bench = np.empty((time_index_max, 3, 3), dtype = np.complex128)
    # number_of_squares = np.asarray(number_of_squares)
    # error = np.empty_like(number_of_squares, dtype = np.float64)
    
    # blocks_per_grid = (time_index_max + (threads_per_block - 1)) // threads_per_block
    # benchmark_number_of_squares_matrix[blocks_per_grid, threads_per_block](norm_bound, number_of_squares[0], result_bench)

    # for number_of_squares_index in range(number_of_squares.size):
    #     benchmark_number_of_squares_matrix[blocks_per_grid, threads_per_block](norm_bound, number_of_squares[number_of_squares_index], result)
    #     result_difference = (result - result_bench)
    #     error[number_of_squares_index] = np.sqrt(np.sum(np.real(result_difference*np.conj(result_difference))))/time_index_max

    print("\033[32mDone!\033[0m")

    benchmark_results = BenchmarkResults(BenchmarkType.number_of_squares_MATRIX, number_of_squares, error)
    benchmark_results.write_to_archive(archive)
    benchmark_results.plot(archive)

    return benchmark_results

def new_benchmark_number_of_squares(archive:Archive, signal:test_signal.TestSignal, frequency, number_of_squares):
    """
    Runs a benchmark for the trotter exponentiator using the integrator.

    Specifically, let :math:`(\\psi_{f,\\tau})_{m,t}` be the calculated state of the spin system, with magnetic number (`state_index`) :math:`m` at time :math:`t`, simulated with a dressing of :math:`f` with a trotter cutoff of :math:`\\tau`. Let :math:`\\tau_0` be the first such trotter cutoff in `number_of_squares` (generally the largest one). Then the error :math:`e_\\tau` calculated by this benchmark is

    .. math::
        \\begin{align*}
            e_\\tau &= \\frac{1}{\\#t\\#f}\\sum_{t,f,m} |(\\psi_{f,\\tau})_{m,t} - (\\psi_{f,\\tau_0})_{m,t}|,
        \\end{align*}

    where :math:`\\#t` is the number of coarse time samples, :math:`\\#f` is the length of `frequency`.

    Parameters
    ----------
    archive : :class:`archive.Archive`
        Specifies where to save results and plots.
    signal : :obj:`list` of :class:`test_signal.TestSignal`
        The signals being simulated in the benchmark.
    frequency : :class:`numpy.ndarray` of :class:`numpy.float64`
        The dressing frequencies being simulated in the benchmark.
    number_of_squares : :class:`numpy.ndarray` of :class:`numpy.int`
        An array of values of the trotter cutoff to run the simulations at. The accuracy of the simulation output with each of these values are then compared.

    Returns
    -------
    benchmark_results : :class:`BenchmarkResults`
        Contains the errors found by the benchmark.
    """
    state_output = []
    error = []
    simulation_manager = manager.SimulationManager(signal, frequency, archive, None, state_output, number_of_squares)
    simulation_manager.evaluate(False)
    for number_of_squares_index in range(number_of_squares.size):
        error_temp = 0
        for frequency_index in range(frequency.size):
            state_difference = state_output[frequency_index + number_of_squares_index*frequency.size] - state_output[frequency_index]
            error_temp += np.sum(np.sqrt(np.real(np.conj(state_difference)*state_difference)))
        error += [error_temp/(frequency.size*state_output[0].size)]
    
    number_of_squares = np.asarray(number_of_squares)
    error = np.asarray(error)

    benchmark_results = BenchmarkResults(BenchmarkType.number_of_squares, number_of_squares, error)
    benchmark_results.write_to_archive(archive)
    benchmark_results.plot(archive)

    return benchmark_results

def new_benchmark_time_step_fine(archive:Archive, signal_template:test_signal.TestSignal, frequency, time_step_fine, state_properties:sim.manager.StateProperties):
    """
    Runs a benchmark to test error induced by raising the size of the time step in the integrator, comparing the output state.

    Specifically, let :math:`(\\psi_{f,\\mathrm{d}t})_{m,t}` be the calculated state of the spin system, with magnetic number (`state_index`) :math:`m` at time :math:`t`, simulated with a dressing of :math:`f` with a fine time step of :math:`\\mathrm{d}t`. Let :math:`\\mathrm{d}t_0` be the first such time step in `time_step_fine` (generally the smallest one). Then the error :math:`e_{\\mathrm{d}t}` calculated by this benchmark is

    .. math::
        \\begin{align*}
            e_{\\mathrm{d}t} &= \\frac{1}{\\#t\\#f}\\sum_{t,f,m} |(\\psi_{f,\\mathrm{d}t})_{m,t} - (\\psi_{f,\\mathrm{d}t_0})_{m,t}|,
        \\end{align*}

    where :math:`\\#t` is the number of coarse time samples, :math:`\\#f` is the length of `frequency`.

    Parameters
    ----------
    archive : :class:`archive.Archive`
        Specifies where to save results and plots.
    signal_template : :class:`test_signal.TestSignal`
        A description of the signal to use for the environment during the simulation. For each entry in `time_step_fine`, this template is modified so that its :attr:`test_signal.TestSignal.time_properties.time_step_fine` is equal to that entry. All modified versions of the signal are then simulated for comparison.
    frequency : :class:`numpy.ndarray` of :class:`numpy.float64`
        The dressing frequencies being simulated in the benchmark.
    time_step_fine : :class:`numpy.ndarray` of :class:`numpy.float64`
        An array of time steps to run the simulations with. The accuracy of the simulation output with each of these values are then compared.

    Returns
    -------
    benchmark_results : :class:`BenchmarkResults`
        Contains the errors found by the benchmark.
    """
    time_step_fine = np.asarray(time_step_fine)
    state_output = []
    error = []

    signal = []
    for time_step_fine_index in time_step_fine:
        time_properties = test_signal.TimeProperties(signal_template.time_properties.time_step_coarse, time_step_fine_index, signal_template.time_properties.time_step_source)
        signal_instance = test_signal.TestSignal(signal_template.neural_pulses, signal_template.sinusoidal_noises, time_properties, False)
        signal += [signal_instance]

    simulation_manager = manager.SimulationManager(signal, frequency, archive, state_properties, state_output)
    simulation_manager.evaluate(False)

    for time_step_fine_index in range(time_step_fine.size):
        error_temp = 0
        for frequency_index in range(frequency.size):
            state_difference = state_output[frequency_index + time_step_fine_index*frequency.size] - state_output[frequency_index]

            error_temp += np.sum(np.sqrt(np.real(np.conj(state_difference)*state_difference)))
        error += [error_temp/(frequency.size*state_output[0].size)]
    
    error = np.asarray(error)

    benchmark_results = BenchmarkResults(BenchmarkType.TIME_STEP_FINE, time_step_fine, error)
    benchmark_results.write_to_archive(archive)
    benchmark_results.plot(archive)

    return benchmark_results

def new_benchmark_time_step_source(archive:Archive, signal_template:test_signal.TestSignal, frequency, state_properties:sim.manager.StateProperties, time_step_source):
    """
    **(benchmark for obsolete interpolation mode, might be useful to keep if such a mode is re-added in the future)**

    Runs a benchmark to test error induced by raising the size of the time step in the integrator, comparing the output state.

    Specifically, let :math:`(\\psi_{f,\\mathrm{d}t})_{m,t}` be the calculated state of the spin system, with magnetic number (`state_index`) :math:`m` at time :math:`t`, simulated with a dressing of :math:`f` with a fine time step of :math:`\\mathrm{d}t`. Let :math:`\\mathrm{d}t_0` be the first such time step in `time_step_fine` (generally the smallest one). Then the error :math:`e_{\\mathrm{d}t}` calculated by this benchmark is

    .. math::
        \\begin{align*}
            e_{\\mathrm{d}t} &= \\frac{1}{\\#t\\#f}\\sum_{t,f,m} |(\\psi_{f,\\mathrm{d}t})_{m,t} - (\\psi_{f,\\mathrm{d}t_0})_{m,t}|,
        \\end{align*}

    where :math:`\\#t` is the number of coarse time samples, :math:`\\#f` is the length of `frequency`.

    Parameters
    ----------
    archive : :class:`archive.Archive`
        Specifies where to save results and plots.
    signal_template : :class:`test_signal.TestSignal`
        A description of the signal to use for the environment during the simulation. For each entry in `time_step_fine`, this template is modified so that its :attr:`test_signal.TestSignal.time_properties.time_step_fine` is equal to that entry. All modified versions of the signal are then simulated for comparison.
    frequency : :class:`numpy.ndarray` of :class:`numpy.float64`
        The dressing frequencies being simulated in the benchmark.
    time_step_source : :class:`numpy.ndarray` of :class:`numpy.float64`
        An array of time steps to run the interpolation with. The accuracy of the simulation output with each of these values are then compared.

    Returns
    -------
    benchmark_results : :class:`BenchmarkResults`
        Contains the errors found by the benchmark.
    """
    time_step_source = np.asarray(time_step_source)
    state_output = []
    error = []

    signal = []
    for timeStep_source_instance in time_step_source:
        time_properties = test_signal.TimeProperties(signal_template.time_properties.time_step_coarse, signal_template.time_properties.time_step_fine, timeStep_source_instance)
        signal_instance = test_signal.TestSignal(signal_template.neural_pulses, signal_template.sinusoidal_noises, time_properties, False)
        signal += [signal_instance]

    simulation_manager = manager.SimulationManager(signal, frequency, archive, state_properties, state_output)
    simulation_manager.evaluate(False)

    for timeStep_source_index in range(time_step_source.size):
        error_temp = 0
        for frequency_index in range(frequency.size):
            state_difference = state_output[frequency_index + timeStep_source_index*frequency.size] - state_output[frequency_index]
            error_temp += np.sum(np.sqrt(np.real(np.conj(state_difference)*state_difference)))
        error += [error_temp/(frequency.size*state_output[0].size)]
    
    error = np.asarray(error)

    benchmark_results = BenchmarkResults(BenchmarkType.TIME_STEP_SOURCE, time_step_source, error)
    benchmark_results.write_to_archive(archive)
    benchmark_results.plot(archive)

    return benchmark_results

def new_benchmark_time_step_fine_frequency_drift(archive:Archive, signal_template:test_signal.TestSignal, time_step_fines, dressing_frequency):
    """
    Runs a benchmark to test error induced by raising the size of the time step in the integrator, comparing measured frequency coefficients.

    Parameters
    ----------
    archive : :class:`archive.Archive`
        Specifies where to save results and plots.
    signal_template : :class:`test_signal.TestSignal`
        A description of the signal to use for the environment during the simulation. For each entry in `time_step_fines`, this template is modified so that its :attr:`test_signal.TestSignal.time_properties.time_step_fine` is equal to that entry. All modified versions of the signal are then simulated for comparison.
    time_step_fines : :class:`numpy.ndarray` of :class:`numpy.float64`
        An array of time steps to run the simulations with. The accuracy of the simulation output with each of these values are then compared.
    dressing_frequency : :class:`numpy.ndarray` of :class:`numpy.float64`
        The dressing frequencies being simulated in the benchmark.
    """
    dressing_frequency = np.asarray(dressing_frequency)
    signal_template.get_amplitude()
    signal_template.get_frequency_amplitude()

    signals = []
    for time_step_fine in time_step_fines:
        time_properties = test_signal.TimeProperties(signal_template.time_properties.time_step_coarse, time_step_fine, signal_template.time_properties.time_step_source)
        signal = test_signal.TestSignal(signal_template.neural_pulses, signal_template.sinusoidal_noises, time_properties)
        signals += [signal]

    simulation_manager = manager.SimulationManager(signals, dressing_frequency, archive)
    simulation_manager.evaluate()

    frequency_drift = np.zeros(len(signals))
    for signal_index in range(len(signals)):
        for frequency_index, frequency in enumerate(dressing_frequency):
            frequency_drift[signal_index] += np.abs(simulation_manager.frequency_amplitude[frequency_index + signal_index*dressing_frequency.size] - signal_template.frequency_amplitude[signal_template.frequency == frequency])
        frequency_drift[signal_index] /= dressing_frequency.size

    benchmark_results = BenchmarkResults(BenchmarkType.TIME_STEP_FINE_FREQUENCY_DRIFT, np.asarray(time_step_fines), frequency_drift)
    benchmark_results.write_to_archive(archive)
    benchmark_results.plot(archive)

    return benchmark_results

def new_benchmark_mathematica(archive:Archive, time_step_fines, errors, execution_times):
    benchmark_results = BenchmarkResults(BenchmarkType.TIME_STEP_FINE, np.asarray(time_step_fines), errors)
    benchmark_results.write_to_archive(archive)
    benchmark_results.plot(archive)

    benchmark_results_execution_time = BenchmarkResults(BenchmarkType.TIME_STEP_FINE_EXECUTION_TIME, np.asarray(time_step_fines), execution_times)
    benchmark_results_execution_time.write_to_archive(archive)
    benchmark_results_execution_time.plot(archive)

    benchmark_results_execution_time_error = BenchmarkResults(BenchmarkType.EXECUTION_TIME_ERROR, execution_times, errors)
    benchmark_results_execution_time_error.write_to_archive(archive)
    benchmark_results_execution_time_error.plot(archive)

    return benchmark_results, benchmark_results_execution_time, benchmark_results_execution_time_error

def new_benchmark_scipy(archive:Archive, signal_template:test_signal.TestSignal, frequency, time_step_fine, state_properties:sim.manager.StateProperties):
    """
    Runs a benchmark to test error induced by raising the size of the time step in the integrator, comparing the output state.

    Specifically, let :math:`(\\psi_{f,\\mathrm{d}t})_{m,t}` be the calculated state of the spin system, with magnetic number (`state_index`) :math:`m` at time :math:`t`, simulated with a dressing of :math:`f` with a fine time step of :math:`\\mathrm{d}t`. Let :math:`\\mathrm{d}t_0` be the first such time step in `time_step_fine` (generally the smallest one). Then the error :math:`e_{\\mathrm{d}t}` calculated by this benchmark is

    .. math::
        \\begin{align*}
            e_{\\mathrm{d}t} &= \\frac{1}{\\#t\\#f}\\sum_{t,f,m} |(\\psi_{f,\\mathrm{d}t})_{m,t} - (\\psi_{f,\\mathrm{d}t_0})_{m,t}|,
        \\end{align*}

    where :math:`\\#t` is the number of coarse time samples, :math:`\\#f` is the length of `frequency`.

    Parameters
    ----------
    archive : :class:`archive.Archive`
        Specifies where to save results and plots.
    signal_template : :class:`test_signal.TestSignal`
        A description of the signal to use for the environment during the simulation. For each entry in `time_step_fine`, this template is modified so that its :attr:`test_signal.TestSignal.time_properties.time_step_fine` is equal to that entry. All modified versions of the signal are then simulated for comparison.
    frequency : :class:`numpy.ndarray` of :class:`numpy.float64`
        The dressing frequencies being simulated in the benchmark.
    time_step_fine : :class:`numpy.ndarray` of :class:`numpy.float64`
        An array of time steps to run the simulations with. The accuracy of the simulation output with each of these values are then compared.

    Returns
    -------
    benchmark_results : :class:`BenchmarkResults`
        Contains the errors found by the benchmark.
    """
    Jx = (1/math.sqrt(2))*np.asarray(
        [
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0]
        ],
        dtype = np.complex128
    )
    Jy = (1/math.sqrt(2))*np.asarray(
        [
            [ 0, -1j,   0],
            [1j,   0, -1j],
            [ 0,  1j,   0]
        ],
        dtype = np.complex128
    )
    Jz = np.asarray(
        [
            [1, 0,  0],
            [0, 0,  0],
            [0, 0, -1]
        ],
        dtype = np.complex128
    )
    Q = (1/3)*np.asarray(
        [
            [1,  0, 0],
            [0, -2, 0],
            [0,  0, 1]
        ],
        dtype = np.complex128
    )


    time_step_fine = np.asarray(time_step_fine)
    state_output = []
    error = []

    signal = []
    # source_properties = []

    execution_time_output = []

    print("Idx\tCmp\tTm\tdTm")
    execution_time_end_points = np.empty(2)
    execution_time_end_points[0] = tm.time()
    execution_time_end_points[1] = execution_time_end_points[0]
    simulation_index = 0
    time = np.arange(0e-3, 1e-1, 5e-7, dtype = np.float64)
    for time_step_fine_instance in time_step_fine:
        source_properties = manager.SourceProperties(signal_template, state_properties, bias_amplitude = 700e3)

        for frequency_instance in frequency:
            def evaluate_dressing(time, dressing):
                source_properties.evaluate_dressing(time, frequency_instance, dressing)

            # def evaluate_dressing(time, dressing):
            #     dressing[0] = 2*frequency_instance*math.cos(math.tau*700e3*time)
            #     dressing[1] = 0
            #     dressing[2] = 700e3

            def derivative(time, state):
                dressing = np.empty(4, np.float64)
                evaluate_dressing(time, dressing)
                matrix = -1j*math.tau*(dressing[0]*Jx + dressing[1]*Jy + dressing[2]*Jz + dressing[3]*Q)
                return np.matmul(matrix, state)

            results = scipy.integrate.solve_ivp(derivative, [0e-3, 1e-1], state_properties.state_init, t_eval = time, max_step = time_step_fine_instance)
            state_output += [np.transpose(results.y)]
            
            print(f"{simulation_index:4d}\t{100*(simulation_index + 1)/(len(frequency)*len(time_step_fine)):3.0f}%\t{tm.time() - execution_time_end_points[0]:3.0f}s\t{tm.time() - execution_time_end_points[0]:2.3f}s")

            execution_time_output += [tm.time() - execution_time_end_points[1]]

            simulation_index += 1
            execution_time_end_points[1] = tm.time()

    for time_step_fine_index in range(time_step_fine.size):
        error_temp = 0
        for frequency_index in range(frequency.size):
            state_difference = state_output[frequency_index + time_step_fine_index*frequency.size] - state_output[frequency_index]

            error_temp += np.sum(np.sqrt(np.real(np.conj(state_difference)*state_difference)))
        error += [error_temp/(frequency.size*state_output[0].size)]
    
    error = np.asarray(error)

    benchmark_results = BenchmarkResults(BenchmarkType.TIME_STEP_FINE, time_step_fine, error)
    benchmark_results.write_to_archive(archive)
    benchmark_results.plot(archive)

    benchmark_results_execution_time = BenchmarkResults(BenchmarkType.TIME_STEP_FINE_EXECUTION_TIME, np.asarray(time_step_fine), np.asarray(execution_time_output))
    benchmark_results_execution_time.write_to_archive(archive)
    benchmark_results_execution_time.plot(archive)

    benchmark_results_execution_time_error = BenchmarkResults(BenchmarkType.EXECUTION_TIME_ERROR, np.asarray(execution_time_output), error)
    benchmark_results_execution_time_error.write_to_archive(archive)
    benchmark_results_execution_time_error.plot(archive)

    

    return benchmark_results, benchmark_results_execution_time, benchmark_results_execution_time_error

def new_benchmark_spinsim(archive:Archive, signal_template:test_signal.TestSignal, frequency, time_step_fine, state_properties:sim.manager.StateProperties):
    """
    Runs a benchmark to test error induced by raising the size of the time step in the integrator, comparing the output state.

    Specifically, let :math:`(\\psi_{f,\\mathrm{d}t})_{m,t}` be the calculated state of the spin system, with magnetic number (`state_index`) :math:`m` at time :math:`t`, simulated with a dressing of :math:`f` with a fine time step of :math:`\\mathrm{d}t`. Let :math:`\\mathrm{d}t_0` be the first such time step in `time_step_fine` (generally the smallest one). Then the error :math:`e_{\\mathrm{d}t}` calculated by this benchmark is

    .. math::
        \\begin{align*}
            e_{\\mathrm{d}t} &= \\frac{1}{\\#t\\#f}\\sum_{t,f,m} |(\\psi_{f,\\mathrm{d}t})_{m,t} - (\\psi_{f,\\mathrm{d}t_0})_{m,t}|,
        \\end{align*}

    where :math:`\\#t` is the number of coarse time samples, :math:`\\#f` is the length of `frequency`.

    Parameters
    ----------
    archive : :class:`archive.Archive`
        Specifies where to save results and plots.
    signal_template : :class:`test_signal.TestSignal`
        A description of the signal to use for the environment during the simulation. For each entry in `time_step_fine`, this template is modified so that its :attr:`test_signal.TestSignal.time_properties.time_step_fine` is equal to that entry. All modified versions of the signal are then simulated for comparison.
    frequency : :class:`numpy.ndarray` of :class:`numpy.float64`
        The dressing frequencies being simulated in the benchmark.
    time_step_fine : :class:`numpy.ndarray` of :class:`numpy.float64`
        An array of time steps to run the simulations with. The accuracy of the simulation output with each of these values are then compared.

    Returns
    -------
    benchmark_results : :class:`BenchmarkResults`
        Contains the errors found by the benchmark.
    """
    time_step_fine = np.asarray(time_step_fine)
    state_output = []
    error = []

    signal = []
    execution_time_output = []
    for time_step_fine_index in time_step_fine:
        time_properties = test_signal.TimeProperties(signal_template.time_properties.time_step_coarse, time_step_fine_index, signal_template.time_properties.time_step_source)
        signal_instance = test_signal.TestSignal(signal_template.neural_pulses, signal_template.sinusoidal_noises, time_properties, False)
        signal += [signal_instance]

    simulation_manager = manager.SimulationManager(signal, frequency, archive, state_properties, state_output, execution_time_output = execution_time_output, bias_amplitude = 700e3)
    simulation_manager.evaluate(False)

    for time_step_fine_index in range(time_step_fine.size):
        error_temp = 0
        for frequency_index in range(frequency.size):
            state_difference = state_output[frequency_index + time_step_fine_index*frequency.size] - state_output[frequency_index]

            error_temp += np.sum(np.sqrt(np.real(np.conj(state_difference)*state_difference)))
        error += [error_temp/(frequency.size*state_output[0].size)]
    
    execution_time_output = execution_time_output[1::2]
    error = np.asarray(error)

    benchmark_results = BenchmarkResults(BenchmarkType.TIME_STEP_FINE, time_step_fine, error)
    benchmark_results.write_to_archive(archive)
    benchmark_results.plot(archive)

    benchmark_results_execution_time = BenchmarkResults(BenchmarkType.TIME_STEP_FINE_EXECUTION_TIME, np.asarray(time_step_fine), np.asarray(execution_time_output))
    benchmark_results_execution_time.write_to_archive(archive)
    benchmark_results_execution_time.plot(archive)

    benchmark_results_execution_time_error = BenchmarkResults(BenchmarkType.EXECUTION_TIME_ERROR, np.asarray(execution_time_output), error)
    benchmark_results_execution_time_error.write_to_archive(archive)
    benchmark_results_execution_time_error.plot(archive)

    return benchmark_results

def plot_benchmark_comparison(archive:Archive, archive_times, legend, title):
    """
    Plots multiple benchmarks on one plot from previous archives.

    Parameters
    ----------
    archive : :class:`archive.Archive`
        Specifies the path to save the plot to.
    archive_times : :obj:`list` of :obj:`str`
        The identifiers of the archvies containing the benchmark results to be compared.
    legend : :obj:`list` of :obj:`str`
        Labels that describe what each of the benchmark result curves respresent.
    title : :obj:`str`
        What this comparison is trying to compare.
    """
    for benchmark_type in [BenchmarkType.TIME_STEP_FINE, BenchmarkType.TIME_STEP_FINE_EXECUTION_TIME, BenchmarkType.EXECUTION_TIME_ERROR]:
        plt.figure()
        for archive_index, archive_time in enumerate(archive_times):
            archive_previous = Archive(archive.archive_path[:-25], "")
            archive_previous.open_archive_file(archive_time)
            benchmark_results = BenchmarkResults.read_from_archive(archive_previous, benchmark_type = benchmark_type)
            benchmark_results.plot(None, False)
            archive_previous.close_archive_file(False)

            if archive_index != 0:
                if benchmark_type == BenchmarkType.TIME_STEP_FINE_EXECUTION_TIME:
                    benchmark_results.error /= 8
                    benchmark_results.plot(None, False)
                if benchmark_type == BenchmarkType.EXECUTION_TIME_ERROR:
                    benchmark_results.parameter /= 8
                    benchmark_results.plot(None, False)
        
        if benchmark_type == BenchmarkType.TIME_STEP_FINE:
            legend_real = [legend[0]] + legend[1::2]
        else:
            legend_real = legend
        plt.legend(legend_real)

        if archive:
            archive.write_plot(title, f"benchmark_comparison_time_{benchmark_type._value_}")

            archive_group_benchmark_results = archive.archive_file.require_group(f"benchmark_results/benchmark_comparison_time_{benchmark_type._value_}")
            archive_group_benchmark_results["archive_times"] = np.asarray(archive_times, dtype='|S32')
            archive_group_benchmark_results["legend"] = np.asarray(legend, dtype='|S32')
            archive_group_benchmark_results["title"] = np.asarray([title], dtype='|S32')
        plt.draw()

def new_benchmark_external_spinsim(archive:Archive, signal_template:test_signal.TestSignal, frequency, time_step_fine, state_properties:sim.manager.StateProperties, integration_method:spinsim.IntegrationMethod = None, use_rotating_frame:bool = None, number_of_squares = None, device:spinsim.Device = spinsim.Device.CUDA):
    """
    Runs a benchmark to test error induced by raising the size of the time step in the integrator, comparing the output state.

    Specifically, let :math:`(\\psi_{f,\\mathrm{d}t})_{m,t}` be the calculated state of the spin system, with magnetic number (`state_index`) :math:`m` at time :math:`t`, simulated with a dressing of :math:`f` with a fine time step of :math:`\\mathrm{d}t`. Let :math:`\\mathrm{d}t_0` be the first such time step in `time_step_fine` (generally the smallest one). Then the error :math:`e_{\\mathrm{d}t}` calculated by this benchmark is

    .. math::
        \\begin{align*}
            e_{\\mathrm{d}t} &= \\frac{1}{\\#t\\#f}\\sum_{t,f,m} |(\\psi_{f,\\mathrm{d}t})_{m,t} - (\\psi_{f,\\mathrm{d}t_0})_{m,t}|,
        \\end{align*}

    where :math:`\\#t` is the number of coarse time samples, :math:`\\#f` is the length of `frequency`.

    Parameters
    ----------
    archive : :class:`archive.Archive`
        Specifies where to save results and plots.
    signal_template : :class:`test_signal.TestSignal`
        A description of the signal to use for the environment during the simulation. For each entry in `time_step_fine`, this template is modified so that its :attr:`test_signal.TestSignal.time_properties.time_step_fine` is equal to that entry. All modified versions of the signal are then simulated for comparison.
    frequency : :class:`numpy.ndarray` of :class:`numpy.float64`
        The dressing frequencies being simulated in the benchmark.
    time_step_fine : :class:`numpy.ndarray` of :class:`numpy.float64`
        An array of time steps to run the simulations with. The accuracy of the simulation output with each of these values are then compared.

    Returns
    -------
    benchmark_results : :class:`BenchmarkResults`
        Contains the errors found by the benchmark.
    """
    time_step_fine = np.asarray(time_step_fine)
    state_output = []

    signal = []
    execution_time_output = []
    for time_step_fine_index in time_step_fine:
        time_properties = test_signal.TimeProperties(signal_template.time_properties.time_step_coarse, time_step_fine_index, signal_template.time_properties.time_step_source)
        signal_instance = test_signal.TestSignal(signal_template.neural_pulses, signal_template.sinusoidal_noises, time_properties, False)
        signal += [signal_instance]
    if integration_method:
        simulation_manager = manager.SimulationManager(signal, frequency, archive, state_properties, state_output, execution_time_output = execution_time_output, bias_amplitude = 700e3, integration_method = integration_method, use_rotating_frame = use_rotating_frame)
    elif number_of_squares:
        simulation_manager = manager.SimulationManager(signal, frequency, archive, state_properties, state_output, execution_time_output = execution_time_output, bias_amplitude = 700e3, number_of_squares = [number_of_squares])
    else:
        simulation_manager = manager.SimulationManager(signal, frequency, archive, state_properties, state_output, execution_time_output = execution_time_output, bias_amplitude = 700e3, number_of_squares = [32])
    simulation_manager.evaluate(False)

    archive_group = archive.archive_file.require_group("benchmark_results/benchmark_external")
    if integration_method:
        archive_group.attrs["name"] = f"IM = {integration_method.name}, RF = {use_rotating_frame}"
    elif number_of_squares:
        archive_group.attrs["name"] = f"TC = {number_of_squares}"
    else:
        if device == spinsim.Device.CPU:
            archive_group.attrs["name"] = "spinsim (CPU)"
        else:
            archive_group.attrs["name"] = "spinsim"
    for state_index, state in enumerate(state_output):
        if (state_index % 2) == 1:
            archive_group[f"state{int(np.floor(state_index/2)):d}"] = state
            archive_group[f"state{int(np.floor(state_index/2)):d}"].attrs["time_step_fine"] = time_step_fine[int(np.floor(state_index/2))]
            archive_group[f"state{int(np.floor(state_index/2)):d}"].attrs["execution_time"] = execution_time_output[state_index]

def new_benchmark_external_scipy(archive:Archive, signal_template:test_signal.TestSignal, frequency, time_step_fine, state_properties:sim.manager.StateProperties):
    """
    Runs a benchmark to test error induced by raising the size of the time step in the integrator, comparing the output state.

    Specifically, let :math:`(\\psi_{f,\\mathrm{d}t})_{m,t}` be the calculated state of the spin system, with magnetic number (`state_index`) :math:`m` at time :math:`t`, simulated with a dressing of :math:`f` with a fine time step of :math:`\\mathrm{d}t`. Let :math:`\\mathrm{d}t_0` be the first such time step in `time_step_fine` (generally the smallest one). Then the error :math:`e_{\\mathrm{d}t}` calculated by this benchmark is

    .. math::
        \\begin{align*}
            e_{\\mathrm{d}t} &= \\frac{1}{\\#t\\#f}\\sum_{t,f,m} |(\\psi_{f,\\mathrm{d}t})_{m,t} - (\\psi_{f,\\mathrm{d}t_0})_{m,t}|,
        \\end{align*}

    where :math:`\\#t` is the number of coarse time samples, :math:`\\#f` is the length of `frequency`.

    Parameters
    ----------
    archive : :class:`archive.Archive`
        Specifies where to save results and plots.
    signal_template : :class:`test_signal.TestSignal`
        A description of the signal to use for the environment during the simulation. For each entry in `time_step_fine`, this template is modified so that its :attr:`test_signal.TestSignal.time_properties.time_step_fine` is equal to that entry. All modified versions of the signal are then simulated for comparison.
    frequency : :class:`numpy.ndarray` of :class:`numpy.float64`
        The dressing frequencies being simulated in the benchmark.
    time_step_fine : :class:`numpy.ndarray` of :class:`numpy.float64`
        An array of time steps to run the simulations with. The accuracy of the simulation output with each of these values are then compared.

    Returns
    -------
    benchmark_results : :class:`BenchmarkResults`
        Contains the errors found by the benchmark.
    """
    if state_properties.spin_quantum_number == spinsim.SpinQuantumNumber.ONE:
        Jx = (1/math.sqrt(2))*np.asarray(
            [
                [0, 1, 0],
                [1, 0, 1],
                [0, 1, 0]
            ],
            dtype = np.complex128
        )
        Jy = (1/math.sqrt(2))*np.asarray(
            [
                [ 0, -1j,   0],
                [1j,   0, -1j],
                [ 0,  1j,   0]
            ],
            dtype = np.complex128
        )
        Jz = np.asarray(
            [
                [1, 0,  0],
                [0, 0,  0],
                [0, 0, -1]
            ],
            dtype = np.complex128
        )
        Q = (1/3)*np.asarray(
            [
                [1,  0, 0],
                [0, -2, 0],
                [0,  0, 1]
            ],
            dtype = np.complex128
        )
    else:
        Jx = (1/2)*np.asarray(
            [
                [0, 1],
                [1, 0]
            ],
            dtype = np.complex128
        )
        Jy = (1/2)*np.asarray(
            [
                [ 0, -1j],
                [1j,   0]
            ],
            dtype = np.complex128
        )
        Jz = (1/2)*np.asarray(
            [
                [1,  0],
                [0, -1]
            ],
            dtype = np.complex128
        )


    time_step_fine = np.asarray(time_step_fine)
    state_output = []
    error = []

    signal = []
    # source_properties = []

    execution_time_output = []

    print("Idx\tCmp\tTm\tdTm")
    execution_time_end_points = np.empty(2)
    execution_time_end_points[0] = tm.time()
    execution_time_end_points[1] = execution_time_end_points[0]
    simulation_index = 0
    time = np.arange(0e-3, 1e-1, 5e-7, dtype = np.float64)
    for time_step_fine_instance in time_step_fine:
        source_properties = manager.SourceProperties(signal_template, state_properties, bias_amplitude = 700e3)

        for frequency_instance in frequency:
            # def evaluate_dressing(time, dressing):
            #     source_properties.evaluate_dressing(time, frequency_instance, dressing)

            # # def evaluate_dressing(time, dressing):
            # #     dressing[0] = 2*frequency_instance*math.cos(math.tau*700e3*time)
            # #     dressing[1] = 0
            # #     dressing[2] = 700e3
            
            # if state_properties.spin_quantum_number == spinsim.SpinQuantumNumber.ONE:
            #     def derivative(time, state):
            #         dressing = np.empty(4, np.float64)
            #         evaluate_dressing(time, dressing)
            #         matrix = -1j*math.tau*(dressing[0]*Jx + dressing[1]*Jy + dressing[2]*Jz + dressing[3]*Q)
            #         return np.matmul(matrix, state)
            # else:
            #     def derivative(time, state):
            #         dressing = np.empty(4, np.float64)
            #         evaluate_dressing(time, dressing)
            #         matrix = -1j*math.tau*(dressing[0]*Jx + dressing[1]*Jy + dressing[2]*Jz)
            #         return np.matmul(matrix, state)
            f_bias = 700e3
            f_dressing = 1000
            f_neural = 70
            t_pulse = 0.02333333
            def ham_x(time, args):
                return math.tau*2*f_dressing*math.cos(math.tau*f_bias*time)
            def ham_y(time, args):
                return 0.0
            def ham_z(time, args):
                neural = 0
                if time > t_pulse and time < t_pulse + 1/f_dressing:
                    neural = f_neural*math.sin(math.tau*f_dressing*(time - t_pulse))
                return math.tau*(f_bias + neural)
            def ham_q(time, args):
                return 0.0
            def derivative(time, state):
                matrix = -1j*(ham_x(time, 0)*Jx + ham_y(time, 0)*Jy + ham_z(time, 0)*Jz + ham_q(time, 0)*Q)
                return np.matmul(matrix, state)
            results = scipy.integrate.solve_ivp(derivative, [0e-3, 1e-1], state_properties.state_init, t_eval = time, max_step = time_step_fine_instance)
            state_output += [np.transpose(results.y)]

            state = np.transpose(results.y)
            spin = np.empty((state.shape[0], 3), np.float64)
            for spin_index, state_instance in enumerate(state):
                spin[spin_index, 0] = np.matmul(np.matmul(state[spin_index, :].conj().T, Jx), state[spin_index, :]).real
                spin[spin_index, 1] = np.matmul(np.matmul(state[spin_index, :].conj().T, Jy), state[spin_index, :]).real
                spin[spin_index, 2] = np.matmul(np.matmul(state[spin_index, :].conj().T, Jz), state[spin_index, :]).real

            print(f"{simulation_index:4d}\t{100*(simulation_index + 1)/(len(frequency)*len(time_step_fine)):3.0f}%\t{tm.time() - execution_time_end_points[0]:3.0f}s\t{tm.time() - execution_time_end_points[1]:2.3f}s")

            execution_time_output += [tm.time() - execution_time_end_points[1]]

            simulation_index += 1
            execution_time_end_points[1] = tm.time()

    archive_group = archive.archive_file.require_group("benchmark_results/benchmark_external")
    archive_group.attrs["name"] = "SciPy"
    for state_index, state in enumerate(state_output):
        archive_group[f"state{state_index:d}"] = state
        archive_group[f"state{state_index:d}"].attrs["time_step_fine"] = time_step_fine[state_index]
        archive_group[f"state{state_index:d}"].attrs["execution_time"] = execution_time_output[state_index]

def new_benchmark_external_qutip(archive:Archive, signal_template:test_signal.TestSignal, frequency, time_step_fine, state_properties:sim.manager.StateProperties):
    """
    Runs a benchmark to test error induced by raising the size of the time step in the integrator, comparing the output state.

    Specifically, let :math:`(\\psi_{f,\\mathrm{d}t})_{m,t}` be the calculated state of the spin system, with magnetic number (`state_index`) :math:`m` at time :math:`t`, simulated with a dressing of :math:`f` with a fine time step of :math:`\\mathrm{d}t`. Let :math:`\\mathrm{d}t_0` be the first such time step in `time_step_fine` (generally the smallest one). Then the error :math:`e_{\\mathrm{d}t}` calculated by this benchmark is

    .. math::
        \\begin{align*}
            e_{\\mathrm{d}t} &= \\frac{1}{\\#t\\#f}\\sum_{t,f,m} |(\\psi_{f,\\mathrm{d}t})_{m,t} - (\\psi_{f,\\mathrm{d}t_0})_{m,t}|,
        \\end{align*}

    where :math:`\\#t` is the number of coarse time samples, :math:`\\#f` is the length of `frequency`.

    Parameters
    ----------
    archive : :class:`archive.Archive`
        Specifies where to save results and plots.
    signal_template : :class:`test_signal.TestSignal`
        A description of the signal to use for the environment during the simulation. For each entry in `time_step_fine`, this template is modified so that its :attr:`test_signal.TestSignal.time_properties.time_step_fine` is equal to that entry. All modified versions of the signal are then simulated for comparison.
    frequency : :class:`numpy.ndarray` of :class:`numpy.float64`
        The dressing frequencies being simulated in the benchmark.
    time_step_fine : :class:`numpy.ndarray` of :class:`numpy.float64`
        An array of time steps to run the simulations with. The accuracy of the simulation output with each of these values are then compared.

    Returns
    -------
    benchmark_results : :class:`BenchmarkResults`
        Contains the errors found by the benchmark.
    """
    if state_properties.spin_quantum_number == spinsim.SpinQuantumNumber.ONE:
        Jx = qutip.spin_Jx(1)
        Jy = qutip.spin_Jy(1)
        Jz = qutip.spin_Jz(1)
        Q = qutip.qdiags([1/3, -2/3, 1/3], 0)
    else:
        Jx = (1/2)*np.asarray(
            [
                [0, 1],
                [1, 0]
            ],
            dtype = np.complex128
        )
        Jy = (1/2)*np.asarray(
            [
                [ 0, -1j],
                [1j,   0]
            ],
            dtype = np.complex128
        )
        Jz = (1/2)*np.asarray(
            [
                [1,  0],
                [0, -1]
            ],
            dtype = np.complex128
        )


    time_step_fine = np.asarray(time_step_fine)
    state_output = []
    error = []

    signal = []
    # source_properties = []

    execution_time_output = []

    print("Idx\tCmp\tTm\tdTm")
    execution_time_end_points = np.empty(2)
    execution_time_end_points[0] = tm.time()
    execution_time_end_points[1] = execution_time_end_points[0]
    simulation_index = 0
    time = np.arange(0e-3, 1e-1, 5e-7, dtype = np.float64)
    for time_step_fine_instance in time_step_fine:
        source_properties = manager.SourceProperties(signal_template, state_properties, bias_amplitude = 700e3)

        for frequency_instance in frequency:
            f_bias = 700e3
            f_dressing = 1000
            f_neural = 70
            t_pulse = 0.02333333
            # def ham_x(time, args):
            #     return math.tau*2*f_dressing*math.cos(math.tau*f_bias*time)
            # def ham_y(time, args):
            #     return 0.0
            # def ham_z(time, args):
            #     neural = 0
            #     if time > t_pulse and time < t_pulse + 1/f_dressing:
            #         neural = f_neural*math.sin(math.tau*f_dressing*(time - t_pulse))
            #     return math.tau*(f_bias + neural)
            # def ham_q(time, args):
            #     return 0.0
            
            ham_x = f"2*pi*2*{f_dressing}*cos(2*pi*{f_bias}*t)"
            ham_y = f"0.0"
            ham_z = f"2*pi*({f_bias} + ((t > {t_pulse}) and (t < {t_pulse} + 1/{f_dressing}))*{f_neural}*sin(2*pi*{f_dressing}*(t - {t_pulse})))"
            # (t > {t_pulse}) and (t < {t_pulse} + 1/{f_dressing}))*{f_neural}*sin(2*pi*{f_dressing}*(t - {t_pulse})
            ham_q = f"0.0"
            qutip.sesolve([[Jx, ham_x], [Jy, ham_y], [Jz, ham_z],[Q, ham_q]], qutip.qutrit_basis()[0], time, e_ops=None, args = None, options=qutip.Options(first_step = time_step_fine_instance, max_step = time_step_fine_instance, store_states = True, nsteps = 1e6, atol = 1e-8/(5e-7/time_step_fine_instance), rtol = 1e-8/(5e-7/time_step_fine_instance)), progress_bar=None, _safe_mode=True)
            execution_time_end_points[1] = tm.time()
            results = qutip.sesolve([[Jx, ham_x], [Jy, ham_y], [Jz, ham_z],[Q, ham_q]], qutip.qutrit_basis()[0], time, e_ops=None, args = None, options=qutip.Options(first_step = time_step_fine_instance, max_step = time_step_fine_instance, store_states = True, nsteps = 1e6, atol = 1e-8/(5e-7/time_step_fine_instance), rtol = 1e-8/(5e-7/time_step_fine_instance), rhs_reuse=True), progress_bar=None, _safe_mode=True)
            # atol = 1e-16, rtol = 1e-16
            # nsteps = 1000*(5e-7/time_step_fine_instance)
            # [Jx, Jy, Jz]
            state = np.empty((time.size, 3), np.complex128)
            for state_index, state_instance in enumerate(results.states):
                state[state_index, :] = state_instance.full()[:, 0]
            state_output += [state]
            expect = results.expect
            # state = np.asarray(results.states)

            # print(type(state_output[0][0, :]))

            # state = args["vec"]
            # state_output += [state]
            
            print(f"{simulation_index:4d}\t{100*(simulation_index + 1)/(len(frequency)*len(time_step_fine)):3.0f}%\t{tm.time() - execution_time_end_points[0]:3.0f}s\t{tm.time() - execution_time_end_points[1]:2.3f}s")

            execution_time_output += [tm.time() - execution_time_end_points[1]]

            simulation_index += 1
            execution_time_end_points[1] = tm.time()

    archive_group = archive.archive_file.require_group("benchmark_results/benchmark_external")
    archive_group.attrs["name"] = "QuTip"
    for state_index, state in enumerate(state_output):
        archive_group[f"state{state_index:d}"] = state
        archive_group[f"state{state_index:d}"].attrs["time_step_fine"] = time_step_fine[state_index]
        archive_group[f"state{state_index:d}"].attrs["execution_time"] = execution_time_output[state_index]

def new_benchmark_true_external_spinsim(archive:Archive, frequency, time_step_fine, device:spinsim.Device, spin_quantum_number:spinsim.SpinQuantumNumber = None, integration_method:spinsim.IntegrationMethod = None, use_rotating_frame:bool = None):
    """
    Runs a benchmark to test error induced by raising the size of the time step in the integrator, comparing the output state.

    Specifically, let :math:`(\\psi_{f,\\mathrm{d}t})_{m,t}` be the calculated state of the spin system, with magnetic number (`state_index`) :math:`m` at time :math:`t`, simulated with a dressing of :math:`f` with a fine time step of :math:`\\mathrm{d}t`. Let :math:`\\mathrm{d}t_0` be the first such time step in `time_step_fine` (generally the smallest one). Then the error :math:`e_{\\mathrm{d}t}` calculated by this benchmark is

    .. math::
        \\begin{align*}
            e_{\\mathrm{d}t} &= \\frac{1}{\\#t\\#f}\\sum_{t,f,m} |(\\psi_{f,\\mathrm{d}t})_{m,t} - (\\psi_{f,\\mathrm{d}t_0})_{m,t}|,
        \\end{align*}

    where :math:`\\#t` is the number of coarse time samples, :math:`\\#f` is the length of `frequency`.

    Parameters
    ----------
    archive : :class:`archive.Archive`
        Specifies where to save results and plots.
    signal_template : :class:`test_signal.TestSignal`
        A description of the signal to use for the environment during the simulation. For each entry in `time_step_fine`, this template is modified so that its :attr:`test_signal.TestSignal.time_properties.time_step_fine` is equal to that entry. All modified versions of the signal are then simulated for comparison.
    frequency : :class:`numpy.ndarray` of :class:`numpy.float64`
        The dressing frequencies being simulated in the benchmark.
    time_step_fine : :class:`numpy.ndarray` of :class:`numpy.float64`
        An array of time steps to run the simulations with. The accuracy of the simulation output with each of these values are then compared.

    Returns
    -------
    benchmark_results : :class:`BenchmarkResults`
        Contains the errors found by the benchmark.
    """

    time_step_fine = np.asarray(time_step_fine)
    state_output = []
    error = []

    signal = []
    # source_properties = []

    execution_time_output = []

    print("Idx\tCmp\tTm\tdTm")
    execution_time_end_points = np.empty(2)
    execution_time_end_points[0] = tm.time()
    execution_time_end_points[1] = execution_time_end_points[0]
    simulation_index = 0
    time = np.arange(0e-3, 1e-1, 5e-7, dtype = np.float64)
    for time_step_fine_instance in time_step_fine:
        for frequency_instance in frequency:
            f_bias = 700e3
            f_dressing = 1000
            f_neural = 70
            t_pulse = 0.02333333
            # def ham_x(time, args):
            #     return math.tau*2*f_dressing*math.cos(math.tau*f_bias*time)
            # def ham_y(time, args):
            #     return 0.0
            # def ham_z(time, args):
            #     neural = 0
            #     if time > t_pulse and time < t_pulse + 1/f_dressing:
            #         neural = f_neural*math.sin(math.tau*f_dressing*(time - t_pulse))
            #     return math.tau*(f_bias + neural)
            # def ham_q(time, args):
            #     return 0.0

            def sample_field(time, _, field):
                field[0] = math.tau*2*f_dressing*math.cos(math.tau*f_bias*time)
                field[1] = 0.0
                field[2] = math.tau*(f_bias + ((time > t_pulse) and (time < t_pulse + 1/f_dressing))*f_neural*math.sin(math.tau*f_dressing*(time - t_pulse)))
                field[3] = 0.0
            if integration_method:
                integration_method_use = integration_method
                use_rotating_frame_use = use_rotating_frame
                spin_quantum_number_use = spin_quantum_number
                if spin_quantum_number == spinsim.SpinQuantumNumber.HALF:
                    initial_state_use = np.array([1, 0], np.complex128)
                else:
                    initial_state_use = np.array([1, 0, 0], np.complex128)
            else:
                integration_method_use = spinsim.IntegrationMethod.MAGNUS_CF4
                use_rotating_frame_use = True
                initial_state_use = np.array([1, 0, 0], np.complex128)
                spin_quantum_number_use = spinsim.SpinQuantumNumber.ONE
            
            simulator = spinsim.Simulator(sample_field, spin_quantum_number_use, device, integration_method = integration_method_use, use_rotating_frame = use_rotating_frame_use)
            simulator.evaluate(0, 1e-1, time_step_fine_instance, 5e-7, initial_state_use)
            execution_time_end_points[1] = tm.time()
            results = simulator.evaluate(0, 1e-1, time_step_fine_instance, 5e-7, initial_state_use)
            state = results.state
            state_output += [state]
            spin = results.spin
            
            print(f"{simulation_index:4d}\t{100*(simulation_index + 1)/(len(frequency)*len(time_step_fine)):3.0f}%\t{tm.time() - execution_time_end_points[0]:3.0f}s\t{tm.time() - execution_time_end_points[1]:2.3f}s")

            execution_time_output += [tm.time() - execution_time_end_points[1]]

            simulation_index += 1
            execution_time_end_points[1] = tm.time()

    archive_group = archive.archive_file.require_group("benchmark_results/benchmark_external")
    if integration_method:
        name = f"IM = {integration_method.name}, RF = {use_rotating_frame}"
    else:
        name = "spinsim"
        if device == spinsim.Device.CPU:
            name = f"{name} (CPU)"
    archive_group.attrs["name"] = name
    for state_index, state in enumerate(state_output):
        archive_group[f"state{state_index:d}"] = state
        archive_group[f"state{state_index:d}"].attrs["time_step_fine"] = time_step_fine[state_index]
        archive_group[f"state{state_index:d}"].attrs["execution_time"] = execution_time_output[state_index]

def new_benchmark_external_evaluation(archive:Archive, archive_times, reference_name:str = "SciPy", is_external:bool = True, is_trotter = False, title:str = None):

    # === Load external states ===
    states = []
    time_step_fines = []
    execution_times = []
    names = []

    for archive_time in archive_times:
        archive_previous = Archive(archive.archive_path[:-25], "")
        archive_previous.open_archive_file(archive_time)
        
        archive_previous_group = archive_previous.archive_file.require_group("benchmark_results/benchmark_external")

        name = archive_previous_group.attrs["name"]
        names += [name]

        external_states = []
        external_time_step_fines = []
        external_execution_times = []
        
        simulation_index = 0
        while f"state{simulation_index}" in archive_previous_group:
            if name == "Mathematica":
                h5py.get_config().complex_names = ("Re", "Im")
            else:
                h5py.get_config().complex_names = ("r", "i")
            
            state = np.asarray(archive_previous_group[f"state{simulation_index}"], np.cdouble)
            external_states += [state]
            time_step_fine = archive_previous_group[f"state{simulation_index}"].attrs["time_step_fine"]
            external_time_step_fines += [time_step_fine]
            execution_time = archive_previous_group[f"state{simulation_index}"].attrs["execution_time"]
            external_execution_times += [execution_time]
            simulation_index += 1

        states += [external_states]
        time_step_fines += [np.asarray(external_time_step_fines, np.double)]
        execution_times += [np.asarray(external_execution_times, np.double)]

        archive_previous.close_archive_file(False)

    # === Find reference state ===
    reference_index = 0
    for name_index, name in enumerate(names):
        if name == reference_name:
            reference_index = name_index
            break
    # print(states[reference_index])
    state_reference = states[reference_index][0]

    # === Calculate errors ===
    errors = []
    number_of_samples = 200000
    # number_of_samples = 2000
    for external_states in states:
        external_errors = []
        for state in external_states:
            state_difference = state[0:number_of_samples] - state_reference[0:number_of_samples]
            # error = np.sum(np.sqrt(np.real(np.conj(state_difference)*state_difference)))/(3*number_of_samples)
            error = np.sqrt(np.sum(np.real(np.conj(state_difference)*state_difference)))/(number_of_samples)
            external_errors += [error]
        errors += [np.asarray(external_errors, np.double)]

    # === Save to file ===
    archive_group_external_evaluation = archive.archive_file.require_group("benchmark_results/benchmark_external_evaluation")
    archive_group_external_evaluation.attrs["reference"] = archive_times[reference_index]
    archive_group_external_evaluation.attrs["reference_name"] = reference_name
    for name, external_time_step_fines, external_execution_times, external_errors, archive_time in zip(names, time_step_fines, execution_times, errors, archive_times):
        if name in archive_group_external_evaluation:
            archive_group_name = archive_group_external_evaluation.require_group(name)
        else:
            name_index = 1
            while f"{name} {name_index}" in archive_group_external_evaluation:
                name_index += 1
            archive_group_name = archive_group_external_evaluation.require_group(f"{name} {name_index}")
        archive_group_name["time_step_fine"] = external_time_step_fines
        archive_group_name["execution_time"] = external_execution_times
        archive_group_name["errors"] = external_errors
        archive_group_name.attrs["archive"] = archive_time

    # === Plot legends ===
    # legend_legend = {
    #     "spinsim" : "ss",
    #     "AtomicPy" : "ap",
    #     "Mathematica" : "mm",
    #     "SciPy" : "sp",

    #     "IM = MAGNUS_CF4, RF = True" : "CF4 (RF)",
    #     "IM = MAGNUS_CF4, RF = False" : "CF4 (LF)",
    #     "IM = MIDPOINT_SAMPLE, RF = True" : "MP (RF)",
    #     "IM = MIDPOINT_SAMPLE, RF = False" : "MP (LF)",
    #     "IM = HALF_STEP, RF = True" : "HS (RF)",
    #     "IM = HALF_STEP, RF = False" : "HS (LF)",

    #     "TC = 4" : "4",
    #     "TC = 8" : "8",
    #     "TC = 12" : "12",
    #     "TC = 16" : "16",
    #     "TC = 20" : "20",
    #     "TC = 24" : "24",
    #     "TC = 28" : "28",
    #     "TC = 32" : "32",
    #     "TC = 36" : "36",
    #     "TC = 40" : "40",
    #     "TC = 44" : "44",
    #     "TC = 48" : "48",
    #     "TC = 52" : "52",
    #     "TC = 56" : "56",
    #     "TC = 60" : "60",
    #     "TC = 64" : "64"
    # }

    legend_legend = {
        "spinsim" : "spinsim",
        "spinsim (CPU)" : "spinsim (CPU)",
        "QuTip" : "QuTip",
        "AtomicPy" : "AtomicPy",
        "Mathematica" : "Mathematica",
        "SciPy" : "SciPy",

        "IM = MAGNUS_CF4, RF = True" : "Magnus CF4",
        "IM = MAGNUS_CF4, RF = False" : "Magnus CF4",
        "IM = MIDPOINT_SAMPLE, RF = True" : "Midpoint Euler",
        "IM = MIDPOINT_SAMPLE, RF = False" : "Midpoint Euler",
        "IM = HALF_STEP, RF = True" : "Heun Euler",
        "IM = HALF_STEP, RF = False" : "Heun Euler",

        "IM = EULER, RF = True" : "Midpoint Euler",
        "IM = EULER, RF = False" : "Midpoint Euler",
        "IM = HEUN, RF = True" : "Heun Euler",
        "IM = HEUN, RF = False" : "Heun Euler",

        "TC = 4" : "4",
        "TC = 8" : "8",
        "TC = 12" : "12",
        "TC = 16" : "16",
        "TC = 20" : "20",
        "TC = 24" : "24",
        "TC = 28" : "28",
        "TC = 32" : "32",
        "TC = 36" : "36",
        "TC = 40" : "40",
        "TC = 44" : "44",
        "TC = 48" : "48",
        "TC = 52" : "52",
        "TC = 56" : "56",
        "TC = 60" : "60",
        "TC = 64" : "64"
    }

    colour_legend = {
        "spinsim" : "ko",
        "spinsim (CPU)" : "k.",
        "QuTip" : "c^",
        "AtomicPy" : "y>",
        "Mathematica" : "ms",
        "SciPy" : "gP",

        "IM = MAGNUS_CF4, RF = True" : "ko",
        "IM = MAGNUS_CF4, RF = False" : "k.",
        "IM = MIDPOINT_SAMPLE, RF = True" : "rD",
        "IM = MIDPOINT_SAMPLE, RF = False" : "rd",
        "IM = HALF_STEP, RF = True" : "bX",
        "IM = HALF_STEP, RF = False" : "bx",

        "IM = EULER, RF = True" : "rD",
        "IM = EULER, RF = False" : "rd",
        "IM = HEUN, RF = True" : "bX",
        "IM = HEUN, RF = False" : "bx",

        "TC = 4" : ".-",
        "TC = 8" : ".-",
        "TC = 12" : ".-",
        "TC = 16" : ".-",
        "TC = 20" : ".-",
        "TC = 24" : ".-",
        "TC = 28" : ".-",
        "TC = 32" : ".-",
        "TC = 36" : ".-",
        "TC = 40" : ".-",
        "TC = 44" : ".-",
        "TC = 48" : ".-",
        "TC = 52" : ".-",
        "TC = 56" : ".-",
        "TC = 60" : ".-",
        "TC = 64" : ".-"
    }

    def draw_spin(spin_dimension = 3):
        if spin_dimension == 3:
            plt.text(
                0.97, 1.03,
                "Spin one",
                ha = "right", va = "bottom", transform = plt.gca().transAxes,
                bbox = {
                    "boxstyle" : "round",
                    "facecolor" : "c",
                    "alpha" : 0.5
                }
            )
        else:
            plt.text(
                0.97, 1.03,
                "Spin half",
                ha = "right", va = "bottom", transform = plt.gca().transAxes,
                bbox = {
                    "boxstyle" : "round",
                    "facecolor" : "y",
                    "alpha" : 0.5
                }
            )

    if title:
        title = f"{title}:\n"
    else:
        title = ""
    spin_dimension = state_reference.shape[1]
    # print(spin_dimension)

    # === Plot time step vs error ===
    error_max = 1e-3
    error_min = 1e-11
    plt.rcParams.update({'font.size': 12})
    # np.logical_and(error_min < external_errors, external_errors < error_max) = np.logical_and(error_min < external_errors, external_errors < error_max)
    legend = []
    legend_lines = []
    plt.figure()
    for name, external_time_step_fines, external_errors in zip(names, time_step_fines, errors):
        indices_keep = np.logical_and(error_min < external_errors, external_errors < error_max)
        have_started = False
        for index_keep_index in range(indices_keep.size - 1, -1, -1):
            if have_started:
                if not indices_keep[index_keep_index + 1]:
                    indices_keep[index_keep_index] = False
            else:
                if indices_keep[index_keep_index]:
                    have_started = True
        if is_external:
            plt.loglog(external_time_step_fines[indices_keep], external_errors[indices_keep], f"{colour_legend[name]}-")
            legend += [legend_legend[name]]
            legend_lines += [lns.Line2D([0], [0], 1, "-", colour_legend[name][0], colour_legend[name][1])]
        else:
            if name != reference_name:
                if "RF = True" in name:
                    plt.loglog(external_time_step_fines[indices_keep], external_errors[indices_keep], f"{colour_legend[name]}-")
                    legend += [f"{legend_legend[name]} (Rotating)"]
                    legend_lines += [lns.Line2D([0], [0], 1, "-", colour_legend[name][0], colour_legend[name][1])]
                else:
                    plt.loglog(external_time_step_fines[indices_keep], external_errors[indices_keep], f"{colour_legend[name]}--")
                    legend += [f"{legend_legend[name]} (Lab)"]
                    legend_lines += [lns.Line2D([0], [0], 1, "--", colour_legend[name][0], colour_legend[name][1])]
    if is_external:
        plt.legend(legend_lines, legend, loc = "upper left", ncol = 1)
    else:
        # legend += ["Rotating frame", "Lab frame"]
        # legend_lines += [
        #     lns.Line2D([0], [0], 1, "-", "grey", "o"),
        #     lns.Line2D([0], [0], 1, "--", "grey", ".")
        # ]
        legend_lines_transpose = []
        legend_transpose = []
        for double_index in range(2):
            for legend_index in range(0, len(legend_lines), 2):
                legend_transpose += [legend[legend_index + double_index]]
                legend_lines_transpose += [legend_lines[legend_index + double_index]]
        plt.legend(legend_lines_transpose, legend_transpose, loc = "upper right", ncol = 2)
    draw_spin(spin_dimension)
    plt.xlabel("Integration time step (s)")
    plt.ylabel("Error")
    plt.subplots_adjust(right=0.88, left = 0.01)
    plt.gca().invert_xaxis()
    plt.gca().yaxis.tick_right()
    plt.gca().yaxis.set_label_position("right")
    plt.gca().spines["left"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)
    # plt.tick_params(axis='y', which='right', labelleft='off', labelright='on')
    plt.grid()
    if archive:
        archive.write_plot(f"{title}Error vs integration time step", "benchmark_external_step_error")
    plt.draw()

    # === Plot time step vs execution time ===
    legend = []
    legend_lines = []
    plt.figure()
    for name, external_time_step_fines, external_execution_times, external_errors in zip(names, time_step_fines, execution_times, errors):
        indices_keep = np.logical_and(error_min < external_errors, external_errors < error_max)
        have_started = False
        for index_keep_index in range(indices_keep.size - 1, -1, -1):
            if have_started:
                if not indices_keep[index_keep_index + 1]:
                    indices_keep[index_keep_index] = False
            else:
                if indices_keep[index_keep_index]:
                    have_started = True
        if is_external:
            plt.loglog(external_time_step_fines[indices_keep], external_execution_times[indices_keep], f"{colour_legend[name]}-")
            legend += [legend_legend[name]]
            legend_lines += [lns.Line2D([0], [0], 1, "-", colour_legend[name][0], colour_legend[name][1])]
            if "spinsim" not in name and "QuTip" not in name:
                # plt.loglog(external_time_step_fines[np.logical_and(error_min < external_errors, external_errors < error_max)], external_execution_times[np.logical_and(error_min < external_errors, external_errors < error_max)]/4, f"{colour_legend[name]}--", alpha = 0.25)
                # legend += [f"{legend_legend[name]} (/4)"]
                plt.loglog(external_time_step_fines[indices_keep], external_execution_times[indices_keep]/8, f"{colour_legend[name]}:", alpha = 0.25)
                # legend += [f"{legend_legend[name]} (/8)"]
        else:
            if name != reference_name:
                if "RF = True" in name:
                    plt.loglog(external_time_step_fines[indices_keep], external_execution_times[indices_keep], f"{colour_legend[name]}-")
                    legend += [f"{legend_legend[name]} (Rotating)"]
                    legend_lines += [lns.Line2D([0], [0], 1, "-", colour_legend[name][0], colour_legend[name][1])]
                else:
                    plt.loglog(external_time_step_fines[indices_keep], external_execution_times[indices_keep], f"{colour_legend[name]}--")
                    legend += [f"{legend_legend[name]} (Lab)"]
                    legend_lines += [lns.Line2D([0], [0], 1, "--", colour_legend[name][0], colour_legend[name][1])]
    if is_external:
        legend += ["Ideal 8 threads"]
        legend_lines += [
            # lns.Line2D([0], [0], 1, "--", "grey"),
            lns.Line2D([0], [0], 1, ":", "grey")
        ]
        plt.legend(legend_lines, legend, loc = "upper left", ncol = 1)
    else:
        # legend += ["Rotating frame", "Lab frame"]
        # legend_lines += [
        #     lns.Line2D([0], [0], 1, "-", "grey", "o"),
        #     lns.Line2D([0], [0], 1, "--", "grey", ".")
        # ]
        legend_lines_transpose = []
        legend_transpose = []
        for double_index in range(2):
            for legend_index in range(0, len(legend_lines), 2):
                legend_transpose += [legend[legend_index + double_index]]
                legend_lines_transpose += [legend_lines[legend_index + double_index]]
        plt.legend(legend_lines_transpose, legend_transpose, loc = "upper right", ncol = 2)
        # plt.legend(legend_lines, legend, loc = "upper right", ncol = 1)
    draw_spin(spin_dimension)
    plt.xlabel("Integration time step (s)")
    plt.ylabel("Execution time (s)")
    plt.subplots_adjust(right=0.99)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)
    plt.grid()
    if archive:
        archive.write_plot(f"{title}Execution time vs integration time step", "benchmark_external_step_execution")
    plt.draw()

    # === Execution time vs error ===
    legend = []
    legend_lines = []
    plt.figure()
    for name, external_execution_times, external_errors in zip(names, execution_times, errors):
        indices_keep = np.logical_and(error_min < external_errors, external_errors < error_max)
        have_started = False
        for index_keep_index in range(indices_keep.size - 1, -1, -1):
            if have_started:
                if not indices_keep[index_keep_index + 1]:
                    indices_keep[index_keep_index] = False
            else:
                if indices_keep[index_keep_index]:
                    have_started = True
        if is_external:
            plt.loglog(external_execution_times[indices_keep], external_errors[indices_keep], f"{colour_legend[name]}-")
            legend += [legend_legend[name]]
            legend_lines += [lns.Line2D([0], [0], 1, "-", colour_legend[name][0], colour_legend[name][1])]
            if "spinsim" not in name and "QuTip" not in name:
                # plt.loglog(external_execution_times[np.logical_and(error_min < external_errors, external_errors < error_max)]/4, external_errors[np.logical_and(error_min < external_errors, external_errors < error_max)], f"{colour_legend[name]}--", alpha = 0.25)
                # legend += [f"{legend_legend[name]} (/4)"]
                plt.loglog(external_execution_times[indices_keep]/8, external_errors[indices_keep], f"{colour_legend[name]}:", alpha = 0.25)
                legend += ["(Ideal 8 threads)"]
                legend_lines += [lns.Line2D([0], [0], 1, ":", colour_legend[name][0], colour_legend[name][1], alpha = 0.25)]
                # legend += [f"{legend_legend[name]} (/8)"]
        else:
            if name != reference_name:
                if "RF = True" in name:
                    plt.loglog(external_execution_times[indices_keep], external_errors[indices_keep], f"{colour_legend[name]}-")
                    legend += [f"{legend_legend[name]} (Rotating)"]
                    legend_lines += [lns.Line2D([0], [0], 1, "-", colour_legend[name][0], colour_legend[name][1])]
                else:
                    plt.loglog(external_execution_times[indices_keep], external_errors[indices_keep], f"{colour_legend[name]}--")
                    legend += [f"{legend_legend[name]} (Lab)"]
                    legend_lines += [lns.Line2D([0], [0], 1, "--", colour_legend[name][0], colour_legend[name][1])]
    if is_external:
        # legend += ["Ideal 8 threads"]
        # legend_lines += [
        #     # lns.Line2D([0], [0], 1, "--", "grey"),
        #     lns.Line2D([0], [0], 1, ":", "grey")
        # ]
        plt.legend(legend_lines, legend, loc = "upper left", ncol = 1)
    else:
        # legend += ["Rotating frame", "Lab frame"]
        # legend_lines += [
        #     lns.Line2D([0], [0], 1, "-", "grey", "o"),
        #     lns.Line2D([0], [0], 1, "--", "grey", ".")
        # ]
        legend_lines_transpose = []
        legend_transpose = []
        for double_index in range(2):
            for legend_index in range(0, len(legend_lines), 2):
                legend_transpose += [legend[legend_index + double_index]]
                legend_lines_transpose += [legend_lines[legend_index + double_index]]
        plt.legend(legend_lines_transpose, legend_transpose, loc = "upper right", ncol = 2)
        # plt.legend(legend_lines, legend, loc = "upper right", ncol = 1)
    draw_spin(spin_dimension)
    plt.xlabel("Execution time (s)")
    plt.ylabel("Error")
    plt.subplots_adjust(right=0.99)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)
    plt.grid()
    if archive:
        archive.write_plot(f"{title}Error vs execution time", "benchmark_external_execution_error")
    plt.draw()

def new_benchmark_internal_spinsim(signal, frequency, time_step_fines, state_properties):
    for integration_method in [spinsim.IntegrationMethod.MAGNUS_CF4, spinsim.IntegrationMethod.HALF_STEP, spinsim.IntegrationMethod.MIDPOINT_SAMPLE]:
        for use_rotating_frame in [True, False]:
            profile_state, archive_path = handle_arguments()
            archive = Archive(archive_path, "spinsim internal", profile_state)
            archive.new_archive_file()
            new_benchmark_external_spinsim(archive, signal, frequency, time_step_fines, state_properties, integration_method = integration_method, use_rotating_frame = use_rotating_frame)
            archive.close_archive_file()

def new_benchmark_true_external_internal_spinsim(archive, frequency, time_step_fines):
    for spin_quantum_number in [spinsim.SpinQuantumNumber.ONE, spinsim.SpinQuantumNumber.HALF]:
        for integration_method in [spinsim.IntegrationMethod.MAGNUS_CF4, spinsim.IntegrationMethod.EULER, spinsim.IntegrationMethod.HEUN]:
            for use_rotating_frame in [True, False]:
                profile_state, archive_path = handle_arguments()
                archive = Archive(archive_path, "spinsim internal", profile_state)
                archive.new_archive_file()
                new_benchmark_true_external_spinsim(archive, frequency, time_step_fines, device = spinsim.Device.CUDA, spin_quantum_number = spin_quantum_number, integration_method = integration_method, use_rotating_frame = use_rotating_frame)
                archive.close_archive_file()

def new_benchmark_internal_trotter_spinsim(signal, frequency, time_step_fines, state_properties):
    for number_of_squares in range(4, 44, 4):
        profile_state, archive_path = handle_arguments()
        archive = Archive(archive_path, "spinsim internal trotter", profile_state)
        archive.new_archive_file()
        new_benchmark_external_spinsim(archive, signal, frequency, time_step_fines, state_properties, number_of_squares = number_of_squares)
        archive.close_archive_file()