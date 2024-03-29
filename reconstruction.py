from statistics import stdev
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
from numba import cuda
import numba as nb
import time as tm
import scipy.optimize
import scipy.signal
import scipy.linalg
import scipy.special
from cmcrameri import cm

import util
from util import PrettyTritty as C
from archive import *
from test_signal import *

class Reconstruction():
  """
  Controls a reconstruction of the signal from Fourier sine coefficients using compressive sensing

  Reconstruction modes:
  ---------------------
  Up to date:
  - fista_backtracking
  - fista_fit
  - fista_frequency_fit
  - least_squares
  - fista_adaptive
  - informed_least_squares

  Haven't checked in a while:
  - fista_ayanzadeh
  - ista_backtracking
  - fista
  """
  def __init__(self, time_properties:TimeProperties, step_size_sparse = 0.1, step_size_manifold = 100000):
    # self.end_buffer = 5
    self.end_buffer = 0
    self.time_properties = time_properties
    self.time_properties.time_coarse = self.time_properties.time_coarse[0:self.time_properties.time_coarse.size - self.end_buffer]
    self.frequency = None
    self.frequency_amplitude = None
    self.amplitude = None

    self.frequency_full = None
    self.frequency_amplitude_full = None
    self.frequency_amplitude_difference = None

    self.step_size_sparse = step_size_sparse
    self.step_size_manifold = step_size_manifold

  def read_frequencies_directly(self, frequency, frequency_amplitude, number_of_samples = 100, frequency_cutoff_low = 0, frequency_cutoff_high = 7500, random_seed = None):
    """
    Import arbitrary frequency values for reconstruction
    """
    number_of_samples_max = np.sum(np.logical_and(frequency_cutoff_low < frequency, frequency < frequency_cutoff_high))
    if number_of_samples > number_of_samples_max:
      number_of_samples = number_of_samples_max
      C.print(f"{C.r}number_of_samples is greater than the total number of samples input. Resetting{C.d}")
    if random_seed is not None:
      np.random.seed(random_seed)
    permutation = np.random.choice(range(number_of_samples_max), number_of_samples_max, replace = False)
    permutation = permutation[0:number_of_samples]
    self.frequency = np.ascontiguousarray(frequency[permutation])
    self.frequency_amplitude = np.ascontiguousarray(frequency_amplitude[permutation])

  def read_frequencies_from_experiment_results(self, experiment_results:ExperimentResults, number_of_samples = 100, frequency_cutoff_low = 0, frequency_cutoff_high = 100000, random_seed = None):
    """
    Import frequency values from an experimental results object
    """
    self.read_frequencies_directly(experiment_results.frequency, experiment_results.frequency_amplitude, number_of_samples, frequency_cutoff_low, frequency_cutoff_high, random_seed = random_seed)

  def read_frequencies_from_test_signal(self, test_signal:TestSignal, number_of_samples = 100, random_seed = None):
    """
    Import frequency values from the dot product Fourier transform of a signal
    """
    self.read_frequencies_directly(test_signal.frequency, test_signal.frequency_amplitude, number_of_samples, random_seed = random_seed)
  
  def evaluate_frequency_amplitude(self, test_signal:TestSignal = None):
    self.frequency_full = np.empty_like(self.time_properties.time_coarse, dtype = np.double)
    self.frequency_amplitude_full = np.empty_like(self.time_properties.time_coarse, dtype = np.double)
    self.frequency_amplitude_difference = np.empty_like(self.time_properties.time_coarse, dtype = np.double)

    # GPU control variables
    threads_per_block = 128
    blocks_per_grid = (self.time_properties.time_index_max + (threads_per_block - 1)) // threads_per_block
    # Run GPU code
    get_frequency_amplitude[blocks_per_grid, threads_per_block](self.time_properties.time_end_points, self.time_properties.time_coarse, self.time_properties.time_step_coarse, self.amplitude, self.frequency_full, self.frequency_amplitude_full)
    if test_signal:
      amplitude_difference = self.amplitude.copy()
      neural_pulse = test_signal.neural_pulses[0]
      # amplitude_difference[self.time_properties.time_coarse > neural_pulse.time_start - 0.5/neural_pulse.frequency and self.time_properties.time_coarse < neural_pulse.time_start + 1.5/neural_pulse.frequency] = 0
      amplitude_difference[0:int(amplitude_difference.size/2)] = 0
      get_frequency_amplitude[blocks_per_grid, threads_per_block](self.time_properties.time_end_points, self.time_properties.time_coarse, self.time_properties.time_step_coarse, amplitude_difference, self.frequency_full, self.frequency_amplitude_difference)

  def write_to_file(self, archive:h5py.Group, reconstruction_index = None):
    """
    Save the reconstruction results to a hdf5 file
    """
    if reconstruction_index is None:
      archive_group_reconstruction = archive.require_group("reconstruction")
    else:
      archive_group_reconstruction = archive.require_group(f"reconstructions/{reconstruction_index}")
    self.time_properties.write_to_file(archive_group_reconstruction)
    archive_group_reconstruction["frequency"] = self.frequency
    archive_group_reconstruction["frequency_amplitude"] = self.frequency_amplitude
    archive_group_reconstruction["amplitude"] = self.amplitude

    if hasattr(self, "reconstruction_step"):
      archive_group_reconstruction["reconstruction_step"] = self.reconstruction_step
    if hasattr(self, "iteration_max"):
      archive_group_reconstruction["iteration_max"] = self.iteration_max
    if hasattr(self, "norm_scale_factor"):
      archive_group_reconstruction["norm_scale_factor"] = self.norm_scale_factor
    if hasattr(self, "reconstruction_type"):
      archive_group_reconstruction["reconstruction_type"] = self.reconstruction_type

    if hasattr(self, "expected_amplitude"):
      archive_group_reconstruction["expected_amplitude"] = self.expected_amplitude
    if hasattr(self, "expected_frequency"):
      archive_group_reconstruction["expected_frequency"] = self.expected_frequency
    if hasattr(self, "expected_error_measurement"):
      archive_group_reconstruction["expected_error_measurement"] = self.expected_error_measurement
    if hasattr(self, "backtrack_scale"):
      archive_group_reconstruction["backtrack_scale"] = self.backtrack_scale
    if hasattr(self, "shrink_size_max"):
      archive_group_reconstruction["shrink_size_max"] = self.shrink_size_max

  def plot(self, archive:Archive, test_signal:TestSignal = None):
    """
    Plot the reconstruction signal, possibly against a template test signal
    """
    plt.figure()
    if test_signal is not None:
      plt.plot(test_signal.time_properties.time_coarse, test_signal.amplitude[0:test_signal.amplitude.size - self.end_buffer], "-k")
    plt.plot(self.time_properties.time_coarse, self.amplitude, "-r")
    if test_signal is not None:
      plt.legend(["Original", "Reconstruction"])
      plt.xlim(test_signal.time_properties.time_end_points)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (Hz)")
    plt.grid()
    if archive is not None:
      archive.write_plot("Reconstruction", "reconstruction")
    plt.draw()

    if test_signal:
      if test_signal.neural_pulses:
        plt.figure()
        plt.plot(test_signal.time_properties.time_coarse, test_signal.amplitude[0:test_signal.amplitude.size - self.end_buffer], "-k")
        plt.plot(self.time_properties.time_coarse, self.amplitude, "-xr")
        plt.legend(["Original", "Reconstruction"])
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude (Hz)")
        plt.grid()
        neural_pulse = test_signal.neural_pulses[0]
        plt.xlim([neural_pulse.time_start - 0.5/neural_pulse.frequency, neural_pulse.time_start + 1.5/neural_pulse.frequency])
        if archive:
          archive.write_plot("Reconstruction", "reconstruction_zoom")
        plt.draw()

    if self.frequency_amplitude_full is not None:
      plt.figure()
      if test_signal:
        plt.plot(test_signal.frequency, test_signal.frequency_amplitude, "-k")
      plt.plot(self.frequency, self.frequency_amplitude, ".g")
      plt.plot(self.frequency_full, self.frequency_amplitude_full, "--r")

      if test_signal:
        plt.legend(["Fourier Transform", "Measured", "Reconstructed"])
      plt.xlabel("Frequency (Hz)")
      plt.ylabel("Amplitude (Hz)")
      plt.xlim([0, np.max(self.frequency_full)])
      # plt.ylim([-0.08, 0.08])
      plt.grid()
      if archive:
        archive.write_plot("Reconstructed frequency amplitude", "reconstructed_frequency_amplitude")
      plt.draw()
      
    if self.frequency_amplitude_difference is not None:
      plt.figure()
      plt.plot(self.frequency_full, self.frequency_amplitude_difference, "--r")

      plt.xlabel("Frequency (Hz)")
      plt.ylabel("Amplitude (Hz)")
      plt.xlim([0, np.max(self.frequency_full)])
      # plt.ylim([-0.08, 0.08])
      plt.grid()
      if archive:
        archive.write_plot("Reconstructed frequency amplitude (difference)", "reconstructed_frequency_amplitude_difference")
      plt.draw()

  def evaluate_ista(self, expected_amplitude = 995.5, expected_frequency = 5025, expected_error_measurement = 11.87):
    C.starting("reconstruction (ISTA)")
    execution_time_endpoints = np.zeros(2, np.float64)
    execution_time_endpoints[0] = tm.time()

    threads_per_block = 128
    blocks_per_grid_time = (self.time_properties.time_coarse.size + (threads_per_block - 1)) // threads_per_block
    blocks_per_grid_frequency = (self.frequency.size + (threads_per_block - 1)) // threads_per_block

    self.expected_amplitude = expected_amplitude
    self.expected_frequency = expected_frequency
    self.expected_error_measurement = expected_error_measurement

    # # iteration_max = 50
    # iteration_max = 100
    # scale = self.time_properties.time_step_coarse/(self.time_properties.time_end_points[1] - self.time_properties.time_end_points[0])
    # self.fourier_scale = self.time_properties.time_step_coarse/(2*(self.time_properties.time_end_points[1] - self.time_properties.time_end_points[0]))
    self.fourier_scale = self.time_properties.time_step_coarse/(self.time_properties.time_step_coarse*(self.time_properties.time_coarse.size + 1))
    # ((self.time_properties.time_end_points[1] - self.time_properties.time_end_points[0]))
    self.reconstruction_step = 1e-4/self.fourier_scale
    expected_error_density = expected_amplitude/(math.pi*expected_frequency*self.time_properties.time_step_coarse)
    # self.norm_scale_factor = 0.25*((11.87*self.frequency_amplitude.size)**2)/3169
    self.norm_scale_factor = ((expected_error_measurement*self.frequency_amplitude.size)**2)/expected_error_density
    # self.norm_scale_factor = norm_scale_factor_modifier*self.frequency_amplitude.size*(expected_error_measurement**2)/expected_error_density
    # self.iteration_max = int(417)
    # self.iteration_max = int(math.ceil((expected_amplitude**2)/((4*(self.time_properties.time_end_points[1] - self.time_properties.time_end_points[0])*expected_frequency)*(2*expected_error_measurement))))
    self.iteration_max = int(math.ceil((expected_amplitude**2)/((4*(self.time_properties.time_step_coarse*(self.time_properties.time_coarse.size + 1))*expected_frequency)*(2*expected_error_measurement))))

    C.print(f"reconstruction_step: {self.reconstruction_step}\n\titeration_max: {self.iteration_max}\n\tnorm_scale_factor: {self.norm_scale_factor}")
    
    fourier_transform = cuda.device_array((self.frequency.size, self.time_properties.time_coarse.size), np.float64)
    evaluate_fourier_transform[blocks_per_grid_time, threads_per_block](cuda.to_device(self.frequency), cuda.to_device(self.time_properties.time_coarse), fourier_transform, self.fourier_scale)

    self.amplitude = np.zeros(self.time_properties.time_coarse.size, np.float64)
    amplitude = cuda.to_device(self.amplitude)

    frequency_amplitude = cuda.to_device(self.frequency_amplitude)

    frequency_amplitude_prediction = np.zeros(self.frequency_amplitude.size, np.float64)
    frequency_amplitude_prediction = cuda.to_device(frequency_amplitude_prediction)

    for iteration_index in range(self.iteration_max):
      evaluate_frequency_amplitude_prediction[blocks_per_grid_frequency, threads_per_block](amplitude, fourier_transform, frequency_amplitude_prediction)
      evaluate_next_iteration_ista[blocks_per_grid_time, threads_per_block](amplitude, frequency_amplitude, frequency_amplitude_prediction, fourier_transform, self.fourier_scale, self.norm_scale_factor, self.reconstruction_step)

    self.amplitude = amplitude.copy_to_host()

    execution_time_endpoints[1] = tm.time()
    C.print(f"reconstruction_time: {execution_time_endpoints[1] - execution_time_endpoints[0]:4.2f}")

    self.reconstruction_type = "ISTA"
    C.finished("reconstruction (ISTA)")

  def evaluate_ista_backtracking(self, expected_amplitude = 995.5, expected_frequency = 5025, expected_error_measurement = 11.87, backtrack_scale = 0.9):
    C.starting("reconstruction (ISTA with backtracking)")
    execution_time_endpoints = np.zeros(2, np.float64)
    execution_time_endpoints[0] = tm.time()

    threads_per_block = 128
    blocks_per_grid_time = (self.time_properties.time_coarse.size + (threads_per_block - 1)) // threads_per_block
    blocks_per_grid_frequency = (self.frequency.size + (threads_per_block - 1)) // threads_per_block

    self.expected_amplitude = expected_amplitude
    self.expected_frequency = expected_frequency
    self.expected_error_measurement = expected_error_measurement
    self.backtrack_scale = backtrack_scale

    # self.fourier_scale = self.time_properties.time_step_coarse/(2*(self.time_properties.time_end_points[1] - self.time_properties.time_end_points[0]))
    self.fourier_scale = self.time_properties.time_step_coarse/(self.time_properties.time_step_coarse*(self.time_properties.time_coarse.size + 1))
    # ((self.time_properties.time_end_points[1] - self.time_properties.time_end_points[0]))
    self.reconstruction_step = 1e-1/self.fourier_scale
    expected_error_density = expected_amplitude/(math.pi*expected_frequency*self.time_properties.time_step_coarse)
    self.norm_scale_factor = 0.2*((expected_error_measurement*self.frequency_amplitude.size)**2)/expected_error_density
    
    # self.iteration_max = int(math.ceil(backtrack_scale*(expected_amplitude**2)/((4*(self.time_properties.time_end_points[1] - self.time_properties.time_end_points[0])*expected_frequency)*(2*expected_error_measurement))))
    self.iteration_max = int(math.ceil(backtrack_scale*(expected_amplitude**2)/((4*(self.time_properties.time_step_coarse*(self.time_properties.time_coarse.size + 1))*expected_frequency)*(2*expected_error_measurement))))

    C.print(f"reconstruction_step: {self.reconstruction_step}\n\titeration_max: {self.iteration_max}\n\tnorm_scale_factor: {self.norm_scale_factor}")
    
    fourier_transform = cuda.device_array((self.frequency.size, self.time_properties.time_coarse.size), np.float64)
    evaluate_fourier_transform[blocks_per_grid_time, threads_per_block](cuda.to_device(self.frequency), cuda.to_device(self.time_properties.time_coarse), fourier_transform, self.fourier_scale)

    self.amplitude = 0*np.linalg.lstsq(fourier_transform, self.frequency_amplitude, rcond = None)[0]
    amplitude = cuda.to_device(self.amplitude)
    amplitude_previous = cuda.to_device(1*self.amplitude)

    frequency_amplitude = cuda.to_device(self.frequency_amplitude)

    frequency_amplitude_prediction = np.zeros(self.frequency_amplitude.size, np.float64)
    frequency_amplitude_prediction = cuda.to_device(frequency_amplitude_prediction)

    norm = 0
    norm_previous = 0
    reconstruction_step_backtrack = self.reconstruction_step

    for iteration_index in range(self.iteration_max):
      if reconstruction_step_backtrack < 1e-10:
        break
      do_backtrack = True
      while do_backtrack:
        copy_amplitude[blocks_per_grid_time, threads_per_block](amplitude, amplitude_previous)
        # average = np.mean(amplitude.copy_to_host())
        # subtract_constant[blocks_per_grid_time, threads_per_block](amplitude, 0.1*average)
        evaluate_frequency_amplitude_prediction[blocks_per_grid_frequency, threads_per_block](amplitude, fourier_transform, frequency_amplitude_prediction)
        evaluate_next_iteration_ista[blocks_per_grid_time, threads_per_block](amplitude, frequency_amplitude, frequency_amplitude_prediction, fourier_transform, self.fourier_scale, self.norm_scale_factor, reconstruction_step_backtrack)
        # evaluate_frequency_amplitude_prediction[blocks_per_grid_frequency, threads_per_block](amplitude, fourier_transform, frequency_amplitude_prediction)
        norm_previous = norm
        norm = self.norm_scale_factor*np.sum(np.abs(amplitude.copy_to_host())) + np.sqrt(np.sum((frequency_amplitude_prediction.copy_to_host() - frequency_amplitude.copy_to_host())**2))
        if iteration_index > 0:
          if norm > norm_previous:
            reconstruction_step_backtrack *= backtrack_scale
            copy_amplitude[blocks_per_grid_time, threads_per_block](amplitude_previous, amplitude)
            norm = norm_previous
            if reconstruction_step_backtrack < 1e-10:
              break
          else:
            do_backtrack = False
        else:
          do_backtrack = False
        C.print(f"Index: {iteration_index}, Reconstruction step: {reconstruction_step_backtrack}, norm: {norm}", end = "\r")
    C.print("")
    self.amplitude = amplitude.copy_to_host()

    execution_time_endpoints[1] = tm.time()
    C.print(f"reconstruction_time: {execution_time_endpoints[1] - execution_time_endpoints[0]:4.2f}")

    self.reconstruction_type = "ISTA with backtracking"
    C.finished("reconstruction (ISTA with backtracking)")

  def evaluate_fista_backtracking(self, expected_amplitude = 995.5, expected_frequency = 5025, expected_error_measurement = 11.87, backtrack_scale = 0.9, norm_scale_factor_modifier = 2.0, is_fast = False, norm_scale_factor = None):
    C.starting("reconstruction (FISTA with backtracking)")
    execution_time_endpoints = np.zeros(2, np.float64)
    execution_time_endpoints[0] = tm.time()

    threads_per_block = 128
    blocks_per_grid_time = (self.time_properties.time_coarse.size + (threads_per_block - 1)) // threads_per_block
    blocks_per_grid_frequency = (self.frequency.size + (threads_per_block - 1)) // threads_per_block

    self.expected_amplitude = expected_amplitude
    self.expected_frequency = expected_frequency
    self.expected_error_measurement = expected_error_measurement
    self.backtrack_scale = backtrack_scale

    # # tolerable_error = 2*expected_error_measurement
    # tolerable_error = expected_error_measurement*np.sqrt(self.frequency_amplitude.size)
    # gradient_lipschitz = self.time_properties.time_step_coarse/(self.time_properties.time_step_coarse*(self.time_properties.time_coarse.size + 1))
    # # gradient_lipschitz = self.time_properties.time_step_coarse/(self.time_properties.time_end_points[1] - self.time_properties.time_end_points[0])
    # self.reconstruction_step = 1/gradient_lipschitz
    # self.fourier_scale = gradient_lipschitz
    # expected_sparsity = 1/(expected_frequency*self.time_properties.time_step_coarse)
    # expected_signal_energy = 0.5*(expected_amplitude**2)*expected_sparsity

    self.fourier_scale = self.time_properties.time_step_coarse/(self.time_properties.time_step_coarse*(self.time_properties.time_coarse.size + 1))
    fourier_transform = cuda.device_array((self.frequency.size, self.time_properties.time_coarse.size), np.float64)
    evaluate_fourier_transform[blocks_per_grid_time, threads_per_block](cuda.to_device(self.frequency), cuda.to_device(self.time_properties.time_coarse), fourier_transform, self.fourier_scale)
    gradient_lipschitz = 2*np.max(np.abs(scipy.linalg.svdvals(fourier_transform.copy_to_host()))**2)

    tolerable_error = expected_error_measurement*np.sqrt(self.frequency_amplitude.size)
    self.reconstruction_step = 1/gradient_lipschitz
    expected_sparsity = 1/(expected_frequency*self.time_properties.time_step_coarse)
    expected_signal_energy = (0.5*(expected_amplitude**2)*expected_sparsity)

    iteration_max_rate = 0.5*gradient_lipschitz*expected_signal_energy*expected_frequency
    if is_fast:
      iteration_max_rate = 2*np.sqrt(iteration_max_rate)
    self.iteration_max = int(np.ceil(iteration_max_rate/tolerable_error))

    # # self.fourier_scale = self.time_properties.time_step_coarse/(2*(self.time_properties.time_end_points[1] - self.time_properties.time_end_points[0]))
    # # self.fourier_scale = self.time_properties.time_step_coarse/(self.time_properties.time_end_points[1] - self.time_properties.time_end_points[0])
    # # self.reconstruction_step = 2e-2/self.fourier_scale
    # # self.reconstruction_step = 1e-1/self.fourier_scale
    # # self.reconstruction_step = 5e-2/self.fourier_scale
    # expected_error_density = expected_amplitude/(math.pi*expected_frequency*self.time_properties.time_step_coarse)
    # # self.norm_scale_factor = norm_scale_factor_modifier*((expected_error_measurement*self.frequency_amplitude.size)**2)/expected_error_density
    # self.norm_scale_factor = norm_scale_factor_modifier*self.frequency_amplitude.size*(expected_error_measurement**2)/expected_error_density
    # # self.iteration_max = int(math.ceil(2*np.sqrt(backtrack_scale*(expected_amplitude**2)/((4*(self.time_properties.time_end_points[1] - self.time_properties.time_end_points[0])*expected_frequency)*(2*expected_error_measurement)))))
    # # self.iteration_max = int(math.ceil(2*np.sqrt((expected_amplitude**2)/((4*(self.time_properties.time_end_points[1] - self.time_properties.time_end_points[0])*expected_frequency)*(2*expected_error_measurement)))))
    
    if norm_scale_factor is not None:
      self.norm_scale_factor = norm_scale_factor
    else:
      # self.norm_scale_factor = norm_scale_factor_modifier*4*expected_error_measurement*math.sqrt(gradient_lipschitz)*scipy.special.erfcinv(2/(self.frequency.size + 1)) # Chichignoud et al 2016, probability max unmodified
      # self.norm_scale_factor = norm_scale_factor_modifier*4*expected_error_measurement*math.sqrt(gradient_lipschitz)*scipy.special.erfcinv(5/(4*self.frequency.size + 1)) # Chichignoud et al 2016, probability max
      self.norm_scale_factor = norm_scale_factor_modifier*4*tolerable_error*gradient_lipschitz # Chichignoud et al 2016, probability mean
      # self.norm_scale_factor = 4*norm_scale_factor_modifier*expected_signal_energy*np.max(np.sqrt(np.sum(fourier_transform.copy_to_host()**2, axis = 0))) # Chichignoud et al 2016, operator norm
      # self.norm_scale_factor = 4*np.min(np.abs(np.sqrt(np.sum((fourier_transform.copy_to_host()*expected_error_measurement)**2, axis = 0)))) # Chichignoud et al 2016, more advanced
      # self.norm_scale_factor = norm_scale_factor_modifier*4*self.frequency.size*expected_error_measurement*gradient_lipschitz # Chichignoud et al 2016
      # self.norm_scale_factor = norm_scale_factor_modifier*math.sqrt(8*expected_error_measurement*math.log(self.time_properties.time_coarse.size - expected_sparsity)) # Eldar and Kutyniok 2012


    self.shrink_size_max = 1e2

    C.print(f"reconstruction_step: {self.reconstruction_step}\niteration_max: {self.iteration_max}\nnorm_scale_factor: {self.norm_scale_factor}")

    self.amplitude = np.linalg.lstsq(fourier_transform.copy_to_host(), self.frequency_amplitude, rcond = None)[0]
    # # self.amplitude = 150*(1 - 2*np.fmod(self.time_properties.time_coarse/self.time_properties.time_coarse[1], 2))
    # self.amplitude = 0*self.time_properties.time_coarse
    # self.amplitude[0] = 1
    amplitude = cuda.to_device(self.amplitude)
    amplitude_previous = cuda.to_device(1*self.amplitude)

    frequency_amplitude = cuda.to_device(self.frequency_amplitude)

    frequency_amplitude_prediction = np.zeros(self.frequency_amplitude.size, np.float64)
    frequency_amplitude_prediction = cuda.to_device(frequency_amplitude_prediction)

    norm = 0
    norm_previous = np.infty
    reconstruction_step_backtrack = self.reconstruction_step
    moving_away_strike_max = 0
    moving_away_strike = moving_away_strike_max

    fast_step_size = 1
    fast_step_size_previous = 1
    fast_step_size_previous_previous = 1

    for iteration_index in range(self.iteration_max):
      if reconstruction_step_backtrack < 1e-6:
        break
      do_backtrack = True
      while do_backtrack:
        copy_amplitude[blocks_per_grid_time, threads_per_block](amplitude, amplitude_previous)
        evaluate_frequency_amplitude_prediction[blocks_per_grid_frequency, threads_per_block](amplitude, fourier_transform, frequency_amplitude_prediction)
        shrink_scale = np.max(np.abs(amplitude.copy_to_host()))
        shrink_scale_denominator = shrink_scale - self.norm_scale_factor*self.reconstruction_step
        if shrink_scale_denominator < 0:
          shrink_scale_denominator = 0
          shrink_scale = self.shrink_size_max
        else:
          shrink_scale = shrink_scale/shrink_scale_denominator
          if shrink_scale > self.shrink_size_max:
            shrink_scale = self.shrink_size_max
        shrink_scale = 1
        evaluate_next_iteration_ista_shrink_scale[blocks_per_grid_time, threads_per_block](amplitude, frequency_amplitude, frequency_amplitude_prediction, fourier_transform, self.norm_scale_factor, reconstruction_step_backtrack, shrink_scale)
        fast_step_size_previous_previous = fast_step_size_previous
        fast_step_size_previous = fast_step_size
        fast_step_size = (1 + math.sqrt(1 + 4*fast_step_size**2))/2
        if is_fast:
          evaluate_fista_fast_step[blocks_per_grid_time, threads_per_block](amplitude, amplitude_previous, fast_step_size, fast_step_size_previous)
        norm_previous = norm
        norm = self.norm_scale_factor*np.sum(np.abs(amplitude.copy_to_host())) + np.sqrt(np.sum((frequency_amplitude_prediction.copy_to_host() - frequency_amplitude.copy_to_host())**2))
        if iteration_index > 0:#-1:
          if norm > norm_previous:
            moving_away_strike -= 1
            norm = norm_previous
          if moving_away_strike == 0 or norm == 0:
            moving_away_strike = moving_away_strike_max
            reconstruction_step_backtrack *= backtrack_scale
            copy_amplitude[blocks_per_grid_time, threads_per_block](amplitude_previous, amplitude)
            norm = norm_previous
            fast_step_size = fast_step_size_previous
            fast_step_size_previous = fast_step_size_previous_previous
            if reconstruction_step_backtrack < 1e-6:
              break
          else:
            do_backtrack = False
            # reconstruction_step_backtrack = self.reconstruction_step
        else:
          do_backtrack = False
        C.print(f"Index: {iteration_index}, Reconstruction step: {reconstruction_step_backtrack}, norm: {norm}", end = "\r")
    C.print("")
    self.amplitude = amplitude.copy_to_host()

    execution_time_endpoints[1] = tm.time()
    C.print(f"reconstruction_time: {execution_time_endpoints[1] - execution_time_endpoints[0]:4.2f}")

    self.reconstruction_type = "FISTA with backtracking"
    C.finished("reconstruction (FISTA with backtracking)")

  def evaluate_fista_adaptive(self, expected_amplitude = 995.5, expected_frequency = 5025, expected_error_measurement = 11.87, backtrack_scale = 0.9, norm_scale_factor_modifier = 2.0, is_fast = False):
    C.starting("reconstruction (FISTA adaptive)")
    execution_time_endpoints = np.zeros(2, np.float64)
    execution_time_endpoints[0] = tm.time()

    threads_per_block = 128
    blocks_per_grid_time = (self.time_properties.time_coarse.size + (threads_per_block - 1)) // threads_per_block
    blocks_per_grid_frequency = (self.frequency.size + (threads_per_block - 1)) // threads_per_block

    self.expected_amplitude = expected_amplitude
    self.expected_frequency = expected_frequency
    self.expected_error_measurement = expected_error_measurement
    self.backtrack_scale = backtrack_scale

    self.fourier_scale = self.time_properties.time_step_coarse/(self.time_properties.time_step_coarse*(self.time_properties.time_coarse.size + 1))
    fourier_transform = cuda.device_array((self.frequency.size, self.time_properties.time_coarse.size), np.float64)
    evaluate_fourier_transform[blocks_per_grid_time, threads_per_block](cuda.to_device(self.frequency), cuda.to_device(self.time_properties.time_coarse), fourier_transform, self.fourier_scale)
    gradient_lipschitz = 2*np.max(np.abs(scipy.linalg.svdvals(fourier_transform.copy_to_host()))**2)

    tolerable_error = expected_error_measurement*np.sqrt(self.frequency_amplitude.size)
    self.reconstruction_step = 1/gradient_lipschitz
    expected_sparsity = 1/(expected_frequency*self.time_properties.time_step_coarse)
    expected_signal_energy = (0.5*(expected_amplitude**2)*expected_sparsity)

    iteration_max_rate = 0.5*gradient_lipschitz*expected_signal_energy*expected_frequency
    if is_fast:
      iteration_max_rate = 2*np.sqrt(iteration_max_rate)
    self.iteration_max = int(np.ceil(iteration_max_rate/tolerable_error))
    
    self.norm_scale_factor = norm_scale_factor_modifier*4*expected_error_measurement*math.sqrt(gradient_lipschitz)*scipy.special.erfcinv(5/(4*self.frequency.size + 1)) # Chichignoud et al 2016, probability max


    self.shrink_size_max = 1e2

    C.print(f"reconstruction_step: {self.reconstruction_step}\niteration_max: {self.iteration_max}\nnorm_scale_factor: {self.norm_scale_factor}")

    self.amplitude = 0*self.time_properties.time_coarse
    amplitude = cuda.to_device(self.amplitude)
    amplitude_previous = cuda.to_device(1*self.amplitude)

    frequency_amplitude = cuda.to_device(self.frequency_amplitude)

    frequency_amplitude_prediction = np.zeros(self.frequency_amplitude.size, np.float64)
    frequency_amplitude_prediction = cuda.to_device(frequency_amplitude_prediction)

    norm = 0
    norm_previous = np.infty
    reconstruction_step_backtrack = self.reconstruction_step
    moving_away_strike_max = 0
    moving_away_strike = moving_away_strike_max

    fast_step_size = 1
    fast_step_size_previous = 1
    fast_step_size_previous_previous = 1

    refinement_level = 1
    weight_maximum = 2 #10000
    for refinement_index in range(refinement_level):
      weights = np.abs(self.amplitude)
      weights = (weights/np.max(weights) + 1/weight_maximum)
      # weights = (weights + 1/weight_maximum)
      weights = weights**(-1)
      weights = cuda.to_device(weights)

      amplitude = cuda.to_device(self.amplitude)
      amplitude_previous = cuda.to_device(self.amplitude.copy())

      frequency_amplitude = cuda.to_device(self.frequency_amplitude)

      frequency_amplitude_prediction = np.zeros(self.frequency_amplitude.size, np.float64)
      frequency_amplitude_prediction = cuda.to_device(frequency_amplitude_prediction)

      norm = 0
      norm_previous = np.infty
      reconstruction_step_backtrack = self.reconstruction_step
      moving_away_strike_max = 0
      moving_away_strike = moving_away_strike_max

      fast_step_size = 1
      fast_step_size_previous = 1
      fast_step_size_previous_previous = 1

      for iteration_index in range(self.iteration_max):
        if reconstruction_step_backtrack < 1e-6:
          break
        do_backtrack = True
        while do_backtrack:
          copy_amplitude[blocks_per_grid_time, threads_per_block](amplitude, amplitude_previous)
          evaluate_frequency_amplitude_prediction[blocks_per_grid_frequency, threads_per_block](amplitude, fourier_transform, frequency_amplitude_prediction)
          evaluate_next_iteration_ista_adaptive[blocks_per_grid_time, threads_per_block](amplitude, frequency_amplitude, frequency_amplitude_prediction, fourier_transform, self.norm_scale_factor, reconstruction_step_backtrack, weights)
          if is_fast:
            fast_step_size_previous_previous = fast_step_size_previous
            fast_step_size_previous = fast_step_size
            fast_step_size = (1 + math.sqrt(1 + 4*fast_step_size**2))/2
            evaluate_fista_fast_step[blocks_per_grid_time, threads_per_block](amplitude, amplitude_previous, fast_step_size, fast_step_size_previous)
          norm_previous = norm
          norm = self.norm_scale_factor*np.sum(np.abs(amplitude.copy_to_host())) + np.sqrt(np.sum((frequency_amplitude_prediction.copy_to_host() - frequency_amplitude.copy_to_host())**2))
          if iteration_index > 0:
            if norm > norm_previous:
              moving_away_strike -= 1
              norm = norm_previous
            if moving_away_strike == 0 or norm == 0:
              moving_away_strike = moving_away_strike_max
              reconstruction_step_backtrack *= backtrack_scale
              copy_amplitude[blocks_per_grid_time, threads_per_block](amplitude_previous, amplitude)
              norm = norm_previous
              fast_step_size = fast_step_size_previous
              fast_step_size_previous = fast_step_size_previous_previous
              if reconstruction_step_backtrack < 1e-6:
                break
            else:
              do_backtrack = False
              # reconstruction_step_backtrack = self.reconstruction_step
          else:
            do_backtrack = False
          C.print(f"Index: {iteration_index}, Reconstruction step: {reconstruction_step_backtrack}, norm: {norm}", end = "\r")
      C.print("")
      self.amplitude = amplitude.copy_to_host()

    execution_time_endpoints[1] = tm.time()
    C.print(f"reconstruction_time: {execution_time_endpoints[1] - execution_time_endpoints[0]:4.2f}")

    self.reconstruction_type = "FISTA adaptive"
    C.finished("reconstruction (FISTA adaptive)")

  def evaluate_fista_frequency_fit(self, expected_amplitude = 995.5, expected_frequency = 5025, expected_error_measurement = 11.87, backtrack_scale = 0.9, norm_scale_factor_modifier = 2.0, frequency_fit_step_size = 1, is_fast = False):
    C.starting("reconstruction (FISTA frequency fit)")
    execution_time_endpoints = np.zeros(2, np.float64)
    execution_time_endpoints[0] = tm.time()

    threads_per_block = 128
    blocks_per_grid_time = (self.time_properties.time_coarse.size + (threads_per_block - 1)) // threads_per_block
    blocks_per_grid_frequency = (self.frequency.size + (threads_per_block - 1)) // threads_per_block

    self.expected_amplitude = expected_amplitude
    self.expected_frequency = expected_frequency
    self.expected_error_measurement = expected_error_measurement
    self.backtrack_scale = backtrack_scale

    tolerable_error = 2*expected_error_measurement
    gradient_lipschitz = self.time_properties.time_step_coarse/(self.time_properties.time_step_coarse*(self.time_properties.time_coarse.size + 1))
    self.reconstruction_step = 1/gradient_lipschitz
    self.fourier_scale = gradient_lipschitz
    expected_signal_energy = 0.5*(expected_amplitude**2)/(expected_frequency*self.time_properties.time_step_coarse)

    iteration_max_rate = 0.5*gradient_lipschitz*expected_signal_energy
    if is_fast:
      iteration_max_rate = 2*np.sqrt(iteration_max_rate)
    self.iteration_max = int(np.ceil(iteration_max_rate/tolerable_error))

    self.norm_scale_factor = norm_scale_factor_modifier*4*self.frequency.size*expected_error_measurement*gradient_lipschitz # Chichignoud et al 2016
    # self.norm_scale_factor = norm_scale_factor_modifier*math.sqrt(8*expected_error_measurement*math.log(self.time_properties.time_coarse.size - expected_sparsity)) # Eldar and Kutyniok 2012


    self.shrink_size_max = 1e2

    C.print(f"reconstruction_step: {self.reconstruction_step}\niteration_max: {self.iteration_max}\nnorm_scale_factor: {self.norm_scale_factor}")
    
    frequency = cuda.to_device(self.frequency)
    self.frequency = np.array(self.frequency, np.float64)
    fourier_transform = cuda.device_array((self.frequency.size, self.time_properties.time_coarse.size), np.float64)
    derivative_fourier_transform = cuda.device_array((self.frequency.size, self.time_properties.time_coarse.size), np.float64)
    evaluate_fourier_transform[blocks_per_grid_time, threads_per_block](frequency, cuda.to_device(self.time_properties.time_coarse), fourier_transform, self.fourier_scale)
    evaluate_derivative_fourier_transform[blocks_per_grid_time, threads_per_block](frequency, cuda.to_device(self.time_properties.time_coarse), derivative_fourier_transform, self.fourier_scale)

    self.amplitude = np.linalg.lstsq(fourier_transform.copy_to_host(), self.frequency_amplitude, rcond = None)[0]

    refinement_level = 1
    weight_maximum = 10000 #10000
    for refinement_index in range(refinement_level):
      weights = np.abs(self.amplitude)
      weights = (weights/np.max(weights) + 1/weight_maximum)
      weights = weights**(-1)
      weights = cuda.to_device(weights)

      amplitude = cuda.to_device(self.amplitude)
      amplitude_previous = cuda.to_device(self.amplitude.copy())

      frequency_amplitude = cuda.to_device(self.frequency_amplitude)

      frequency_amplitude_prediction = np.zeros(self.frequency_amplitude.size, np.float64)
      frequency_amplitude_prediction = cuda.to_device(frequency_amplitude_prediction)

      norm = 0
      norm_previous = np.infty
      reconstruction_step_backtrack = self.reconstruction_step
      moving_away_strike_max = 0
      moving_away_strike = moving_away_strike_max

      fast_step_size = 1
      fast_step_size_previous = 1
      fast_step_size_previous_previous = 1

      for iteration_index in range(self.iteration_max):
        if reconstruction_step_backtrack < 1e-6:
          break
        do_backtrack = True
        while do_backtrack:
          copy_amplitude[blocks_per_grid_time, threads_per_block](amplitude, amplitude_previous)
          if True:#iteration_index > self.iteration_max/2:
            frequency_fit_step_size_use = frequency_fit_step_size*reconstruction_step_backtrack
          else:
            frequency_fit_step_size_use = 0
          evaluate_frequency_amplitude_prediction_frequency_fit[blocks_per_grid_frequency, threads_per_block](amplitude, frequency, frequency_amplitude, fourier_transform, derivative_fourier_transform, frequency_amplitude_prediction, frequency_fit_step_size_use)
          evaluate_next_iteration_ista_adaptive[blocks_per_grid_time, threads_per_block](amplitude, frequency_amplitude, frequency_amplitude_prediction, fourier_transform, self.norm_scale_factor, reconstruction_step_backtrack, weights)
          if is_fast:
            fast_step_size_previous_previous = fast_step_size_previous
            fast_step_size_previous = fast_step_size
            fast_step_size = (1 + math.sqrt(1 + 4*fast_step_size**2))/2
            evaluate_fista_fast_step[blocks_per_grid_time, threads_per_block](amplitude, amplitude_previous, fast_step_size, fast_step_size_previous)
          norm_previous = norm
          norm = self.norm_scale_factor*np.sum(np.abs(amplitude.copy_to_host())) + np.sqrt(np.sum((frequency_amplitude_prediction.copy_to_host() - frequency_amplitude.copy_to_host())**2))
          if iteration_index > 0:
            if norm > norm_previous:
              moving_away_strike -= 1
              norm = norm_previous
            if moving_away_strike == 0 or norm == 0:
              moving_away_strike = moving_away_strike_max
              reconstruction_step_backtrack *= backtrack_scale
              copy_amplitude[blocks_per_grid_time, threads_per_block](amplitude_previous, amplitude)
              norm = norm_previous
              fast_step_size = fast_step_size_previous
              fast_step_size_previous = fast_step_size_previous_previous
              if reconstruction_step_backtrack < 1e-6:
                break
            else:
              do_backtrack = False
          else:
            do_backtrack = False
          C.print(f"Index: {iteration_index}, Reconstruction step: {reconstruction_step_backtrack}, norm: {norm}", end = "\r")
      C.print("")
      self.amplitude = amplitude.copy_to_host()
      self.frequency = frequency.copy_to_host()

    execution_time_endpoints[1] = tm.time()
    C.print(f"reconstruction_time: {execution_time_endpoints[1] - execution_time_endpoints[0]:4.2f}")

    self.reconstruction_type = "FISTA frequency fit"
    C.finished("reconstruction (FISTA frequency fit)")

  def evaluate_fista_fit(self, expected_amplitude = 995.5, expected_frequency = 5025, expected_error_measurement = 11.87, backtrack_scale = 0.9, norm_scale_factor_modifier = 2.0, rabi_frequency_readout = 2e4, frequency_line_noise = 50, is_fast = False):
    C.starting("reconstruction (FISTA with noise fitting)")
    execution_time_endpoints = np.zeros(2, np.float64)
    execution_time_endpoints[0] = tm.time()

    number_of_fit_parameters = 1

    threads_per_block = 128
    blocks_per_grid_time = (self.time_properties.time_coarse.size + number_of_fit_parameters + (threads_per_block - 1)) // threads_per_block
    blocks_per_grid_frequency = (self.frequency.size + (threads_per_block - 1)) // threads_per_block

    self.expected_amplitude = expected_amplitude
    self.expected_frequency = expected_frequency
    self.expected_error_measurement = expected_error_measurement
    self.backtrack_scale = backtrack_scale

    self.fourier_scale = self.time_properties.time_step_coarse/(self.time_properties.time_step_coarse*(self.time_properties.time_coarse.size + 1))
    # (self.time_properties.time_end_points[1] - self.time_properties.time_end_points[0])

    # self.reconstruction_step = 1e-4/self.fourier_scale
    self.reconstruction_step = 1e-1/self.fourier_scale
    expected_error_density = expected_amplitude/(math.pi*expected_frequency*self.time_properties.time_step_coarse)
    # self.norm_scale_factor = norm_scale_factor_modifier*((expected_error_measurement*self.frequency_amplitude.size)**2)/expected_error_density
    # self.norm_scale_factor = norm_scale_factor_modifier*self.frequency_amplitude.size*(expected_error_measurement**2)/expected_error_density

    self.norm_scale_factor = norm_scale_factor_modifier*4*self.frequency.size*expected_error_measurement*gradient_lipschitz # Chichignoud et al 2016
    # self.norm_scale_factor = norm_scale_factor_modifier*math.sqrt(8*expected_error_measurement*math.log(self.time_properties.time_coarse.size - expected_sparsity)) # Eldar and Kutyniok 2012
    
    # self.iteration_max = int(math.ceil(2*np.sqrt(backtrack_scale*(expected_amplitude**2)/((4*(self.time_properties.time_end_points[1] - self.time_properties.time_end_points[0])*expected_frequency)*(2*expected_error_measurement)))))
    self.iteration_max = int(math.ceil(2*np.sqrt(backtrack_scale*(expected_amplitude**2)/((4*(self.time_properties.time_step_coarse*(self.time_properties.time_coarse.size + 1))*expected_frequency)*(2*expected_error_measurement)))))
    self.shrink_size_max = 1e2

    C.print(f"reconstruction_step: {self.reconstruction_step}\niteration_max: {self.iteration_max}\nnorm_scale_factor: {self.norm_scale_factor}")
    
    fourier_transform = cuda.device_array((self.frequency.size, self.time_properties.time_coarse.size), np.float64)
    evaluate_fourier_transform[blocks_per_grid_time, threads_per_block](cuda.to_device(self.frequency), cuda.to_device(self.time_properties.time_coarse), fourier_transform, self.fourier_scale)

    # self.amplitude = 100 + 0*np.linalg.lstsq(fourier_transform.copy_to_host(), self.frequency_amplitude, rcond = None)[0]
    # self.amplitude = 150*(1 - 2*np.fmod(self.time_properties.time_coarse/self.time_properties.time_coarse[1], 2))
    self.amplitude = 0*self.time_properties.time_coarse
    self.amplitude[0] = 1
    amplitude = cuda.to_device(self.amplitude)
    amplitude_previous = cuda.to_device(1*self.amplitude)

    frequency_amplitude = cuda.to_device(self.frequency_amplitude)
    fit_parameters = cuda.to_device(np.array([800], np.float64))
    fit_parameters_previous = cuda.to_device(np.array([800], np.float64))
    line_noise_derivative_amplitude = cuda.to_device(0*self.frequency_amplitude)

    frequency_amplitude_prediction = np.zeros(self.frequency_amplitude.size, np.float64)
    frequency_amplitude_prediction = cuda.to_device(frequency_amplitude_prediction)

    norm = 0
    norm_previous = 0
    reconstruction_step_backtrack = self.reconstruction_step

    fast_step_size = 1
    fast_step_size_previous = 1
    fast_step_size_previous_previous = 1

    for iteration_index in range(self.iteration_max):
      if reconstruction_step_backtrack < 1e-10:
        break
      do_backtrack = True
      while do_backtrack:
        copy_amplitude_fit[blocks_per_grid_time, threads_per_block](amplitude, amplitude_previous, fit_parameters, fit_parameters_previous)
        evaluate_frequency_amplitude_prediction_fit[blocks_per_grid_frequency, threads_per_block](amplitude, fourier_transform, frequency_amplitude_prediction, cuda.to_device(self.frequency), fit_parameters, line_noise_derivative_amplitude, self.time_properties.time_end_points[1] - self.time_properties.time_end_points[0] + 0.0/rabi_frequency_readout, rabi_frequency_readout, frequency_line_noise)
        shrink_scale = np.max(np.abs(amplitude.copy_to_host()))
        shrink_scale_denominator = shrink_scale - self.norm_scale_factor*self.reconstruction_step
        if shrink_scale_denominator < 0:
          shrink_scale_denominator = 0
          shrink_scale = self.shrink_size_max
        else:
          shrink_scale = shrink_scale/shrink_scale_denominator
          if shrink_scale > self.shrink_size_max:
            shrink_scale = self.shrink_size_max
        # shrink_scale = 1
        evaluate_next_iteration_ista_fit[blocks_per_grid_time, threads_per_block](amplitude, frequency_amplitude, frequency_amplitude_prediction, fourier_transform, self.norm_scale_factor, reconstruction_step_backtrack, shrink_scale, fit_parameters, line_noise_derivative_amplitude)
        if is_fast:
          fast_step_size_previous_previous = fast_step_size_previous
          fast_step_size_previous = fast_step_size
          fast_step_size = (1 + math.sqrt(1 + 4*fast_step_size**2))/2
          evaluate_fista_fast_step_fit[blocks_per_grid_time, threads_per_block](amplitude, amplitude_previous, fast_step_size, fast_step_size_previous, fit_parameters, fit_parameters_previous)
        norm_previous = norm
        norm = self.norm_scale_factor*np.sum(np.abs(amplitude.copy_to_host())) + np.sqrt(np.sum((frequency_amplitude_prediction.copy_to_host() - frequency_amplitude.copy_to_host())**2))
        if iteration_index > 0:
          if norm > norm_previous or norm == 0:
            reconstruction_step_backtrack *= backtrack_scale
            copy_amplitude_fit[blocks_per_grid_time, threads_per_block](amplitude_previous, amplitude, fit_parameters_previous, fit_parameters)
            norm = norm_previous
            fast_step_size = fast_step_size_previous
            fast_step_size_previous = fast_step_size_previous_previous
            if reconstruction_step_backtrack < 1e-10:
              break
          else:
            do_backtrack = False
            # reconstruction_step_backtrack = self.reconstruction_step
        else:
          do_backtrack = False
        C.print(f"Index: {iteration_index}, Reconstruction step: {reconstruction_step_backtrack}, Norm: {norm}, Noise: {fit_parameters.copy_to_host()[0]}", end = "\r")
    C.print("")
    self.amplitude = amplitude.copy_to_host()

    execution_time_endpoints[1] = tm.time()
    C.print(f"reconstruction_time: {execution_time_endpoints[1] - execution_time_endpoints[0]:4.2f}")

    self.reconstruction_type = "FISTA with noise fitting"
    C.finished("reconstruction (FISTA with noise fitting)")

  def evaluate_least_squares(self):
    C.starting("reconstruction (least squares)")

    threads_per_block = 128
    blocks_per_grid_time = (self.time_properties.time_coarse.size + (threads_per_block - 1)) // threads_per_block
    blocks_per_grid_frequency = (self.frequency.size + (threads_per_block - 1)) // threads_per_block

    # self.fourier_scale = self.time_properties.time_step_coarse/(2*(self.time_properties.time_end_points[1] - self.time_properties.time_end_points[0]))
    self.fourier_scale = self.time_properties.time_step_coarse/(self.time_properties.time_step_coarse*(self.time_properties.time_coarse.size + 1))
    # (self.time_properties.time_end_points[1] - self.time_properties.time_end_points[0])

    fourier_transform = np.empty((self.frequency.size, self.time_properties.time_coarse.size), np.float64)
    evaluate_fourier_transform[blocks_per_grid_time, threads_per_block](cuda.to_device(self.frequency), cuda.to_device(self.time_properties.time_coarse), fourier_transform, self.fourier_scale)

    self.amplitude = np.linalg.lstsq(fourier_transform, self.frequency_amplitude, rcond = None)[0]
    # self.amplitude *= (self.time_properties.time_coarse.size/self.frequency.size)#*0.7

    self.reconstruction_type = "least squares"
    C.finished("reconstruction (least squares)")

  def evaluate_informed_least_squares(self, informed_type = "fista", expected_amplitude = 995.5, expected_frequency = 5025, expected_error_measurement = 11.87, backtrack_scale = 0.9, norm_scale_factor_modifier = 2.0, frequency_fit_step_size = 1):
    C.starting("Informed least squares")
    if informed_type == "fista":
      self.evaluate_fista_backtracking(expected_amplitude, expected_frequency, expected_error_measurement, backtrack_scale, norm_scale_factor_modifier, is_fast = True)
    elif informed_type == "ista":
      self.evaluate_fista_backtracking(expected_amplitude, expected_frequency, expected_error_measurement, backtrack_scale, norm_scale_factor_modifier, is_fast = False)
    elif informed_type == "fista_adaptive":
      self.evaluate_fista_adaptive(expected_amplitude, expected_frequency, expected_error_measurement, backtrack_scale, norm_scale_factor_modifier, is_fast = True)
    elif informed_type == "ista_adaptive":
      self.evaluate_fista_adaptive(expected_amplitude, expected_frequency, expected_error_measurement, backtrack_scale, norm_scale_factor_modifier, is_fast = False)
    elif informed_type == "fista_frequency_fit":
      self.evaluate_fista_frequency_fit(expected_amplitude, expected_frequency, expected_error_measurement, backtrack_scale, norm_scale_factor_modifier, frequency_fit_step_size = frequency_fit_step_size, is_fast = True)
    elif informed_type == "ista_frequency_fit":
      self.evaluate_fista_frequency_fit(expected_amplitude, expected_frequency, expected_error_measurement, backtrack_scale, norm_scale_factor_modifier, frequency_fit_step_size = frequency_fit_step_size, is_fast = False)
    support = self.amplitude != 0
    self.evaluate_least_squares()
    self.amplitude *= support
    self.reconstruction_type = "Informed least squares"
    C.finished("Informed least squares")

  def evaluate_fista_ayanzadeh(self, expected_amplitude = 995.5, expected_frequency = 5025, expected_error_measurement = 11.87, backtrack_scale = 0.9, norm_scale_factor_modifiers = np.geomspace(1.0, 3.0, 10)):
    C.starting("FISTA (Ayanzadeh)")
    amplitudes = []
    for norm_scale_factor_modifier in norm_scale_factor_modifiers:
      C.print(f"{norm_scale_factor_modifier}")
      self.evaluate_fista_backtracking(expected_amplitude = expected_amplitude, expected_frequency = expected_frequency, expected_error_measurement = expected_error_measurement, norm_scale_factor_modifier = norm_scale_factor_modifier)
      amplitudes.append(self.amplitude.copy())

    self.amplitude *= 0
    for amplitude in amplitudes:
      self.amplitude += amplitude
    self.amplitude /= len(amplitudes)
    C.finished("FISTA (Ayanzadeh)")

  def evaluate_fista(self):
    C.starting("reconstruction (FISTA)")
    execution_time_endpoints = np.zeros(2, np.float64)

    execution_time_endpoints[0] = tm.time()

    threads_per_block = 128
    blocks_per_grid_time = (self.time_properties.time_coarse.size + (threads_per_block - 1)) // threads_per_block
    blocks_per_grid_frequency = (self.frequency.size + (threads_per_block - 1)) // threads_per_block

    iteration_max = 100
    scale = self.time_properties.time_step_coarse/(self.time_properties.time_step_coarse*(self.time_properties.time_coarse.size + 1))
    # (self.time_properties.time_end_points[1] - self.time_properties.time_end_points[0])
    
    fourier_transform = cuda.device_array((self.frequency.size, self.time_properties.time_coarse.size), np.float64)
    evaluate_fourier_transform[blocks_per_grid_time, threads_per_block](cuda.to_device(self.frequency), cuda.to_device(self.time_properties.time_coarse), fourier_transform, scale)

    self.amplitude = np.zeros(self.time_properties.time_coarse.size, np.float64)
    amplitude = cuda.to_device(self.amplitude)
    self.amplitude_previous = np.zeros(self.time_properties.time_coarse.size, np.float64)
    amplitude_previous = cuda.to_device(self.amplitude_previous)

    frequency_amplitude = cuda.to_device(self.frequency_amplitude)

    frequency_amplitude_prediction = np.zeros(self.frequency_amplitude.size, np.float64)
    frequency_amplitude_prediction = cuda.to_device(frequency_amplitude_prediction)

    fast_step_size = 1
    fast_step_size_previous = 1

    for iteration_index in range(iteration_max):
      evaluate_frequency_amplitude_prediction[blocks_per_grid_frequency, threads_per_block](amplitude, fourier_transform, frequency_amplitude_prediction)
      fast_step_size_previous = fast_step_size
      fast_step_size = (1 + math.sqrt(1 + 4*fast_step_size**2))/2
      # evaluate_next_iteration_fista[blocks_per_grid_time, threads_per_block](amplitude, amplitude_previous, frequency_amplitude, frequency_amplitude_prediction, fourier_transform, 1/(3 + 20*(iteration_index/iteration_max)), 0.75*scale, fast_step_size, fast_step_size_previous)
      evaluate_next_iteration_fista[blocks_per_grid_time, threads_per_block](amplitude, amplitude_previous, frequency_amplitude, frequency_amplitude_prediction, fourier_transform, 0.5, 0.1*iteration_max*scale, fast_step_size, fast_step_size_previous)

    self.amplitude = amplitude.copy_to_host()

    execution_time_endpoints[1] = tm.time()
    C.print(f"ReTm = {execution_time_endpoints[1] - execution_time_endpoints[0]:4.2f}")

    C.finished("reconstruction (FISTA)")

  def evaluate_fista_gradient(self, expected_amplitude = 995.5, expected_frequency = 5025, expected_error_measurement = 11.87, backtrack_scale = 0.9, norm_scale_factor_modifier = 2.0, is_fast = False, norm_scale_factor = None):
    
    # Define transforms
    time_mesh, frequency_mesh = np.meshgrid(self.time_properties.time_coarse, self.frequency)
    fourier_mesh = math.tau*time_mesh*frequency_mesh
    dst_matrix = self.time_properties.time_step_coarse/self.time_properties.time_end_points[1]*np.sin(fourier_mesh)
    frequency_derivative = math.tau*time_mesh*self.time_properties.time_step_coarse/self.time_properties.time_end_points[1]*np.cos(fourier_mesh)
    gradient_derivative = frequency_mesh*frequency_derivative

    fourier_mesh, time_mesh, frequency_mesh = None, None, None

    # Define problem space
    amplitude = np.zeros_like(self.time_properties.time_coarse)
    frequency_shift = 0
    gradient_shift = 0
    bias_shift = 0

    frequency_amplitude = self.frequency_amplitude.copy()
    gradient_step = 1

    # Define regularisation
    sparse_regularisation = norm_scale_factor
    if sparse_regularisation == None:
      sparse_regularisation = 1
    sparse_regularisation *= norm_scale_factor_modifier

    frequency_regularisation  = (1/40)**4
    gradient_regularisation   = (1/0.5)**2
    bias_regularisation       = (1/100)**2

    # Loop
    for index in range(1000):
      C.print(index)
      matrix = dst_matrix + frequency_shift*frequency_derivative + gradient_shift*gradient_derivative
      error = matrix@(amplitude + bias_shift) - frequency_amplitude
      amplitude_temp = amplitude - gradient_step*2*error.T@matrix
      amplitude_temp = (amplitude_temp - gradient_step*sparse_regularisation*np.sign(amplitude_temp))*(np.abs(amplitude_temp) > gradient_step*sparse_regularisation)
      frequency_shift -= 1e-2*gradient_step*2*(error.T@frequency_derivative@amplitude + frequency_regularisation*frequency_shift)
      gradient_shift -= 1e-9*gradient_step*2*(error.T@gradient_derivative@amplitude - gradient_regularisation*gradient_shift)
      bias_shift -= 1e-5*gradient_step*2*(np.sum(error.T@matrix) + bias_regularisation*bias_shift)
      amplitude = amplitude_temp
      C.print(np.max(np.abs(amplitude)))

    C.print(f"Frequency shift: {frequency_shift}")
    C.print(f"Gradient shift: {gradient_shift}")
    C.print(f"Bias shift: {bias_shift}")
    self.amplitude = amplitude

  def evaluate_coherence(self):
    # C.starting("sensing coherence evaluation")
    coherence_size = int(self.time_properties.time_coarse.size*(self.time_properties.time_coarse.size - 1)/2)
    # time_indices = np.empty((coherence_size, 2), np.int32)
    # for coherence_index in range(coherence_size):
    #   time_index_high = coherence_index
    #   time_indices_to_check = self.time_properties.time_coarse.size - 1
    #   time_index_low = 0
    #   while time_index_high >= time_indices_to_check:
    #     time_index_high -= time_indices_to_check
    #     time_indices_to_check -= 1
    #     time_index_low += 1
    #   time_index_high += time_index_low + 1
    #   time_indices[coherence_index, :] = np.array([time_index_low, time_index_high])
    #   # C.print(f"{time_index_low}, {time_index_high}")

    # C.print(f"coherence size = {coherence_size}")
    threads_per_block = 128
    blocks_per_grid_time = (self.time_properties.time_coarse.size + (threads_per_block - 1)) // threads_per_block
    blocks_per_grid_coherence = (coherence_size + (threads_per_block - 1)) // threads_per_block

    fourier_scale = self.time_properties.time_step_coarse/(self.time_properties.time_step_coarse*(self.time_properties.time_coarse.size + 1))
    # (self.time_properties.time_end_points[1] - self.time_properties.time_end_points[0])
    fourier_transform = cuda.device_array((self.frequency.size, self.time_properties.time_coarse.size), np.float64)
    coherence = cuda.device_array((coherence_size), np.float64)
    # C.print(f"Init")
    evaluate_fourier_transform[blocks_per_grid_time, threads_per_block](cuda.to_device(self.frequency), cuda.to_device(self.time_properties.time_coarse), fourier_transform, fourier_scale)
    # C.print(f"FT")
    evaluate_fourier_transform_unit[blocks_per_grid_time, threads_per_block](fourier_transform)
    # C.print(f"Unit")
    evaluate_coherence[blocks_per_grid_coherence, threads_per_block](fourier_transform, coherence)
    # C.print(f"Coherence")
    coherence = coherence.copy_to_host()
    # print(coherence)
    # C.finished("sensing coherence evaluation")
    return np.max(coherence)

    # fourier_transform = fourier_transform.copy_to_host()

    

@cuda.jit()
def evaluate_fourier_transform(frequency, time_coarse, fourier_transform, scale):
  time_index = cuda.threadIdx.x + cuda.blockIdx.x*cuda.blockDim.x
  if time_index < time_coarse.size:
    for frequency_index in range(frequency.size):
      fourier_transform[frequency_index, time_index] = scale*math.sin(math.tau*frequency[frequency_index]*time_coarse[time_index])

@cuda.jit()
def evaluate_derivative_fourier_transform(frequency, time_coarse, derivative_fourier_transform, scale):
  time_index = cuda.threadIdx.x + cuda.blockIdx.x*cuda.blockDim.x
  if time_index < time_coarse.size:
    for frequency_index in range(frequency.size):
      derivative_fourier_transform[frequency_index, time_index] = math.tau*time_coarse[time_index]*scale*math.cos(math.tau*frequency[frequency_index]*time_coarse[time_index])

@cuda.jit()
def evaluate_coherence(fourier_transform, coherence):
  coherence_index = cuda.threadIdx.x + cuda.blockIdx.x*cuda.blockDim.x
  if coherence_index < coherence.size:
    time_index_high = coherence_index
    time_indices_to_check = fourier_transform.shape[1] - 1
    time_index_low = 0
    while time_index_high >= time_indices_to_check:
      time_index_high -= time_indices_to_check
      time_indices_to_check -= 1
      time_index_low += 1
    time_index_high += time_index_low + 1

    coherence[coherence_index] = 0
    coherence_temp = 0
    for frequency_index in range(fourier_transform.shape[0]):
      # coherence_temp += fourier_transform[frequency_index, time_indices[coherence_index, 0]]*fourier_transform[frequency_index, time_indices[coherence_index, 1]]
      coherence_temp += fourier_transform[frequency_index, time_index_low]*fourier_transform[frequency_index, time_index_high]
    coherence[coherence_index] = abs(coherence_temp)

@cuda.jit()
def evaluate_fourier_transform_unit(fourier_transform):
  time_index = cuda.threadIdx.x + cuda.blockIdx.x*cuda.blockDim.x
  if time_index < fourier_transform.shape[1]:
    norm = 0
    for frequency_index in range(fourier_transform.shape[0]):
      norm += fourier_transform[frequency_index, time_index]**2
    norm = math.sqrt(norm)
    for frequency_index in range(fourier_transform.shape[0]):
      fourier_transform[frequency_index, time_index] /= norm

@cuda.jit()
def evaluate_frequency_amplitude_prediction(amplitude, fourier_transform, frequency_amplitude_prediction):
  frequency_index = cuda.threadIdx.x + cuda.blockIdx.x*cuda.blockDim.x
  if frequency_index < frequency_amplitude_prediction.size:
    frequency_amplitude_prediction[frequency_index] = 0
    for time_index in range(amplitude.size):
      frequency_amplitude_prediction[frequency_index] += fourier_transform[frequency_index, time_index]*amplitude[time_index]

@cuda.jit()
def evaluate_frequency_amplitude_prediction_frequency_fit(amplitude, frequency, frequency_amplitude, fourier_transform, derivative_fourier_transform, frequency_amplitude_prediction, frequency_fit_step_size):
  frequency_index = cuda.threadIdx.x + cuda.blockIdx.x*cuda.blockDim.x
  if frequency_index < frequency_amplitude_prediction.size:
    frequency_amplitude_prediction[frequency_index] = 0
    derivative_frequency_amplitude_prediction = 0
    for time_index in range(amplitude.size):
      frequency_amplitude_prediction[frequency_index] += fourier_transform[frequency_index, time_index]*amplitude[time_index]
      derivative_frequency_amplitude_prediction += derivative_fourier_transform[frequency_index, time_index]*amplitude[time_index]

    frequency_shift_step = frequency_fit_step_size*2*derivative_frequency_amplitude_prediction*(frequency_amplitude_prediction[frequency_index] - frequency_amplitude[frequency_index])
    # frequency_shift_step = frequency_fit_step_size*derivative_frequency_amplitude_prediction
    frequency[frequency_index] -= frequency_shift_step
    for time_index in range(amplitude.size):
      fourier_transform[frequency_index, time_index] -= frequency_shift_step*derivative_fourier_transform[frequency_index, time_index]

@cuda.jit()
def evaluate_frequency_amplitude_prediction_fit(amplitude, fourier_transform, frequency_amplitude_prediction, frequency, fit_parameters, line_noise_derivative_amplitude, time_end, rabi_frequency_readout, frequency_line_noise):
  frequency_index = cuda.threadIdx.x + cuda.blockIdx.x*cuda.blockDim.x
  if frequency_index < frequency_amplitude_prediction.size:
    frequency_amplitude_prediction[frequency_index] = 0

    # Evaluate Fourier transform
    for time_index in range(amplitude.size):
      frequency_amplitude_prediction[frequency_index] += fourier_transform[frequency_index, time_index]*amplitude[time_index]

    # Evaluate noise fit
    noise_end_sample = fit_parameters[0]*math.sin(math.tau*frequency_line_noise*time_end)

    true_rabi = math.sqrt(frequency[frequency_index]**2 + noise_end_sample**2)
    cos_tilt = frequency[frequency_index]/true_rabi
    sin_tilt = noise_end_sample/true_rabi

    true_rabi_readout = math.sqrt(rabi_frequency_readout**2 + noise_end_sample**2)
    cos_tilt_readout = rabi_frequency_readout/true_rabi_readout
    sin_tilt_readout = noise_end_sample/true_rabi_readout

    rabi_phase = math.tau*(frequency[frequency_index] + (fit_parameters[0]**2)/(4*frequency[frequency_index]))*time_end
    cos_rabi_phase = math.cos(rabi_phase)
    tan_rabi_phase = math.tan(rabi_phase)

    readout = math.pi/(2*cos_tilt_readout)
    cos_readout = math.cos(readout)
    sin_readout = math.sin(readout)

    noise_fit = 1/(math.tau*time_end)*cos_rabi_phase*(cos_tilt*cos_readout - sin_tilt*sin_readout)

    frequency_amplitude_prediction[frequency_index] += noise_fit

    line_noise_derivative_amplitude[frequency_index] = (-1/fit_parameters[0])*((sin_tilt**2 + 2*time_end*(math.tau*fit_parameters[0])**2*tan_rabi_phase/(math.tau*frequency[frequency_index]))*noise_fit + 1/(math.tau*time_end)*cos_rabi_phase*sin_tilt*(sin_readout + math.pi/2*sin_tilt_readout*(noise_end_sample*cos_readout + frequency[frequency_index]*sin_readout)/rabi_frequency_readout))

@cuda.jit()
def evaluate_next_iteration_ista(amplitude, frequency_amplitude, frequency_amplitude_prediction, fourier_transform, fourier_scale, norm_scale_factor, reconstruction_step):
  time_index = cuda.threadIdx.x + cuda.blockIdx.x*cuda.blockDim.x
  if time_index < amplitude.size:
    for frequency_index in range(frequency_amplitude.size):
      amplitude[time_index] += fourier_transform[frequency_index, time_index]*((frequency_amplitude[frequency_index] - frequency_amplitude_prediction[frequency_index])/fourier_scale)*reconstruction_step

    if amplitude[time_index] > norm_scale_factor*reconstruction_step:
      amplitude[time_index] -= norm_scale_factor*reconstruction_step
    elif amplitude[time_index] < -norm_scale_factor*reconstruction_step:
      amplitude[time_index] += norm_scale_factor*reconstruction_step
    else:
      amplitude[time_index] = 0

@cuda.jit()
def evaluate_next_iteration_ista_shrink_scale(amplitude, frequency_amplitude, frequency_amplitude_prediction, fourier_transform, norm_scale_factor, reconstruction_step, shrink_scale):
  time_index = cuda.threadIdx.x + cuda.blockIdx.x*cuda.blockDim.x
  if time_index < amplitude.size:
    for frequency_index in range(frequency_amplitude.size):
      amplitude[time_index] += 2*fourier_transform[frequency_index, time_index]*(frequency_amplitude[frequency_index] - frequency_amplitude_prediction[frequency_index])*reconstruction_step

    if amplitude[time_index] > norm_scale_factor*reconstruction_step:
      amplitude[time_index] -= norm_scale_factor*reconstruction_step
    elif amplitude[time_index] < -norm_scale_factor*reconstruction_step:
      amplitude[time_index] += norm_scale_factor*reconstruction_step
    else:
      amplitude[time_index] = 0
    amplitude[time_index] *= shrink_scale

@cuda.jit()
def evaluate_next_iteration_ista_adaptive(amplitude, frequency_amplitude, frequency_amplitude_prediction, fourier_transform, norm_scale_factor, reconstruction_step, weights):
  time_index = cuda.threadIdx.x + cuda.blockIdx.x*cuda.blockDim.x
  if time_index < amplitude.size:
    for frequency_index in range(frequency_amplitude.size):
      amplitude[time_index] += 2*fourier_transform[frequency_index, time_index]*(frequency_amplitude[frequency_index] - frequency_amplitude_prediction[frequency_index])*reconstruction_step

    if amplitude[time_index] > weights[time_index]*norm_scale_factor*reconstruction_step:
      amplitude[time_index] -= weights[time_index]*norm_scale_factor*reconstruction_step
    elif amplitude[time_index] < -weights[time_index]*norm_scale_factor*reconstruction_step:
      amplitude[time_index] += weights[time_index]*norm_scale_factor*reconstruction_step
    else:
      amplitude[time_index] = 0

@cuda.jit()
def evaluate_next_iteration_ista_fit(amplitude, frequency_amplitude, frequency_amplitude_prediction, fourier_transform, norm_scale_factor, reconstruction_step, shrink_scale, fit_parameters, line_noise_derivative_amplitude):
  time_index = cuda.threadIdx.x + cuda.blockIdx.x*cuda.blockDim.x
  if time_index < amplitude.size:
    for frequency_index in range(frequency_amplitude.size):
      amplitude[time_index] += 2*fourier_transform[frequency_index, time_index]*(frequency_amplitude[frequency_index] - frequency_amplitude_prediction[frequency_index])*reconstruction_step

    if amplitude[time_index] > norm_scale_factor*reconstruction_step:
      amplitude[time_index] -= norm_scale_factor*reconstruction_step
    elif amplitude[time_index] < -norm_scale_factor*reconstruction_step:
      amplitude[time_index] += norm_scale_factor*reconstruction_step
    else:
      amplitude[time_index] = 0
    amplitude[time_index] *= shrink_scale

  if time_index == amplitude.size:
    for frequency_index in range(frequency_amplitude.size):
      fit_parameters[0] += 2*line_noise_derivative_amplitude[frequency_index]*(frequency_amplitude[frequency_index] - frequency_amplitude_prediction[frequency_index])*reconstruction_step*8e-2

@cuda.jit()
def evaluate_fista_fast_step(amplitude, amplitude_previous, fast_step_size, fast_step_size_previous):
  time_index = cuda.threadIdx.x + cuda.blockIdx.x*cuda.blockDim.x
  if time_index < amplitude.size:
    amplitude[time_index] += ((fast_step_size_previous - 1)/fast_step_size)*(amplitude[time_index] - amplitude_previous[time_index])

@cuda.jit()
def evaluate_fista_fast_step_fit(amplitude, amplitude_previous, fast_step_size, fast_step_size_previous, fit_parameters, fit_parameters_previous):
  time_index = cuda.threadIdx.x + cuda.blockIdx.x*cuda.blockDim.x
  if time_index < amplitude.size:
    amplitude[time_index] += ((fast_step_size_previous - 1)/fast_step_size)*(amplitude[time_index] - amplitude_previous[time_index])
  elif time_index == amplitude.size:
    fit_parameters[0] += ((fast_step_size_previous - 1)/fast_step_size)*(fit_parameters[0] - fit_parameters_previous[0])

@cuda.jit()
def evaluate_next_iteration_fista(amplitude, amplitude_previous, frequency_amplitude, frequency_amplitude_prediction, fourier_transform, soft_step_size, scale, fast_step_size, fast_step_size_previous):
  time_index = cuda.threadIdx.x + cuda.blockIdx.x*cuda.blockDim.x
  if time_index < amplitude.size:
    amplitude_previous[time_index] = amplitude[time_index]

    for frequency_index in range(frequency_amplitude.size):
      amplitude[time_index] += fourier_transform[frequency_index, time_index]*(frequency_amplitude[frequency_index] - frequency_amplitude_prediction[frequency_index])/scale

    if amplitude[time_index] > soft_step_size/scale:
      amplitude[time_index] -= soft_step_size/scale
    elif amplitude[time_index] < -soft_step_size/scale:
      amplitude[time_index] += soft_step_size/scale
    else:
      amplitude[time_index] = 0

    amplitude[time_index] += ((fast_step_size_previous - 1)/fast_step_size)*(amplitude[time_index] - amplitude_previous[time_index])

@cuda.jit()
def copy_amplitude(amplitude_input, amplitude_output):
  time_index = cuda.threadIdx.x + cuda.blockIdx.x*cuda.blockDim.x
  if time_index < amplitude_output.size:
    amplitude_output[time_index] = amplitude_input[time_index]

@cuda.jit()
def copy_amplitude_fit(amplitude_input, amplitude_output, fit_parameters_input, fit_parameters_output):
  time_index = cuda.threadIdx.x + cuda.blockIdx.x*cuda.blockDim.x
  if time_index < amplitude_output.size:
    amplitude_output[time_index] = amplitude_input[time_index]
  elif time_index == amplitude_output.size:
    fit_parameters_output[0] = fit_parameters_input[0]

@cuda.jit()
def subtract_constant(amplitude, constant):
  time_index = cuda.threadIdx.x + cuda.blockIdx.x*cuda.blockDim.x
  if time_index < amplitude.size:
    amplitude[time_index] -= constant

def run_reconstruction_subsample_sweep(expected_signal:TestSignal, experiment_results:ExperimentResults, sweep_parameters = (30, 10000, 10), archive:Archive = None, frequency_cutoff_low = 0, frequency_cutoff_high = 100000, random_seeds = [util.Seeds.metroid], evaluation_methods = [], expected_amplitude = None, expected_frequency = None, expected_error_measurement = None, rabi_frequency_readout = None, frequency_line_noise = None, norm_scale_factor_modifier = None, frequency_fit_step_size = 1, units = "Hz",ramsey_comparison_results = None, metrics = ["rmse"]):
  reconstruction = Reconstruction(expected_signal.time_properties)

  random_seeds = np.array(random_seeds)
  numbers_of_samples = []
  sweep_samples = range(min(sweep_parameters[1], experiment_results.frequency.size), sweep_parameters[0], -sweep_parameters[2])
  for reconstruction_index, number_of_samples in enumerate(sweep_samples):
      numbers_of_samples.append(number_of_samples)
  numbers_of_samples = np.array(numbers_of_samples)

  C.starting("number of samples sweep")
  # numbers_of_samples = []
  amplitudes = []
  coherence = [[] for random_seed in random_seeds]
  for evaluation_method_index, evaluation_method in enumerate(evaluation_methods):
    for reconstruction_index, number_of_samples in enumerate(sweep_samples):
      # numbers_of_samples.append(number_of_samples)
      for random_index, random_seed in enumerate(random_seeds):
        # Initialise reconstruction
        reconstruction.read_frequencies_from_experiment_results(experiment_results, number_of_samples, frequency_cutoff_low = frequency_cutoff_low, frequency_cutoff_high = frequency_cutoff_high, random_seed = random_seed)

        # Evaluate measurement coherence
        if evaluation_method_index == 0:
          coherence[random_index].append(reconstruction.evaluate_coherence())
        
        # Evaluate resconstructions
        if evaluation_method == "least_squares":
          reconstruction.evaluate_least_squares()
        elif evaluation_method == "fista_ayanzadeh":
          reconstruction.evaluate_fista_ayanzadeh(expected_amplitude = expected_amplitude, expected_frequency = expected_frequency, expected_error_measurement = expected_error_measurement, is_fast = True)
        elif evaluation_method == "ista_ayanzadeh":
          reconstruction.evaluate_fista_ayanzadeh(expected_amplitude = expected_amplitude, expected_frequency = expected_frequency, expected_error_measurement = expected_error_measurement, is_fast = False)
        elif evaluation_method == "fista_fit":
          reconstruction.evaluate_fista_fit(expected_amplitude = expected_amplitude, expected_frequency = expected_frequency, expected_error_measurement = expected_error_measurement, rabi_frequency_readout = rabi_frequency_readout, frequency_line_noise = frequency_line_noise, norm_scale_factor_modifier = norm_scale_factor_modifier, is_fast = True)
        elif evaluation_method == "ista_fit":
          reconstruction.evaluate_fista_fit(expected_amplitude = expected_amplitude, expected_frequency = expected_frequency, expected_error_measurement = expected_error_measurement, rabi_frequency_readout = rabi_frequency_readout, frequency_line_noise = frequency_line_noise, norm_scale_factor_modifier = norm_scale_factor_modifier, is_fast = False)
        elif evaluation_method == "fista_adaptive":
          reconstruction.evaluate_fista_adaptive(expected_amplitude = expected_amplitude, expected_frequency = expected_frequency, expected_error_measurement = expected_error_measurement, norm_scale_factor_modifier = norm_scale_factor_modifier, is_fast = True)
        elif evaluation_method == "ista_adaptive":
          reconstruction.evaluate_fista_adaptive(expected_amplitude = expected_amplitude, expected_frequency = expected_frequency, expected_error_measurement = expected_error_measurement, norm_scale_factor_modifier = norm_scale_factor_modifier, is_fast = False)
        elif evaluation_method == "fista_informed_least_squares":
          reconstruction.evaluate_informed_least_squares(informed_type = "fista", expected_amplitude = expected_amplitude, expected_frequency = expected_frequency, expected_error_measurement = expected_error_measurement, norm_scale_factor_modifier = norm_scale_factor_modifier)
        elif evaluation_method == "ista_informed_least_squares":
          reconstruction.evaluate_informed_least_squares(informed_type = "ista", expected_amplitude = expected_amplitude, expected_frequency = expected_frequency, expected_error_measurement = expected_error_measurement, norm_scale_factor_modifier = norm_scale_factor_modifier)
        elif evaluation_method == "fadaptive_informed_least_squares":
          reconstruction.evaluate_informed_least_squares(informed_type = "fista_adaptive", expected_amplitude = expected_amplitude, expected_frequency = expected_frequency, expected_error_measurement = expected_error_measurement, norm_scale_factor_modifier = norm_scale_factor_modifier)
        elif evaluation_method == "adaptive_informed_least_squares":
          reconstruction.evaluate_informed_least_squares(informed_type = "ista_adaptive", expected_amplitude = expected_amplitude, expected_frequency = expected_frequency, expected_error_measurement = expected_error_measurement, norm_scale_factor_modifier = norm_scale_factor_modifier)
        elif evaluation_method == "fadaptive_frequency_fit":
          reconstruction.evaluate_informed_least_squares(informed_type = "fista_frequency_fit", expected_amplitude = expected_amplitude, expected_frequency = expected_frequency, expected_error_measurement = expected_error_measurement, norm_scale_factor_modifier = norm_scale_factor_modifier, frequency_fit_step_size = frequency_fit_step_size)
        elif evaluation_method == "adaptive_frequency_fit":
          reconstruction.evaluate_informed_least_squares(informed_type = "ista_frequency_fit", expected_amplitude = expected_amplitude, expected_frequency = expected_frequency, expected_error_measurement = expected_error_measurement, norm_scale_factor_modifier = norm_scale_factor_modifier, frequency_fit_step_size = frequency_fit_step_size)
        elif evaluation_method == "fista_backtracking":
          # reconstruction.evaluate_fista_backtracking(expected_amplitude = expected_amplitude, expected_frequency = expected_frequency, expected_error_measurement = expected_error_measurement, norm_scale_factor_modifier = norm_scale_factor_modifier, is_fast = True)
          reconstruction.evaluate_fista_backtracking(expected_amplitude = expected_amplitude, expected_frequency = expected_frequency, expected_error_measurement = expected_error_measurement, norm_scale_factor_modifier = norm_scale_factor_modifier, is_fast = True,
            # norm_scale_factor = 0.8387421955548435
            # norm_scale_factor = 1.291549665014884
            # norm_scale_factor = 3.313982602739096
            # norm_scale_factor = 1.123810254675881
            norm_scale_factor = 0.7488103857590022
          )
        elif evaluation_method == "ista_backtracking":
          reconstruction.evaluate_fista_backtracking(expected_amplitude = expected_amplitude, expected_frequency = expected_frequency, expected_error_measurement = expected_error_measurement, norm_scale_factor_modifier = norm_scale_factor_modifier, is_fast = True)
        reconstruction.write_to_file(archive.archive_file, (numbers_of_samples.size*evaluation_method_index + reconstruction_index)*random_seeds.size + random_index)

        amplitudes.append(reconstruction.amplitude.copy())

    # # Fit coherence
    # arcsech_densities = []
    # for random_index, random_seed in enumerate(random_seeds):
    #   coherence[random_index] = np.array(coherence[random_index])
    #   max_number = max(expected_signal.time_properties.time_coarse.size, np.max(numbers_of_samples))
    #   arcsech_density = np.arccosh(max_number/numbers_of_samples)
    #   arcsech_densities.append(arcsech_density)
    
    # arcsech_density_full = np.array(arcsech_densities)
    # arcsech_density_full = arcsech_density_full.flatten() #.reshape((arcsech_density_full.size, 1))
    # coherence_full = np.array(coherence)
    # coherence_full = coherence_full.flatten()

    # # coherence_coefficient = np.linalg.lstsq(arcsech_density_full, coherence_full)[0][0]
    # coherence_coefficient = np.dot(arcsech_density_full, coherence_full)/np.dot(arcsech_density_full, arcsech_density_full)


  C.finished("number of samples sweep")

  C.starting("error analysis")
  expected_signal_power = np.mean(expected_signal.amplitude**2)

  template_time = np.arange(0, 1/expected_frequency, expected_signal.time_properties.time_step_coarse)
  template_amplitude = expected_amplitude*np.sin(math.tau*expected_frequency*template_time)
  template_energy = np.sum(template_amplitude**2)
  matched_cutoff = template_energy/2
  
  if "rmse" in metrics:
    errors_method_2 = []
    stdevs_method_2 = []

  if "norms" in metrics:
    errors_method_0 = []
    errors_method_1 = []
    errors_method_sup = []
    errors_method_snr = []
    stdevs_method_0 = []
    stdevs_method_1 = []
    stdevs_method_sup = []
    stdevs_method_snr = []

  if "mf" in metrics:
    errors_method_mf = []
    errors_method_mfd = []
    errors_method_mfp = []
    errors_method_mfpd = []
    stdevs_method_mf = []
    stdevs_method_mfd = []
    stdevs_method_mfp = []
    stdevs_method_mfpd = []

    mf_cutoff = expected_signal_power/2 # default
    # mf_cutoff = expected_signal_power/4
    mfp_cutoff = mf_cutoff

  if "confusion_fixed" in metrics:
    errors_method_sensitivity = []
    errors_method_specificity = []
    stdevs_method_sensitivity = []
    stdevs_method_specificity = []
    method_matched_decisions = []

    matched_ground_truth = scipy.signal.correlate(expected_signal.amplitude, template_amplitude) >= matched_cutoff

  if "roc" in metrics:
    errors_method_roc_auc = []
    errors_method_roc_sensitivity = []
    errors_method_roc_specificity = []
    errors_method_roc_amplitude = []
    stdevs_method_roc_auc = []
    stdevs_method_roc_sensitivity = []
    stdevs_method_roc_specificity = []
    stdevs_method_roc_amplitude = []

    roc_cutoff_max = np.sum(template_amplitude**2)*8
    # roc_cutoff_max = np.sum(template_amplitude**2)*1e2
    roc_cutoff_min = np.sum(template_amplitude**2)*1e-4
    # roc_cutoff_min = np.sum(template_amplitude**2)*1e-30
    # roc_resolution = 100
    roc_resolution = 1000
    roc_ground_truth = scipy.signal.correlate(expected_signal.amplitude, template_amplitude) >= matched_cutoff

  for evaluation_method_index, evaluation_method in enumerate(evaluation_methods):
    if "rmse" in metrics:
      errors_2 = []
      stdevs_2 = []
    if "norms" in metrics:
      errors_0 = []
      errors_1 = []
      errors_sup = []
      errors_snr = []
      stdevs_0 = []
      stdevs_1 = []
      stdevs_sup = []
      stdevs_snr = []
    if "mf" in metrics:
      errors_mf = []
      errors_mfd = []
      errors_mfp = []
      errors_mfpd = []
      stdevs_mf = []
      stdevs_mfd = []
      stdevs_mfp = []
      stdevs_mfpd = []
    if "confusion_fixed" in metrics:
      errors_sensitivity = []
      errors_specificity = []
      stdevs_sensitivity = []
      stdevs_specificity = []
      matched_decisions = []
    if "roc" in metrics:
      errors_roc_sensitivity = []
      errors_roc_specificity = []
      errors_roc_auc = []
      errors_roc_amplitude = []
      stdevs_roc_sensitivity = []
      stdevs_roc_specificity = []
      stdevs_roc_auc = []
      stdevs_roc_amplitude = []
    for reconstruction_index, number_of_samples in enumerate(sweep_samples):
      if "rmse" in metrics:
        error_2 = []
      if "norms" in metrics:
        error_0   = []
        error_1   = []
        error_sup = []
        error_snr = []
      if "mf" in metrics:
        error_mf    = []
        error_mfd   = []
        error_mfp   = []
        error_mfpd  = []
      if "confusion_fixed" in metrics:
        error_sensitivity = []
        error_specificity = []
        matched_decision = []
      if "roc" in metrics:
        roc_sensitivities = []
        roc_specificities = []
        error_roc_auc = []
        error_roc_amplitude = []
      for random_index, random_seed in enumerate(random_seeds):
        amplitude = amplitudes[(numbers_of_samples.size*evaluation_method_index + reconstruction_index)*random_seeds.size + random_index]
        if "rmse" in metrics:
          error_2.append(math.sqrt(np.mean((amplitude - expected_signal.amplitude)**2)))
        if "norms" in metrics:
          error_0.append(np.mean((amplitude == 0)*(expected_signal.amplitude != 0) + (amplitude != 0)*(expected_signal.amplitude == 0)))
          error_1.append(np.mean(np.abs(amplitude - expected_signal.amplitude)))
          error_sup.append(np.max(np.abs(amplitude - expected_signal.amplitude)))
          error_snr.append(expected_signal_power/np.mean((amplitude - expected_signal.amplitude)**2))
        if "mf" in metrics:
          error_mf_current = np.mean(amplitude*expected_signal.amplitude)
          error_mf.append(np.mean(amplitude*expected_signal.amplitude))
          error_mfd.append(error_mf_current >= mf_cutoff)

          error_mfp_current = np.max(np.abs(scipy.signal.correlate((amplitude - (error_mf_current/expected_signal_power)*expected_signal.amplitude), expected_signal.amplitude)))/expected_signal.amplitude.size
          error_mfp.append(error_mfp_current)
          error_mfpd.append(error_mfp_current >= mfp_cutoff)

        if "confusion_fixed" in metrics:
          matched_decision.append(scipy.signal.correlate(amplitude, template_amplitude) >= matched_cutoff)
          if np.sum(matched_ground_truth) > 0:
            error_sensitivity.append(np.sum(np.logical_and(matched_ground_truth, matched_decision[-1]))/np.sum(matched_ground_truth))
          else:
            error_sensitivity.append(1)
          if np.sum(np.logical_not(matched_ground_truth)) > 0:
            error_specificity.append(np.sum(np.logical_and(np.logical_not(matched_ground_truth), np.logical_not(matched_decision[-1])))/np.sum(np.logical_not(matched_ground_truth)))
          else:
            error_specificity.append(1)

        if "roc" in metrics:
          roc_sensitivity = [1]
          roc_specificity = [0]
          roc_matched_filter_output = scipy.signal.correlate(amplitude, template_amplitude)
          # for roc_cutoff in (np.geomspace(roc_cutoff_min, roc_cutoff_max, roc_resolution) - 2*roc_cutoff_min):
          for roc_cutoff in (np.min(roc_matched_filter_output) + np.geomspace(roc_cutoff_min, roc_cutoff_max, roc_resolution) - roc_cutoff_min):
            roc_decision = roc_matched_filter_output >= roc_cutoff
            # roc_decision = roc_matched_filter_output > roc_cutoff
            if np.sum(roc_ground_truth) > 0:
              roc_sensitivity.append(np.sum(np.logical_and(roc_ground_truth, roc_decision))/np.sum(roc_ground_truth))
            else:
              roc_sensitivity.append(1)
            if np.sum(np.logical_not(roc_ground_truth)) > 0:
              roc_specificity.append(np.sum(np.logical_and(np.logical_not(roc_ground_truth), np.logical_not(roc_decision)))/np.sum(np.logical_not(roc_ground_truth)))
            else:
              roc_specificity.append(1)
          roc_sensitivity.append(0)
          # roc_specificity.append(roc_specificity[-1])
          roc_specificity.append(1)
          # roc_sensitivity[0] = roc_sensitivity[1]
          roc_sensitivity[0] = 1
          roc_sensitivity.append(0)
          roc_specificity.append(0)

          roc_auc = 0
          for roc_index in range(len(roc_sensitivity) - 2):
            roc_auc -= roc_sensitivity[roc_index + 1]*roc_specificity[roc_index] - roc_sensitivity[roc_index]*roc_specificity[roc_index + 1]
          roc_auc /= 2

          roc_sensitivities.append(np.array(roc_sensitivity))
          roc_specificities.append(np.array(roc_specificity))
          error_roc_auc.append(roc_auc)

          error_roc_amplitude.append(np.max(roc_matched_filter_output*roc_ground_truth)/math.sqrt(template_energy))

      if "rmse" in metrics:
        errors_2.append(np.mean(error_2))
        stdevs_2.append(np.std(error_2))
      if "norms" in metrics:
        errors_0.append   (np.mean(error_0))
        errors_1.append   (np.mean(error_1))
        errors_sup.append (np.mean(error_sup))
        errors_snr.append (10*np.log10(np.mean(error_snr)))
        stdevs_0.append   (np.std(error_0))
        stdevs_1.append   (np.std(error_1))
        stdevs_sup.append (np.std(error_sup))
        stdevs_snr.append (10*np.log10(np.std(error_snr)))
      if "mf" in metrics:
        errors_mf.append  (np.mean(error_mf))
        errors_mfd.append (np.mean(error_mfd))
        errors_mfp.append (np.mean(error_mfp))
        errors_mfpd.append(np.mean(error_mfpd))
        stdevs_mf.append  (np.std(error_mf))
        stdevs_mfd.append (np.std(error_mfd))
        stdevs_mfp.append (np.std(error_mfp))
        stdevs_mfpd.append(np.std(error_mfpd))
      if "confusion_fixed" in metrics:
        errors_sensitivity.append(np.mean(error_sensitivity))
        errors_specificity.append(np.mean(error_specificity))
        stdevs_sensitivity.append(np.std(error_sensitivity))
        stdevs_specificity.append(np.std(error_specificity))
        matched_decisions.append(np.mean(matched_decision, axis = 0))
      if "roc" in metrics:
        errors_roc_auc.append(np.mean(error_roc_auc))
        errors_roc_sensitivity.append(np.mean(roc_sensitivities, axis = 0))
        errors_roc_specificity.append(np.mean(roc_specificities, axis = 0))
        errors_roc_amplitude.append(np.mean(error_roc_amplitude))
        stdevs_roc_auc.append(np.std(error_roc_auc))
        stdevs_roc_sensitivity.append(np.std(roc_sensitivities, axis = 0))
        stdevs_roc_specificity.append(np.std(roc_specificities, axis = 0))
        stdevs_roc_amplitude.append(np.std(error_roc_amplitude))
    
    if "rmse" in metrics:
      errors_method_2.append(np.array(errors_2))
      stdevs_method_2.append(np.array(stdevs_2))
    if "norms" in metrics:
      errors_method_0.append  (np.array(errors_0))
      errors_method_1.append  (np.array(errors_1))
      errors_method_sup.append(np.array(errors_sup))
      errors_method_snr.append(np.array(errors_snr))
      stdevs_method_0.append  (np.array(stdevs_0))
      stdevs_method_1.append  (np.array(stdevs_1))
      stdevs_method_sup.append(np.array(stdevs_sup))
      stdevs_method_snr.append(np.array(stdevs_snr))
    if "mf" in metrics:
      errors_method_mf.append   (np.array(errors_mf))
      errors_method_mfd.append  (np.array(errors_mfd))
      errors_method_mfp.append  (np.array(errors_mfp))
      errors_method_mfpd.append (np.array(errors_mfpd))
      stdevs_method_mf.append   (np.array(stdevs_mf))
      stdevs_method_mfd.append  (np.array(stdevs_mfd))
      stdevs_method_mfp.append  (np.array(stdevs_mfp))
      stdevs_method_mfpd.append (np.array(stdevs_mfpd))
    if "confusion_fixed" in metrics:
      errors_method_sensitivity.append(np.array(errors_sensitivity))
      errors_method_specificity.append(np.array(errors_specificity))
      stdevs_method_sensitivity.append(np.array(stdevs_sensitivity))
      stdevs_method_specificity.append(np.array(stdevs_specificity))
      method_matched_decisions.append(np.array(matched_decisions))
    if "roc" in metrics:
      errors_method_roc_auc.append        (np.array(errors_roc_auc))
      errors_method_roc_sensitivity.append(np.array(errors_roc_sensitivity))
      errors_method_roc_specificity.append(np.array(errors_roc_specificity))
      errors_method_roc_amplitude.append  (np.array(errors_roc_amplitude))
      stdevs_method_roc_auc.append        (np.array(stdevs_roc_auc))
      stdevs_method_roc_sensitivity.append(np.array(stdevs_roc_sensitivity))
      stdevs_method_roc_specificity.append(np.array(stdevs_roc_specificity))
      stdevs_method_roc_amplitude.append  (np.array(stdevs_roc_amplitude))
  C.finished("error analysis")   

  if archive:
    sweep_group = archive.archive_file.require_group("reconstruction_sweeps/number_of_samples")
    sweep_group["number_of_samples"] = numbers_of_samples
    # sweep_group["coherence_coefficient"] = [coherence_coefficient]
    for evaluation_method_index, evaluation_method in enumerate(evaluation_methods):
      evaluation_group = sweep_group.require_group(evaluation_method)
      if "rmse" in metrics:
        evaluation_group["error_2"] = errors_method_2[evaluation_method_index]
        evaluation_group["stdev_2"] = stdevs_method_2[evaluation_method_index]
      if "norms" in metrics:
        evaluation_group["error_0"]   = errors_method_0[evaluation_method_index]
        evaluation_group["error_1"]   = errors_method_1[evaluation_method_index]
        evaluation_group["error_sup"] = errors_method_sup[evaluation_method_index]
        evaluation_group["error_snr"] = errors_method_snr[evaluation_method_index]
        evaluation_group["stdev_0"]   = stdevs_method_0[evaluation_method_index]
        evaluation_group["stdev_1"]   = stdevs_method_1[evaluation_method_index]
        evaluation_group["stdev_sup"] = stdevs_method_sup[evaluation_method_index]
        evaluation_group["stdev_snr"] = stdevs_method_snr[evaluation_method_index]
      if "mf" in metrics:
        evaluation_group["error_mf"]    = errors_method_mf[evaluation_method_index]
        evaluation_group["error_mfd"]   = errors_method_mfd[evaluation_method_index]
        evaluation_group["error_mfp"]   = errors_method_mfp[evaluation_method_index]
        evaluation_group["error_mfpd"]  = errors_method_mfpd[evaluation_method_index]
        evaluation_group["stdev_mf"]    = stdevs_method_mf[evaluation_method_index]
        evaluation_group["stdev_mfd"]   = stdevs_method_mfd[evaluation_method_index]
        evaluation_group["stdev_mfp"]   = stdevs_method_mfp[evaluation_method_index]
        evaluation_group["stdev_mfpd"]  = stdevs_method_mfpd[evaluation_method_index]
      if "confusion_fixed" in metrics:
        evaluation_group["error_sensitivity"] = errors_method_sensitivity[evaluation_method_index]
        evaluation_group["error_specificity"] = errors_method_specificity[evaluation_method_index]
        evaluation_group["stdev_sensitivity"] = stdevs_method_sensitivity[evaluation_method_index]
        evaluation_group["stdev_specificity"] = stdevs_method_specificity[evaluation_method_index]
      if "roc" in metrics:
        evaluation_group["error_roc_auc"] =         errors_method_roc_auc[evaluation_method_index]
        evaluation_group["error_roc_sensitivity"] = errors_method_roc_sensitivity[evaluation_method_index]
        evaluation_group["error_roc_specificity"] = errors_method_roc_specificity[evaluation_method_index]
        evaluation_group["error_roc_amplitude"] =   errors_method_roc_amplitude[evaluation_method_index]
        evaluation_group["stdev_roc_auc"] =         stdevs_method_roc_auc[evaluation_method_index]
        evaluation_group["stdev_roc_sensitivity"] = stdevs_method_roc_sensitivity[evaluation_method_index]
        evaluation_group["stdev_roc_specificity"] = stdevs_method_roc_specificity[evaluation_method_index]
        evaluation_group["stdev_roc_amplitude"] =   stdevs_method_roc_amplitude[evaluation_method_index]

  evaluation_method_labels = {
    "least_squares" : "Least squares",
    "ramsey" : "Ramsey",

    "fista_ayanzadeh" : "FISTA (Ayanzadeh)",
    "fista_adaptive" : "FISTA (Adaptive)",
    "fista_backtracking" : "FISTA (Backtracking)",
    "fista_fit" : "FISTA (Line noise fit)",
    "fista" : "FISTA",
    "fista_informed_least_squares" : "FISTA informed least squares",
    "fadaptive_informed_least_squares" : "Sandwich (fast)",
    "fadaptive_frequency_fit" : "Sandwich frequency fit (fast)",

    "ista_ayanzadeh" : "ISTA (Ayanzadeh)",
    "ista_adaptive" : "ISTA (Adaptive)",
    "ista_backtracking" : "ISTA (Backtracking)",
    "ista_fit" : "ISTA (Line noise fit)",
    "ista" : "ISTA",
    "ista_informed_least_squares" : "ISTA informed least squares",
    "adaptive_informed_least_squares" : "Sandwich",
    "adaptive_frequency_fit" : "Sandwich frequency fit"
  }
  # legend = []
  # for evaluation_method in evaluation_methods:
  #   legend.append(evaluation_method_labels[evaluation_method])
  # if ramsey_comparison_results is not None:
  #   legend = ["Ramsey"] + legend

  evaluation_method_legend = {
    "least_squares" : "g-",
    "ramsey" : "mo",

    "fista_backtracking" : "r-",
    "fista_fit" : "r-x",
    "fista_ayanzadeh" : "y-x",
    "fista_adaptive" : "y-",
    "fista" : "c--x",
    "fista_informed_least_squares" : "g--x",
    "fadaptive_informed_least_squares" : "g-.x",
    "fadaptive_frequency_fit" : "r-.x",

    "ista_backtracking" : "c-+",
    "ista_fit" : "r-+",
    "ista_ayanzadeh" : "y-+",
    "ista_adaptive" : "g-+",
    "ista" : "c--+",
    "ista_informed_least_squares" : "g--+",
    "adaptive_informed_least_squares" : "g-.+",
    "adaptive_frequency_fit" : "r-.+",
  }

  if "Hz" in units:
    unit_factor = 1
  elif "T" in units:
    unit_factor = 1/7e9
  if "n" in units:
    unit_factor *= 1e9
  elif "μ" in units:
    unit_factor *= 1e6
  elif "m" in units:
    unit_factor *= 1e3

  expected_amplitude *= unit_factor

  uncertainty_alpha = 0.25

  # === Plot heat maps ===
  time_mesh, samples_mesh = np.meshgrid(reconstruction.time_properties.time_coarse, sweep_samples)
  amplitude_mesh = np.empty((len(sweep_samples), amplitudes[0].size))
  residual_mesh = np.empty_like(amplitude_mesh)
  for evaluation_method_index, evaluation_method in enumerate(evaluation_methods):
    for reconstruction_index, number_of_samples in enumerate(sweep_samples):
        amplitude_mesh[reconstruction_index, :] = amplitudes[(numbers_of_samples.size*evaluation_method_index + reconstruction_index)*random_seeds.size]*unit_factor
        residual_mesh[reconstruction_index, :] = (amplitudes[(numbers_of_samples.size*evaluation_method_index + reconstruction_index)*random_seeds.size] - expected_signal.amplitude)*unit_factor
    
    plt.figure()
    plt.subplot(2, 1, 2)
    neural_pulse = expected_signal.neural_pulses[0]
    plt.pcolormesh(time_mesh*1e3, samples_mesh, amplitude_mesh, cmap = "seismic", vmin = -2*expected_amplitude, vmax = 2*expected_amplitude)
    plt.xlabel("Time (ms)")
    plt.ylabel("Number of samples")
    plt.xlim([(neural_pulse.time_start - 0.5/neural_pulse.frequency)*1e3, (neural_pulse.time_start + 1.5/neural_pulse.frequency)*1e3])
    colour_bar = plt.colorbar()
    colour_bar.set_label(f"Amplitude ({units})")

    plt.subplot(2, 1, 1)
    plt.pcolormesh(time_mesh*1e3, samples_mesh, amplitude_mesh, cmap = "seismic", vmin = -2*expected_amplitude, vmax = 2*expected_amplitude)
    plt.ylabel("Number of samples")

    if archive:
      archive.write_plot(f"Sweeping the number of samples used in reconstruction\n{evaluation_method_labels[evaluation_method]}", f"number_of_samples_{evaluation_method}")
    plt.draw()

    plt.figure()
    plt.subplot(2, 1, 2)
    neural_pulse = expected_signal.neural_pulses[0]
    plt.pcolormesh(time_mesh*1e3, samples_mesh, residual_mesh, cmap = "seismic", vmin = -2*expected_amplitude, vmax = 2*expected_amplitude)
    plt.xlabel("Time (ms)")
    plt.ylabel("Number of samples")
    plt.xlim([(neural_pulse.time_start - 0.5/neural_pulse.frequency)*1e3, (neural_pulse.time_start + 1.5/neural_pulse.frequency)*1e3])
    colour_bar = plt.colorbar()
    colour_bar.set_label(f"Residual ({units})")

    plt.subplot(2, 1, 1)
    plt.pcolormesh(time_mesh*1e3, samples_mesh, residual_mesh, cmap = "seismic", vmin = -2*expected_amplitude, vmax = 2*expected_amplitude)
    plt.ylabel("Number of samples")

    if archive:
      archive.write_plot(f"Sweeping the number of samples used in reconstruction\n{evaluation_method_labels[evaluation_method]}, Residual", f"number_of_samples_{evaluation_method}_residual")
    plt.draw()
  
  # === Plot metrics ===
  if "rmse" in metrics:
    plt.figure()
    plt.subplot()
    plt.plot([numbers_of_samples[0], numbers_of_samples[-1]], [math.sqrt(expected_signal_power)]*2, "b--", label = "Ground truth RMS amplitude")
    if ramsey_comparison_results is not None:
      if hasattr(ramsey_comparison_results, "error_2"):
        plt.plot([numbers_of_samples[0]], [ramsey_comparison_results.error_2], evaluation_method_legend["ramsey"], label = evaluation_method_labels["ramsey"])
    for evaluation_method_index, evaluation_method in enumerate(evaluation_methods):
      plt.fill_between(numbers_of_samples, (errors_method_2[evaluation_method_index] + stdevs_method_2[evaluation_method_index])*unit_factor, (errors_method_2[evaluation_method_index] - stdevs_method_2[evaluation_method_index])*unit_factor, facecolor = evaluation_method_legend[evaluation_method][0], alpha = uncertainty_alpha)
      plt.plot(numbers_of_samples, errors_method_2[evaluation_method_index]*unit_factor, evaluation_method_legend[evaluation_method], label = evaluation_method_labels[evaluation_method])
    plt.ylim(bottom = 0)
    plt.xlabel("Number of samples used in the reconstruction")
    plt.ylabel(f"RMS error compared to expected signal ({units})")
    plt.legend()
    if archive:
      archive.write_plot("Sweeping the number of samples used in reconstruction\n(2-norm)", "number_of_samples_error_2")
    plt.draw()

  if "norms" in metrics:
    plt.figure()
    if ramsey_comparison_results is not None:
      plt.plot([numbers_of_samples[0]], [ramsey_comparison_results.error_0], "mo")
    for evaluation_method_index, evaluation_method in enumerate(evaluation_methods):
      plt.plot(numbers_of_samples, errors_method_0[evaluation_method_index]*100, evaluation_method_legend[evaluation_method])
    plt.ylim(bottom = 0)
    plt.xlabel("Number of samples used in the reconstruction")
    plt.ylabel("Proportion of points incorrectly classified (%)")
    plt.legend(legend)
    if archive:
      archive.write_plot("Sweeping the number of samples used in reconstruction\n(Determing the support)", "number_of_samples_error_0")
    plt.draw()

    plt.figure()
    if ramsey_comparison_results is not None:
      plt.plot([numbers_of_samples[0]], [ramsey_comparison_results.error_1], "mo")
    for evaluation_method_index, evaluation_method in enumerate(evaluation_methods):
      plt.plot(numbers_of_samples, errors_method_1[evaluation_method_index]*unit_factor, evaluation_method_legend[evaluation_method])
    plt.ylim(bottom = 0)
    plt.xlabel("Number of samples used in the reconstruction")
    plt.ylabel(f"Average error compared to expected signal ({units})")
    plt.legend(legend)
    if archive:
      archive.write_plot("Sweeping the number of samples used in reconstruction\n(1-norm)", "number_of_samples_error_1")
    plt.draw()

    plt.figure()
    plt.subplot()
    if ramsey_comparison_results is not None:
      plt.plot([numbers_of_samples[0]], [ramsey_comparison_results.error_sup], "mo")
    for evaluation_method_index, evaluation_method in enumerate(evaluation_methods):
      plt.plot(numbers_of_samples, errors_method_sup[evaluation_method_index]*unit_factor, evaluation_method_legend[evaluation_method])
    plt.ylim(bottom = 0)
    plt.xlabel("Number of samples used in the reconstruction")
    plt.ylabel(f"Largest error compared to expected signal ({units})")
    plt.legend(legend)
    if archive:
      archive.write_plot("Sweeping the number of samples used in reconstruction\n(sup-norm)", "number_of_samples_error_sup")
    plt.draw()

    plt.figure()
    plt.subplot()
    if ramsey_comparison_results is not None:
      plt.plot([numbers_of_samples[0]], [ramsey_comparison_results.error_snr], "mo")
    for evaluation_method_index, evaluation_method in enumerate(evaluation_methods):
      plt.plot(numbers_of_samples, errors_method_snr[evaluation_method_index], evaluation_method_legend[evaluation_method])
    plt.xlabel("Number of samples used in the reconstruction")
    plt.ylabel(f"Signal to noise ratio (dB)")
    plt.legend(legend)
    if archive:
      archive.write_plot("Sweeping the number of samples used in reconstruction\n(Signal to noise ratio)", "number_of_samples_error_snr")
    plt.draw()

  if "mf" in metrics:
    plt.figure()
    plt.subplot(2, 1, 2)
    if ramsey_comparison_results is not None:
      if hasattr(ramsey_comparison_results, "error_mf"):
        plt.plot([numbers_of_samples[0]], [ramsey_comparison_results.error_mf], evaluation_method_legend["ramsey"], label = evaluation_method_labels["ramsey"])
    for evaluation_method_index, evaluation_method in enumerate(evaluation_methods):
      plt.plot(numbers_of_samples, errors_method_mf[evaluation_method_index]*unit_factor, evaluation_method_legend[evaluation_method], label = evaluation_method_labels[evaluation_method])
    plt.xlabel("Number of samples used in the reconstruction")
    plt.ylabel(f"Detected signal amplitude ({units})")
    plt.ylim(bottom = 0)
    plt.legend()
    if archive:
      archive.write_plot("Sweeping the number of samples used in reconstruction\n(Matched filter detection)", "number_of_samples_error_mf_amplitude")
    plt.draw()

    # plt.figure()
    # plt.subplot(2, 1, 2)
    # if ramsey_comparison_results is not None:
    #   plt.plot([numbers_of_samples[0]], [ramsey_comparison_results.error_mfd], "mo")
    # for evaluation_method_index, evaluation_method in enumerate(evaluation_methods):
    #   plt.plot(numbers_of_samples, errors_method_mfd[evaluation_method_index]*100, evaluation_method_legend[evaluation_method])
    # plt.xlabel("Number of samples used in the reconstruction")
    # plt.ylabel(f"Proportion of reconstructions\n with correct detections (%)")
    # plt.ylim(bottom = 0)
    # plt.legend(legend)
    # plt.subplot(2, 1, 1)
    # plt.plot([numbers_of_samples[0], numbers_of_samples[-1]], [mf_cutoff]*2, "y--")
    # if ramsey_comparison_results is not None:
    #   plt.plot([numbers_of_samples[0]], [ramsey_comparison_results.error_mf], "mo")
    # for evaluation_method_index, evaluation_method in enumerate(evaluation_methods):
    #   plt.plot(numbers_of_samples, errors_method_mf[evaluation_method_index]*(unit_factor**2), evaluation_method_legend[evaluation_method])
    # plt.ylabel(f"Average detection\nenergy ({units}$^2$)")
    # plt.ylim(bottom = 0)
    # plt.legend(["Threshold"] + legend)
    # if archive:
    #   archive.write_plot("Sweeping the number of samples used in reconstruction\n(Matched filter detection)", "number_of_samples_error_mf")
    # plt.draw()

    # plt.figure()
    # plt.subplot(2, 1, 2)
    # if ramsey_comparison_results is not None:
    #   plt.plot([numbers_of_samples[0]], [ramsey_comparison_results.error_mfpd*100], "mo")
    # for evaluation_method_index, evaluation_method in enumerate(evaluation_methods):
    #   plt.plot(numbers_of_samples, errors_method_mfpd[evaluation_method_index]*100, evaluation_method_legend[evaluation_method])
    # plt.xlabel("Number of samples used in the reconstruction")
    # plt.ylabel(f"Proportion of reconstructions\nwith false positives (%)")
    # plt.ylim(bottom = 0)
    # plt.legend(legend)
    # plt.subplot(2, 1, 1)
    # plt.plot([numbers_of_samples[0], numbers_of_samples[-1]], [mfp_cutoff]*2, "y--")
    # if ramsey_comparison_results is not None:
    #   plt.plot([numbers_of_samples[0]], [ramsey_comparison_results.error_mfp], "mo")
    # for evaluation_method_index, evaluation_method in enumerate(evaluation_methods):
    #   plt.plot(numbers_of_samples, errors_method_mfp[evaluation_method_index]*(unit_factor**2), evaluation_method_legend[evaluation_method])
    # plt.ylabel(f"Average false positive\nenergy ({units}$^2$)")
    # plt.ylim(bottom = 0)
    # plt.legend(["Threshold"] + legend)
    # if archive:
    #   archive.write_plot("Sweeping the number of samples used in reconstruction\n(Matched filter false positives)", "number_of_samples_error_mfp")
    # plt.draw()

  if "confusion_fixed" in metrics:
    plt.figure()
    plt.subplot()
    if ramsey_comparison_results is not None:
      plt.plot([numbers_of_samples[0]], [ramsey_comparison_results.error_sensitivity*100], evaluation_method_legend["ramsey"], label = evaluation_method_labels["ramsey"])
    for evaluation_method_index, evaluation_method in enumerate(evaluation_methods):
      plt.fill_between(numbers_of_samples, (errors_method_sensitivity[evaluation_method_index] + stdevs_method_sensitivity[evaluation_method_index])*100, (errors_method_sensitivity[evaluation_method_index] - stdevs_method_sensitivity[evaluation_method_index])*100, facecolor = evaluation_method_legend[evaluation_method][0], alpha = uncertainty_alpha)
      plt.plot(numbers_of_samples, errors_method_sensitivity[evaluation_method_index]*100, evaluation_method_legend[evaluation_method], label = evaluation_method_labels[evaluation_method])
    plt.xlabel("Number of samples used in the reconstruction")
    plt.ylabel(f"Recall (%)")
    plt.legend()
    plt.ylim(bottom = -5, top = 105)
    if archive:
      archive.write_plot("Sweeping the number of samples used in reconstruction\n(Recall)", "number_of_samples_error_sensitivity")
    plt.draw()

    plt.figure()
    plt.subplot()
    if ramsey_comparison_results is not None:
      plt.plot([numbers_of_samples[0]], [ramsey_comparison_results.error_specificity*100], evaluation_method_legend["ramsey"], label = evaluation_method_labels["ramsey"])
    for evaluation_method_index, evaluation_method in enumerate(evaluation_methods):
      plt.fill_between(numbers_of_samples, (errors_method_specificity[evaluation_method_index] + stdevs_method_specificity[evaluation_method_index])*100, (errors_method_specificity[evaluation_method_index] - stdevs_method_specificity[evaluation_method_index])*100, facecolor = evaluation_method_legend[evaluation_method][0], alpha = uncertainty_alpha)
      plt.plot(numbers_of_samples, errors_method_specificity[evaluation_method_index]*100, evaluation_method_legend[evaluation_method], label = evaluation_method_labels[evaluation_method])
    plt.xlabel("Number of samples used in the reconstruction")
    plt.ylabel(f"Specificity (%)")
    plt.legend()
    plt.ylim(bottom = -5, top = 105)
    if archive:
      archive.write_plot("Sweeping the number of samples used in reconstruction\n(Specificity)", "number_of_samples_error_specificity")
    plt.draw()

    plt.figure()
    plt.subplot()
    if ramsey_comparison_results is not None:
      plt.plot([(ramsey_comparison_results.error_specificity - 1)*100], [ramsey_comparison_results.error_sensitivity*100], evaluation_method_legend["ramsey"], label = evaluation_method_labels["ramsey"])
    for evaluation_method_index, evaluation_method in enumerate(evaluation_methods):
      plt.plot((1 - errors_method_specificity[evaluation_method_index])*100, errors_method_sensitivity[evaluation_method_index]*100, evaluation_method_legend[evaluation_method], label = evaluation_method_labels[evaluation_method])
    plt.ylabel("Recall (%)")
    plt.xlabel("Fall out (%)")
    plt.legend()
    plt.ylim(bottom = -5, top = 105)
    plt.xlim(left = -5, right = 105)
    if archive:
      archive.write_plot("ROC parametrised by\nnumber of samples", "number_of_samples_error_roc_sweep")
    plt.draw()

    time_mesh -= 0.5/expected_frequency
    for evaluation_method_index, evaluation_method in enumerate(evaluation_methods):
      for reconstruction_index, number_of_samples in enumerate(sweep_samples):
          amplitude_mesh[reconstruction_index, :] = method_matched_decisions[evaluation_method_index][reconstruction_index, 0:99]
      
      plt.figure()
      plt.subplot(2, 1, 2)
      neural_pulse = expected_signal.neural_pulses[0]
      plt.pcolormesh(time_mesh*1e3, samples_mesh, amplitude_mesh, cmap = "Reds", vmin = 0, vmax = 1)
      plt.xlabel("Time (ms)")
      plt.ylabel("Number of samples")
      plt.xlim([(neural_pulse.time_start - 0.5/neural_pulse.frequency)*1e3, (neural_pulse.time_start + 1.5/neural_pulse.frequency)*1e3])
      colour_bar = plt.colorbar()
      colour_bar.set_label(f"Negative - Positive")

      plt.subplot(2, 1, 1)
      plt.pcolormesh(time_mesh*1e3, samples_mesh, amplitude_mesh, cmap = "Reds", vmin = 0, vmax = 1)
      plt.ylabel("Number of samples")

      if archive:
        archive.write_plot(f"Sweeping the number of samples used in reconstruction\nPulse detection, {evaluation_method_labels[evaluation_method]}", f"number_of_samples_error_confusion_fixed_{evaluation_method}")
      plt.draw()
    time_mesh += 0.5/expected_frequency

  if "roc" in metrics:
    for evaluation_method_index, evaluation_method in enumerate(evaluation_methods):
      plt.figure()
      # plt.plot((1 - errors_method_roc_specificity[evaluation_method_index][0])*100, errors_method_roc_sensitivity[evaluation_method_index][0]*100, evaluation_method_legend[evaluation_method])
      for samples_index in range(numbers_of_samples.size):
        plt.fill((1 - errors_method_roc_specificity[evaluation_method_index][samples_index])*100, errors_method_roc_sensitivity[evaluation_method_index][samples_index]*100, color = "k", alpha = 1/numbers_of_samples.size)
      plt.ylabel("Recall (%)")
      plt.xlabel("Fall out (%)")
      plt.ylim(bottom = -5, top = 105)
      plt.xlim(left = -5, right = 105)
      if archive:
        archive.write_plot(f"ROC, {evaluation_method_labels[evaluation_method]}", f"number_of_samples_error_roc_{evaluation_method}")
      plt.draw()

    plt.figure()
    if ramsey_comparison_results is not None:
      if hasattr(ramsey_comparison_results, "error_roc_auc"):
        plt.plot([numbers_of_samples[0]], [ramsey_comparison_results.error_roc_auc], evaluation_method_legend["ramsey"], label = evaluation_method_labels["ramsey"])
    for evaluation_method_index, evaluation_method in enumerate(evaluation_methods):
      plt.fill_between(numbers_of_samples, (errors_method_roc_auc[evaluation_method_index] + stdevs_method_roc_auc[evaluation_method_index])*100, (errors_method_roc_auc[evaluation_method_index] - stdevs_method_roc_auc[evaluation_method_index])*100, facecolor = evaluation_method_legend[evaluation_method][0], alpha = uncertainty_alpha)
      plt.plot(numbers_of_samples, errors_method_roc_auc[evaluation_method_index]*100, evaluation_method_legend[evaluation_method], label = evaluation_method_labels[evaluation_method])
    plt.ylim(bottom = -5, top = 105)
    plt.xlabel("Number of samples used in the reconstruction")
    plt.ylabel(f"Area under receiver operating characteristic (%)")
    plt.legend()
    if archive:
      archive.write_plot("Sweeping the number of samples used in reconstruction\n(ROC AUC)", "number_of_samples_error_roc_auc")
    plt.draw()

    plt.figure()
    plt.plot([numbers_of_samples[0], numbers_of_samples[-1]], [expected_amplitude]*2, "b--", label = "Ground truth amplitude")
    if ramsey_comparison_results is not None:
      if hasattr(ramsey_comparison_results, "error_roc_amplitude"):
        plt.plot([numbers_of_samples[0]], [ramsey_comparison_results.error_roc_amplitude], evaluation_method_legend["ramsey"], label = evaluation_method_labels["ramsey"])
    for evaluation_method_index, evaluation_method in enumerate(evaluation_methods):
      plt.fill_between(numbers_of_samples, (errors_method_roc_amplitude[evaluation_method_index] + stdevs_method_roc_amplitude[evaluation_method_index])*unit_factor, (errors_method_roc_amplitude[evaluation_method_index] - stdevs_method_roc_amplitude[evaluation_method_index])*unit_factor, facecolor = evaluation_method_legend[evaluation_method][0], alpha = uncertainty_alpha)
      plt.plot(numbers_of_samples, errors_method_roc_amplitude[evaluation_method_index]*unit_factor, evaluation_method_legend[evaluation_method], label = evaluation_method_labels[evaluation_method])
    plt.ylim(bottom = 0)
    plt.xlabel("Number of samples used in the reconstruction")
    plt.ylabel(f"Estimate of signal amplitude ({units})")
    plt.legend()
    if archive:
      archive.write_plot("Sweeping the number of samples used in reconstruction\n(Amplitude estimation)", "number_of_samples_error_roc_amplitude")
    plt.draw()

  # plt.figure()
  # plt.subplot()
  # coherence[random_index]
  # for random_index, random_seed in enumerate(random_seeds):
  #   plt.plot(numbers_of_samples, coherence[random_index], "--")
  # plt.plot(numbers_of_samples, coherence_coefficient*arcsech_densities[0], "k-", label = "asech fit")
  # plt.ylim(bottom = 0, top = 1)
  # plt.xlabel("Number of samples used in the reconstruction")
  # plt.ylabel("Sensing coherence")
  # plt.legend()
  # if archive:
  #   archive.write_plot(f"Sensing coherence for matrices used in reconstructions\nFit coefficient of {coherence_coefficient}", "number_of_samples_coherence")
  # plt.draw()

def run_reconstruction_norm_scale_factor_sweep(expected_signal:TestSignal, experiment_results:ExperimentResults, sweep_parameters = (0.5, 5, 0.5), archive:Archive = None, number_of_samples = 10000, frequency_cutoff_low = 0, frequency_cutoff_high = 100000, random_seeds = [util.Seeds.metroid], evaluation_methods = [], expected_amplitude = None, expected_frequency = None, expected_error_measurement = None, rabi_frequency_readout = None, frequency_line_noise = None, frequency_fit_step_size = 1, units = "Hz"):
  reconstruction = Reconstruction(expected_signal.time_properties)

  random_seeds = np.array(random_seeds)
  # scale_factor_modifiers = np.arange(sweep_parameters[0], sweep_parameters[1], sweep_parameters[2])
  scale_factor_modifiers = np.linspace(sweep_parameters[0], sweep_parameters[1], sweep_parameters[2])

  C.starting("Norm scale factor sweep")

  amplitudes = []
  norm_scale_factors = []
  for evaluation_method_index, evaluation_method in enumerate(evaluation_methods):
    for reconstruction_index, norm_scale_factor_modifier in enumerate(scale_factor_modifiers):
      for random_index, random_seed in enumerate(random_seeds):
        reconstruction.read_frequencies_from_experiment_results(experiment_results, number_of_samples, frequency_cutoff_low = frequency_cutoff_low, frequency_cutoff_high = frequency_cutoff_high, random_seed = random_seed)
        if evaluation_method == "fista_ayanzadeh":
          reconstruction.evaluate_fista_ayanzadeh(expected_amplitude = expected_amplitude, expected_frequency = expected_frequency, expected_error_measurement = expected_error_measurement)
        elif evaluation_method == "fista_fit":
          reconstruction.evaluate_fista_fit(expected_amplitude = expected_amplitude, expected_frequency = expected_frequency, expected_error_measurement = expected_error_measurement, rabi_frequency_readout = rabi_frequency_readout, frequency_line_noise = frequency_line_noise, norm_scale_factor_modifier = norm_scale_factor_modifier, is_fast = True)
        elif evaluation_method == "ista_fit":
          reconstruction.evaluate_fista_fit(expected_amplitude = expected_amplitude, expected_frequency = expected_frequency, expected_error_measurement = expected_error_measurement, rabi_frequency_readout = rabi_frequency_readout, frequency_line_noise = frequency_line_noise, norm_scale_factor_modifier = norm_scale_factor_modifier, is_fast = False)
        elif evaluation_method == "fista_adaptive":
          reconstruction.evaluate_fista_adaptive(expected_amplitude = expected_amplitude, expected_frequency = expected_frequency, expected_error_measurement = expected_error_measurement, norm_scale_factor_modifier = norm_scale_factor_modifier, is_fast = True)
        elif evaluation_method == "ista_adaptive":
          reconstruction.evaluate_fista_adaptive(expected_amplitude = expected_amplitude, expected_frequency = expected_frequency, expected_error_measurement = expected_error_measurement, norm_scale_factor_modifier = norm_scale_factor_modifier, is_fast = False)
        elif evaluation_method == "fista_informed_least_squares":
          reconstruction.evaluate_informed_least_squares(informed_type = "fista", expected_amplitude = expected_amplitude, expected_frequency = expected_frequency, expected_error_measurement = expected_error_measurement, norm_scale_factor_modifier = norm_scale_factor_modifier)
        elif evaluation_method == "ista_informed_least_squares":
          reconstruction.evaluate_informed_least_squares(informed_type = "ista", expected_amplitude = expected_amplitude, expected_frequency = expected_frequency, expected_error_measurement = expected_error_measurement, norm_scale_factor_modifier = norm_scale_factor_modifier)
        elif evaluation_method == "fadaptive_informed_least_squares":
          reconstruction.evaluate_informed_least_squares(informed_type = "fista_adaptive", expected_amplitude = expected_amplitude, expected_frequency = expected_frequency, expected_error_measurement = expected_error_measurement, norm_scale_factor_modifier = norm_scale_factor_modifier)
        elif evaluation_method == "adaptive_informed_least_squares":
          reconstruction.evaluate_informed_least_squares(informed_type = "ista_adaptive", expected_amplitude = expected_amplitude, expected_frequency = expected_frequency, expected_error_measurement = expected_error_measurement, norm_scale_factor_modifier = norm_scale_factor_modifier)
        elif evaluation_method == "fadaptive_frequency_fit":
          reconstruction.evaluate_informed_least_squares(informed_type = "fista_frequency_fit", expected_amplitude = expected_amplitude, expected_frequency = expected_frequency, expected_error_measurement = expected_error_measurement, norm_scale_factor_modifier = norm_scale_factor_modifier, frequency_fit_step_size = frequency_fit_step_size)
        elif evaluation_method == "adaptive_frequency_fit":
          reconstruction.evaluate_informed_least_squares(informed_type = "ista_frequency_fit", expected_amplitude = expected_amplitude, expected_frequency = expected_frequency, expected_error_measurement = expected_error_measurement, norm_scale_factor_modifier = norm_scale_factor_modifier, frequency_fit_step_size = frequency_fit_step_size)
        elif evaluation_method == "fista_backtracking":
          reconstruction.evaluate_fista_backtracking(expected_amplitude = expected_amplitude, expected_frequency = expected_frequency, expected_error_measurement = expected_error_measurement, norm_scale_factor_modifier = norm_scale_factor_modifier, is_fast = True)
        elif evaluation_method == "ista_backtracking":
          reconstruction.evaluate_fista_backtracking(expected_amplitude = expected_amplitude, expected_frequency = expected_frequency, expected_error_measurement = expected_error_measurement, norm_scale_factor_modifier = norm_scale_factor_modifier, is_fast = False)
        reconstruction.write_to_file(archive.archive_file, (scale_factor_modifiers.size*evaluation_method_index + reconstruction_index)*random_seeds.size + random_index)

        amplitudes.append(reconstruction.amplitude.copy())
      if evaluation_method_index == 0:
        norm_scale_factors.append(reconstruction.norm_scale_factor)
  norm_scale_factors = np.array(norm_scale_factors)

  C.finished("Norm scale factor sweep")

  evaluation_method_labels = {
    "least_squares" : "Least squares",
    
    "fista_ayanzadeh" : "FISTA (Ayanzadeh)",
    "fista_adaptive" : "FISTA (Adaptive)",
    "fista_backtracking" : "FISTA (Backtracking)",
    "fista_fit" : "FISTA (Line noise fit)",
    "fista" : "FISTA",
    "fista_informed_least_squares" : "FISTA informed least squares",
    "fadaptive_informed_least_squares" : "Sandwich (fast)",
    "fadaptive_frequency_fit" : "Sandwich frequency fit (fast)",

    "ista_ayanzadeh" : "ISTA (Ayanzadeh)",
    "ista_adaptive" : "ISTA (Adaptive)",
    "ista_backtracking" : "ISTA (Backtracking)",
    "ista_fit" : "ISTA (Line noise fit)",
    "ista" : "ISTA",
    "ista_informed_least_squares" : "ISTA informed least squares",
    "adaptive_informed_least_squares" : "Sandwich",
    "adaptive_frequency_fit" : "Sandwich frequency fit"
  }

  if "Hz" in units:
    unit_factor = 1
  elif "T" in units:
    unit_factor = 1/7e9
  if "n" in units:
    unit_factor *= 1e9
  elif "μ" in units:
    unit_factor *= 1e6
  elif "m" in units:
    unit_factor *= 1e3

  expected_amplitude *= unit_factor

  time_mesh, scale_factors_mesh = np.meshgrid(reconstruction.time_properties.time_coarse, (norm_scale_factors*unit_factor))
  amplitude_mesh = np.empty((norm_scale_factors.size, amplitudes[0].size))
  residual_mesh = np.empty_like(amplitude_mesh)
  for evaluation_method_index, evaluation_method in enumerate(evaluation_methods):
    for reconstruction_index, number_of_samples in enumerate(norm_scale_factors):
        amplitude_mesh[reconstruction_index, :] = amplitudes[(norm_scale_factors.size*evaluation_method_index + reconstruction_index)*random_seeds.size]*unit_factor
        residual_mesh[reconstruction_index, :] = (amplitudes[(norm_scale_factors.size*evaluation_method_index + reconstruction_index)*random_seeds.size] - expected_signal.amplitude)*unit_factor
    
    plt.figure()
    plt.subplot(2, 1, 2)
    neural_pulse = expected_signal.neural_pulses[0]
    plt.pcolormesh(time_mesh*1e3, scale_factors_mesh, amplitude_mesh, cmap = "seismic", vmin = -2*expected_amplitude, vmax = 2*expected_amplitude)
    plt.xlabel("Time (ms)")
    plt.ylabel("Regularisation\nparameter (Hz)")
    plt.xlim([(neural_pulse.time_start - 0.5/neural_pulse.frequency)*1e3, (neural_pulse.time_start + 1.5/neural_pulse.frequency)*1e3])
    colour_bar = plt.colorbar()
    colour_bar.set_label(f"Amplitude ({units})")

    plt.subplot(2, 1, 1)
    plt.pcolormesh(time_mesh*1e3, scale_factors_mesh, amplitude_mesh, cmap = "seismic", vmin = -2*expected_amplitude, vmax = 2*expected_amplitude)
    plt.ylabel(f"Regularisation\nparameter ({units})")

    if archive:
      archive.write_plot(f"Sweeping the regularisation parameter used in reconstruction\n{evaluation_method_labels[evaluation_method]}", f"norm_scale_factor_{evaluation_method}")
    plt.draw()

    plt.figure()
    plt.subplot(2, 1, 2)
    neural_pulse = expected_signal.neural_pulses[0]
    plt.pcolormesh(time_mesh*1e3, scale_factors_mesh, residual_mesh, cmap = "seismic", vmin = -2*expected_amplitude, vmax = 2*expected_amplitude)
    plt.xlabel("Time (ms)")
    plt.ylabel(f"Regularisation\nparameter ({units})")
    plt.xlim([(neural_pulse.time_start - 0.5/neural_pulse.frequency)*1e3, (neural_pulse.time_start + 1.5/neural_pulse.frequency)*1e3])
    colour_bar = plt.colorbar()
    colour_bar.set_label(f"Residual ({units})")

    plt.subplot(2, 1, 1)
    plt.pcolormesh(time_mesh*1e3, scale_factors_mesh, residual_mesh, cmap = "seismic", vmin = -2*expected_amplitude, vmax = 2*expected_amplitude)
    plt.ylabel("Regularisation\nparameter (Hz)")

    if archive:
      archive.write_plot(f"Sweeping the regularisation parameter used in reconstruction\n{evaluation_method_labels[evaluation_method]}, Residual", f"norm_scale_factor_{evaluation_method}_residual")
    plt.draw()

def plot_reconstruction_number_of_samples_sweep_signal_comparison(archive, archive_times, reconstruction_method = "fista_backtracking", metrics = ["2", "roc_auc", "roc_sensitivity", "roc_specificity"], labels = ["Single pulse signal", "Double pulse signal"], units = "Hz"):
  if "Hz" in units:
    unit_factor = 1
  elif "T" in units:
    unit_factor = 1/7e9
  if "n" in units:
    unit_factor *= 1e9
  elif "μ" in units:
    unit_factor *= 1e6
  elif "m" in units:
    unit_factor *= 1e3

  first = True
  errors_signal = []
  stdevs_signal = []
  for archive_time in archive_times:
    archive_previous = Archive(archive.archive_path[:-25], "")
    archive_previous.open_archive_file(archive_time)
    if first:
      archive_group = archive_previous.archive_file.require_group("reconstruction_sweeps/number_of_samples")
      number_of_samples = np.asarray(archive_group["number_of_samples"])
      first = False
    archive_group = archive_previous.archive_file.require_group(f"reconstruction_sweeps/number_of_samples/{reconstruction_method}/")
    errors_metric = []
    stdevs_metric = []
    for metric in metrics:
      errors_metric.append(np.asarray(archive_group[f"error_{metric}"]))
      stdevs_metric.append(np.asarray(archive_group[f"stdev_{metric}"]))
    errors_signal.append(errors_metric)
    stdevs_signal.append(stdevs_metric)

  colours = [cm.lajolla(1/3), cm.lajolla(2/3)]
  ylabel_map = {
    "2" : f"RMSE ({units})",
    "roc_auc" : f"ROC AUC (%)"
  }
  unit_factor_map = {
    "2" : unit_factor,
    "roc_auc" : 100
  }
  metric_label_map = {
    "2" : "RMSE",
    "roc_auc" : "Area under receiver operating characteristic"
  }

  # plt.figure()
  # for metric_index, metric in enumerate(metrics):
  #   plt.subplot(len(metrics), 1, len(metrics) - metric_index)
  #   for signal_index, label in enumerate(labels):
  #     error = errors_signal[signal_index][metric_index]
  #     stdev = stdevs_signal[signal_index][metric_index]
  #     plt.fill_between(number_of_samples, (error + stdev)*unit_factor_map[metric], (error - stdev)*unit_factor_map[metric], color = colours[signal_index], alpha = 0.25)
  #     plt.plot(number_of_samples, error*unit_factor_map[metric], f"-", color = colours[signal_index], label = label)
  #   if metric_index == 0:
  #     plt.xlabel("Number of samples used in reconstruction", size = 16)
  #   else:
  #     plt.gca().axes.xaxis.set_ticklabels([])
  #   plt.ylabel(ylabel_map[metric], size = 16)
  #   plt.ylim(bottom = 0, top = 1.2*np.max(error + stdev)*unit_factor_map[metric])
  #   plt.xlim(left = 0, right = 100)
  #   plt.gca().spines["right"].set_visible(False)
  #   plt.gca().spines["top"].set_visible(False)
  #   plt.text(5, 1.1*np.max(error + stdev)*unit_factor_map[metric], f"({chr(98 - metric_index)})", size = 16)
  # if archive:
  #   archive.write_plot(f"", f"number_of_samples_comparison")
  # plt.draw()

  # subsample_index = 69 # => 30 samples
  subsample_index = 79 # => 20 samples
  metric_index = 1
  metric = "roc_auc"
  fig = plt.figure(figsize = [6.4, 4.8*3/4])
  for signal_index, label in enumerate(labels):
    error = errors_signal[signal_index][metric_index]
    stdev = stdevs_signal[signal_index][metric_index]
    plt.fill_between(number_of_samples, ((error + stdev)*(error + stdev < 1) + 1*(error + stdev >= 1))*unit_factor_map[metric], ((error - stdev)*(error + stdev < 1) + (1- 2*stdev)*(error + stdev >= 1))*unit_factor_map[metric], color = colours[signal_index], alpha = 0.2)
  for signal_index, label in enumerate(labels):
    error = errors_signal[signal_index][metric_index]
    stdev = stdevs_signal[signal_index][metric_index]
    # plt.plot(number_of_samples, ((error + stdev)*(error + stdev < 1) + 1*(error + stdev >= 1))*unit_factor_map[metric], f"--", color = colours[signal_index])
    # plt.plot(number_of_samples, ((error - stdev)*(error + stdev < 1) + (1- 2*stdev)*(error + stdev >= 1))*unit_factor_map[metric], f"--", color = colours[signal_index])
  for signal_index, label in enumerate(labels):
    error = errors_signal[signal_index][metric_index]
    stdev = stdevs_signal[signal_index][metric_index]
    plt.plot(number_of_samples, error*unit_factor_map[metric], f"-", color = colours[signal_index], label = label)
    if signal_index == 1:
      plt.plot([number_of_samples[79]], [error[79]*unit_factor_map[metric]], f"s", color = colours[signal_index])
      plt.plot([number_of_samples[79], 45], [error[79]*unit_factor_map[metric], 45], f"--", color = colours[signal_index])
      plt.plot([number_of_samples[39]], [error[39]*unit_factor_map[metric]], f"s", color = colours[signal_index])
      plt.plot([number_of_samples[39], 70], [error[39]*unit_factor_map[metric], 60], f"--", color = colours[signal_index])
  plt.xlabel("Number of samples used in reconstruction", size = 14, fontname = "Times New Roman")
  plt.ylabel(ylabel_map[metric], size = 14, fontname = "Times New Roman")
  plt.ylim(bottom = 0, top = 103)#1.1*np.max(error)*unit_factor_map[metric])
  plt.xlim(left = 0, right = 100)
  plt.gca().spines["right"].set_visible(False)
  plt.gca().spines["top"].set_visible(False)
  plt.xticks(fontname = "Times New Roman")
  plt.yticks(fontname = "Times New Roman")
  fig.subplots_adjust(bottom=0.15)
  # plt.text(5, 1.1*np.max(error + stdev)*unit_factor_map[metric], f"({chr(98 - metric_index)})", size = 16)

  # error_sensitivity_1_pulse = errors_signal[0][2]
  # error_specificity_1_pulse = errors_signal[0][3]
  # error_sensitivity_2_pulse = errors_signal[1][2]
  # error_specificity_2_pulse = errors_signal[1][3]

  # stdev_sensitivity_1_pulse = stdevs_signal[0][2]
  # stdev_specificity_1_pulse = stdevs_signal[0][3]
  # stdev_sensitivity_2_pulse = stdevs_signal[1][2]
  # stdev_specificity_2_pulse = stdevs_signal[1][3]

  error_sensitivity_1_pulse = errors_signal[1][2]
  error_specificity_1_pulse = errors_signal[1][3]
  error_sensitivity_2_pulse = errors_signal[1][2]
  error_specificity_2_pulse = errors_signal[1][3]

  stdev_sensitivity_1_pulse = stdevs_signal[1][2]
  stdev_specificity_1_pulse = stdevs_signal[1][3]
  stdev_sensitivity_2_pulse = stdevs_signal[1][2]
  stdev_specificity_2_pulse = stdevs_signal[1][3]
  
  # plt.figure()
  ins = plt.gca().inset_axes([0.4, 0.2, 0.25, 0.5])

  # ins.plot([0, 100, 100], [0, 0, 100], "k--", alpha = 0.5)
  ins.text(40, 40, "AUC", size = 14, color = colours[1], fontname = "Times New Roman")
  ins.spines["right"].set_visible(False)
  ins.spines["top"].set_visible(False)

  # ins.set_xticks([0, 50, 100], fontname = "Times New Roman")
  # ins.set_yticks([0, 50, 100], fontname = "Times New Roman")

  # plt.subplot(1, 2, 1)
  specificity_boundary = []
  sensitivity_boundary = []

  subsample_index = 79 # => 30 samples

  for threshold_index in range(error_specificity_1_pulse.shape[1] - 1):
    boundary = error_specificity_1_pulse[subsample_index, threshold_index] + stdev_specificity_1_pulse[subsample_index, threshold_index]
    if boundary >= 1:
      boundary = 1
    if boundary - 2*stdev_specificity_1_pulse[subsample_index, threshold_index] <= 0:
      boundary = 2*stdev_specificity_1_pulse[subsample_index, threshold_index]
    specificity_boundary.append(boundary)

    boundary = error_sensitivity_1_pulse[subsample_index, threshold_index] + stdev_sensitivity_1_pulse[subsample_index, threshold_index]
    if boundary >= 1:
      boundary = 1
    if boundary - 2*stdev_sensitivity_1_pulse[subsample_index, threshold_index] <= 0:
      boundary = 2*stdev_sensitivity_1_pulse[subsample_index, threshold_index]
    sensitivity_boundary.append(boundary)
  for threshold_index in range(error_specificity_1_pulse.shape[1] - 2, -1, -1):
    boundary = error_specificity_1_pulse[subsample_index, threshold_index] - stdev_specificity_1_pulse[subsample_index, threshold_index]
    if boundary <= 0:
      boundary = 0
    if boundary + 2*stdev_specificity_1_pulse[subsample_index, threshold_index] >= 1:
      boundary = 1 - 2*stdev_specificity_1_pulse[subsample_index, threshold_index]
    specificity_boundary.append(boundary)

    boundary = error_sensitivity_1_pulse[subsample_index, threshold_index] - stdev_sensitivity_1_pulse[subsample_index, threshold_index]
    if boundary <= 0:
      boundary = 0
    if boundary + 2*stdev_sensitivity_1_pulse[subsample_index, threshold_index] >= 1:
      boundary = 1 - 2*stdev_sensitivity_1_pulse[subsample_index, threshold_index]
    sensitivity_boundary.append(boundary)
  specificity_boundary = np.array(specificity_boundary)
  sensitivity_boundary = np.array(sensitivity_boundary)
  # ins.fill(100*(1 - specificity_boundary), 100*sensitivity_boundary, color = colours[1], alpha = 0.2)
  ins.fill(100*(1 - error_specificity_1_pulse[subsample_index, :]), 100*error_sensitivity_1_pulse[subsample_index, :], color = "k", alpha = 0.1)
  ins.plot(100*(1 - error_specificity_1_pulse[subsample_index, :-1]), 100*error_sensitivity_1_pulse[subsample_index, :-1], "-", color = colours[1])
  ins.set_xlabel("                                             Fallout (%)", fontname = "Times New Roman")
  ins.set_ylabel("Recall (%)", fontname = "Times New Roman")
  ins.set_xlim([-10, 100])
  ins.set_ylim([0, 103])
  # ins.set_xticks([0, 50, 100], ["0", "50", "100"], fontproperties = "Times New Roman")
  # ins.set_yticks([0, 50, 100], ["0", "50", "100"], fontproperties = "Times New Roman")
  
  ins = plt.gca().inset_axes([0.7, 0.2, 0.25, 0.5])

  subsample_index = 39 # => 70 samples

  # ins.plot([0, 100, 100], [0, 0, 100], "k--", alpha = 0.5)
  ins.text(30, 40, "AUC", size = 14, color = colours[1], fontname = "Times New Roman")

  # ins.set_xticks([0, 50, 100], fontname = "Times New Roman")
  # ins.set_yticks([0, 50, 100], fontname = "Times New Roman")

  specificity_boundary = []
  sensitivity_boundary = []
  for threshold_index in range(error_specificity_2_pulse.shape[1] - 1):
    boundary = error_specificity_2_pulse[subsample_index, threshold_index] + stdev_specificity_2_pulse[subsample_index, threshold_index]
    if boundary >= 1:
      boundary = 1
    if boundary - 2*stdev_specificity_2_pulse[subsample_index, threshold_index] <= 0:
      boundary = 2*stdev_specificity_2_pulse[subsample_index, threshold_index]
    specificity_boundary.append(boundary)

    boundary = error_sensitivity_2_pulse[subsample_index, threshold_index] + stdev_sensitivity_2_pulse[subsample_index, threshold_index]
    if boundary >= 1:
      boundary = 1
    if boundary - 2*stdev_sensitivity_2_pulse[subsample_index, threshold_index] <= 0:
      boundary = 2*stdev_sensitivity_2_pulse[subsample_index, threshold_index]
    sensitivity_boundary.append(boundary)
  for threshold_index in range(error_specificity_2_pulse.shape[1] - 2, -1, -1):
    boundary = error_specificity_2_pulse[subsample_index, threshold_index] - stdev_specificity_2_pulse[subsample_index, threshold_index]
    if boundary <= 0:
      boundary = 0
    if boundary + 2*stdev_specificity_2_pulse[subsample_index, threshold_index] >= 1:
      boundary = 1 - 2*stdev_specificity_2_pulse[subsample_index, threshold_index]
    specificity_boundary.append(boundary)

    boundary = error_sensitivity_2_pulse[subsample_index, threshold_index] - stdev_sensitivity_2_pulse[subsample_index, threshold_index]
    if boundary <= 0:
      boundary = 0
    if boundary + 2*stdev_sensitivity_2_pulse[subsample_index, threshold_index] >= 1:
      boundary = 1 - 2*stdev_sensitivity_2_pulse[subsample_index, threshold_index]
    sensitivity_boundary.append(boundary)
  specificity_boundary = np.array(specificity_boundary)
  sensitivity_boundary = np.array(sensitivity_boundary)
  # ins.fill(100*(1 - specificity_boundary), 100*sensitivity_boundary, color = colours[1], alpha = 0.2)
  ins.fill(100*(1 - error_specificity_2_pulse[subsample_index, :]), 100*error_sensitivity_2_pulse[subsample_index, :], color = "k", alpha = 0.1)
  ins.plot(100*(1 - error_specificity_2_pulse[subsample_index, :-1]), 100*error_sensitivity_2_pulse[subsample_index, :-1], "-", color = colours[1])

  # ins.set_xlabel("Fallout (%)")
  # ins.set_ylabel("Recall (%)")
  ins.set_xlim([-10, 100])
  ins.set_ylim([0, 103])
  ins.yaxis.set_ticklabels([])
  ins.spines["right"].set_visible(False)
  ins.spines["top"].set_visible(False)
  plt.draw()

  if archive:
    archive.write_plot(f"", f"number_of_samples_comparison_auc")

def plot_reconstruction_method_comparison(archive, results_objects, ground_truth, units = "nT"):
  label_size = 14
  if "Hz" in units:
    unit_factor = 1
  elif "T" in units:
    unit_factor = 1/7e9
  if "n" in units:
    unit_factor *= 1e9
  elif "μ" in units:
    unit_factor *= 1e6
  elif "m" in units:
    unit_factor *= 1e3
  
  def colour_cycle(cmap, amount):
    def modified(y):
      x = cmap(y)
      if amount == 1:
        return (x[1], x[2], x[0], x[3])
        # return (min(x[1]/(0.2126/0.7152), 1), min(x[2]/(0.7152/0.0722), 1), min(x[0]/(0.0722/0.2126), 1), x[3])
      if amount == 2:
        return (x[2], x[0], x[1], x[3])
        # return (min(x[2]/(0.2126/0.0722), 1), min(x[0]/(0.7152/0.2126), 1), min(x[1]/(0.0722/0.7152), 1), x[3])
      return x
    return modified

  #  0.2126, + 0.7152, + 0.0722

  # colour_map = [cm.lajolla(2/3), cm.lajolla(2/3), cm.lajolla(2/3)]
  # colour_map = [cm.grayC, cm.bilbao, cm.lajolla]
  # colour_map = [cm.bilbao, colour_cycle(cm.bilbao, 1), colour_cycle(cm.bilbao, 2)]
  # colour_map = [cm.lajolla, cm.lajolla, cm.lajolla]
  colour_map = [cm.bilbao, colour_cycle(cm.bilbao, 1), colour_cycle(cm.bilbao, 2)]
  reorder_map = [0, 3, 1, 4, 2, 5]
  protocol_map = ["Ramsey", "Inverse DST", "Compressive retrieval"]

  # plt.figure(figsize = [(6.4 + 0.4)*2, 4.8])
  # plt.xlabel(f"Time (ms)", size = label_size, fontname = "Times New Roman")
  # plt.ylabel(f"Magnetic field ({units})", size = label_size, fontname = "Times New Roman")
  for result_index, result_object in enumerate(results_objects):
    plt.subplot(2, 3, reorder_map[result_index] + 1)

    plt.xticks(fontname = "Times New Roman")
    plt.yticks(fontname = "Times New Roman")

    if isinstance(result_object, Reconstruction):
      time = result_object.time_properties.time_coarse
      amplitude = result_object.amplitude
    else:
      time = result_object.time
      amplitude = result_object.amplitude
    if reorder_map[result_index] < 3:
      plt.gca().axes.xaxis.set_ticklabels([])
      plt.gca().axes.xaxis.set_label_position("top") 
      plt.xlabel(protocol_map[reorder_map[result_index]], size = label_size, fontname = "Times New Roman", color = colour_map[math.floor(result_index/2)](2/3))
    # else:
    if result_index == 3:
      plt.xlabel(f"Time (ms)", size = label_size, fontname = "Times New Roman")
    if math.fmod(reorder_map[result_index], 3) >= 1:
      plt.gca().axes.yaxis.set_ticklabels([])
    # else:
    if reorder_map[result_index] == 3:
      plt.ylabel(f"                                           Magnetic field ({units})", size = label_size, fontname = "Times New Roman")
    # if result_index == 0:
    plt.plot(time/1e-3, ground_truth[int(math.fmod(result_index, 2))][1:]*unit_factor, "--k",
      # color = cm.lajolla(1/3)
      # color = colour_map[math.floor(result_index/2)](1/3)
      linewidth = 1,
      alpha = 0.5
    )
    plt.plot(time/1e-3, amplitude*unit_factor, "-",
      # f"-{colour_map[math.floor(result_index/2)]}"
      # color = cm.lajolla(2/3)
      # color = colour_map[math.floor(result_index/2)](2/3)
      color = colour_map[math.floor(result_index/2)](2/3),
      # linewidth = 3
    )
    plt.xlim(left = 0, right = 5)
    plt.ylim(top = 800*unit_factor, bottom = -800*unit_factor)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)
    plt.text(0.25, 600*unit_factor, f"({chr(97 + result_index)})", size = label_size, fontname = "Times New Roman")
    plt.subplots_adjust(wspace = 0.05)
  if archive:
    archive.write_plot(f"", f"methods_comparison")
  plt.draw()

# def plot_reconstruction_method_comparison_abstract(archive, results_objects, ground_truth, units = "nT"):
#   if "Hz" in units:
#     unit_factor = 1
#   elif "T" in units:
#     unit_factor = 1/7e9
#   if "n" in units:
#     unit_factor *= 1e9
#   elif "μ" in units:
#     unit_factor *= 1e6
#   elif "m" in units:
#     unit_factor *= 1e3

#   colour_map = ["r", "y", "c"]
#   reorder_map = [0, 3, 1, 4, 2, 5]

#   plt.figure(figsize = [(6.4 + 0.4)*2, 4.8/2])
#   for result_index, result_object in enumerate(results_objects):
#     plt.subplot(1, 3, reorder_map[result_index] + 1)
#     if isinstance(result_object, Reconstruction):
#       time = result_object.time_properties.time_coarse
#       amplitude = result_object.amplitude
#     else:
#       time = result_object.time
#       amplitude = result_object.amplitude
#       plt.xlabel(f"Time (ms)", size = 16)
#     if math.fmod(reorder_map[result_index], 3) >= 1:
#       plt.gca().axes.yaxis.set_ticklabels([])
#     else:
#       plt.ylabel(f"Magnetic\nfield ({units})", size = 16)
#     # if result_index == 0:
      
#     plt.plot(time/1e-3, ground_truth[int(math.fmod(result_index, 2))][1:]*unit_factor, "-k")
#     plt.plot(time/1e-3, amplitude*unit_factor, f"-{colour_map[math.floor(result_index/2)]}")
#     plt.xlim(left = 0, right = 5)
#     plt.ylim(top = 800*unit_factor, bottom = -800*unit_factor)
#     plt.gca().spines["right"].set_visible(False)
#     plt.gca().spines["top"].set_visible(False)
#     plt.text(0.25, 600*unit_factor, f"({chr(97 + result_index)})", size = 16)
#     plt.subplots_adjust(wspace = 0.05)
#   if archive:
#     archive.write_plot(f"", f"methods_comparison")
#   plt.show()

def plot_reconstruction_unknown(archive, results_objects, units = "nT"):
  if "Hz" in units:
    unit_factor = 1
  elif "T" in units:
    unit_factor = 1/7e9
  if "n" in units:
    unit_factor *= 1e9
  elif "μ" in units:
    unit_factor *= 1e6
  elif "m" in units:
    unit_factor *= 1e3

  label_size = 14

  # colour_map = ["m", "b"]
  colour_map = [cm.lajolla(1/3), cm.lajolla(2/3)]
  plt.figure(figsize = [6.4, 4.8*3/2])
  # plt.figure()
  for result_index, result_object in enumerate(results_objects):
    plt.subplot(2, 1, result_index + 1)
    time = result_object.time_properties.time_coarse
    amplitude = result_object.amplitude
    if result_index == 0:
      plt.gca().axes.xaxis.set_ticklabels([])
      plt.ylabel(f"Magnetic field ({units})                                            ", size = label_size)
    else:
      plt.xlabel(f"Time (ms)", size = label_size)
    plt.plot(time/1e-3, amplitude*unit_factor, f"-", color = colour_map[result_index])
    plt.xlim(left = 0, right = 5)
    plt.ylim(top = 800*unit_factor, bottom = -800*unit_factor)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)
    plt.text(0.25, 600*unit_factor, f"({chr(97 + result_index)})", size = label_size)
    plt.subplots_adjust(wspace = 0.05, bottom = 0.2)
  if archive:
    archive.write_plot(f"", f"unknown_reconstructions")
  plt.show()





#   def evaluate_ista_complete(self):
#     """
#     Run compressive sensing based on the Iterative Shrinkage Thresholding Algorithm (ISTA)
#     """
#     print("\033[33mStarting reconstruction...\033[0m")

#     # Start timing reconstruction
#     execution_time_end_points = np.empty(2)
#     execution_time_end_points[0] = tm.time()
#     execution_time_end_points[1] = execution_time_end_points[0]

#     self.amplitude = np.empty_like(self.time_properties.time_coarse)              # Reconstructed signal
#     frequency_amplitude_prediction = np.empty_like(self.frequency_amplitude)           # Partial sine Fourier transform of reconstructed signal
#     fourier_transform = np.empty((self.frequency.size, self.time_properties.time_coarse.size)) # Storage for sine Fourier transform operator

#     # Setup GPU block and grid sizes
#     threads_per_block = 128
#     blocks_per_grid_time = (self.time_properties.time_coarse.size + (threads_per_block - 1)) // threads_per_block

#     reconstruct_ista_complete[blocks_per_grid_time, threads_per_block](self.time_properties.time_coarse, self.amplitude, self.frequency, self.frequency_amplitude, frequency_amplitude_prediction, fourier_transform, self.time_properties.time_step_coarse, 0, 100.0, 10.0)

#     print(str(tm.time() - execution_time_end_points[1]) + "s")
#     print("\033[32m_done!\033[0m")
#     execution_time_end_points[1] = tm.time()

#   def evaluate_fista(self):
#     """
#     Run compressive sensing based on the Fast Iterative Shrinkage Thresholding Algorithm (FISTA)
#     """
#     print("\033[33m_starting reconstruction...\033[0m")

#     # Start timing reconstruction
#     execution_time_end_points = np.empty(2)
#     execution_time_end_points[0] = tm.time()
#     execution_time_end_points[1] = execution_time_end_points[0]

#     self.amplitude = np.empty_like(self.time_properties.time_coarse)              # Reconstructed signal
#     frequency_amplitude_prediction = np.empty_like(self.frequency_amplitude)           # Partial sine Fourier transform of reconstructed signal
#     fourier_transform = np.empty((self.frequency.size, self.time_properties.time_coarse.size)) # Storage for sine Fourier transform operator
#     fourier_transform = cuda.to_device(fourier_transform)

#     # Setup GPU block and grid sizes
#     threads_per_block = 128
#     blocks_per_grid_time = (self.time_properties.time_coarse.size + (threads_per_block - 1)) // threads_per_block
#     blocks_per_grid_frequency = (self.frequency.size + (threads_per_block - 1)) // threads_per_block

#     # Initialise 
#     reconstruct_ista_initialisation_step[blocks_per_grid_time, threads_per_block](self.frequency, self.frequency_amplitude, self.time_properties.time_step_coarse, self.time_properties.time_coarse, self.amplitude, fourier_transform)

#     amplitude_previous = 0*self.amplitude  # The last amplitude, used in the fast step, and to check (Cauchy) convergence
#     fast_step_size = 1            # Initialise the fast step size to one
#     fastStep_size_previous = 1

#     while sum((self.amplitude - amplitude_previous)**2) > 1e0:   # Stop if signal has converged
#       amplitude_previous = 1*self.amplitude  # Keep track of previous amplitude

#       # Run ISTA steps
#       reconstruct_ista_prediction_step[blocks_per_grid_frequency, threads_per_block](self.frequency_amplitude, self.amplitude, fourier_transform, frequency_amplitude_prediction)
#       reconstruct_ista_manifold_step[blocks_per_grid_time, threads_per_block](frequency_amplitude_prediction, self.step_size_manifold, fourier_transform, self.amplitude)
#       reconstruct_ista_sparse_step[blocks_per_grid_time, threads_per_block](self.step_size_sparse, self.amplitude)

#       # Run the fast step
#       fastStep_size_previous = fast_step_size
#       fast_step_size = (1 + math.sqrt(1 + 4*fast_step_size**2))/2
#       self.amplitude = self.amplitude + ((fastStep_size_previous - 1)/fast_step_size)*(self.amplitude - amplitude_previous)

#     print(str(tm.time() - execution_time_end_points[1]) + "s")
#     print("\033[32mDone!\033[0m")
#     execution_time_end_points[1] = tm.time()
  
#   def evaluate_naive_ista(self):
#     """
#     Run compressive sensing based on the Iterative Shrinkage Thresholding Algorithm (ISTA).
#     The same as FISTA, but without the fast step.
#     """
#     self.amplitude = np.empty_like(self.time_properties.time_coarse)
#     frequency_amplitude_prediction = np.empty_like(self.frequency_amplitude)
#     fourier_transform = np.empty((self.frequency.size, self.time_properties.time_coarse.size))

#     threads_per_block = 128
#     blocks_per_grid_time = (self.time_properties.time_coarse.size + (threads_per_block - 1)) // threads_per_block
#     blocks_per_grid_frequency = (self.frequency.size + (threads_per_block - 1)) // threads_per_block
#     reconstructNaive_initialisation_step[blocks_per_grid_time, threads_per_block](self.frequency, self.frequency_amplitude, self.time_properties.time_step_coarse, self.time_properties.time_coarse, self.amplitude, fourier_transform)
#     # manifold_step_size = 100000
#     reconstructNaive_prediction_step[blocks_per_grid_frequency, threads_per_block](self.frequency_amplitude, self.amplitude, fourier_transform, frequency_amplitude_prediction)
#     square_loss = sum(frequency_amplitude_prediction**2)

#     # frequency = cuda.to_device(frequency)
#     # frequency_amplitude = cuda.to_device(frequency_amplitude)
#     # frequency_amplitude_prediction = cuda.to_device(frequency_amplitude_prediction)
#     # time_coarse = cuda.to_device(time_coarse)
#     # amplitude = cuda.to_device(amplitude)
#     # print(square_loss)

#     # plt.figure()
#     # plt.plot(time_coarse, amplitude)
#     # plt.draw()
#     amplitude_previous = 0*self.amplitude
#     while sum((self.amplitude - amplitude_previous)**2) > 1e0:
#       # if iteration_index == 0:
#       #   manifold_step_size = 2
#       # else:
#       #   manifold_step_size = 0.00005
#       # if square_loss < 1e-4:
#       amplitude_previous = 1*self.amplitude
#       reconstruct_ista_sparse_step[blocks_per_grid_time, threads_per_block](self.step_size_sparse, self.amplitude)
#       reconstruct_ista_prediction_step[blocks_per_grid_frequency, threads_per_block](self.frequency_amplitude, self.amplitude, fourier_transform, frequency_amplitude_prediction)
#       # square_loss_previous = square_loss
#       # square_loss = sum(frequency_amplitude_prediction**2)
#       # if square_loss > square_loss_previous:
#       #   manifold_step_size *= 2
#       # else:
#       #   manifold_step_size /= 2
#       reconstruct_ista_manifold_step[blocks_per_grid_time, threads_per_block](frequency_amplitude_prediction, self.step_size_manifold, fourier_transform, self.amplitude)
#       # if iteration_index % 1 == 0:
#       #   # print(square_loss)
#       #   # print(frequency_amplitude_prediction)

#       #   plt.figure()
#       #   plt.plot(time_coarse, amplitude)
#       #   plt.draw()

#     # time_coarse = time_coarse.copy_to_host()
#     # amplitude = amplitude.copy_to_host()
#     # plt.figure()
#     # plt.plot(self.time_properties.time_coarse, self.amplitude)
#     # plt.draw()

# @cuda.jit()
# def reconstruct_ista_complete(
#   time_coarse, amplitude,                      # Time
#   frequency, frequency_amplitude, frequency_amplitude_prediction,  # Frequency
#   fourier_transform, time_step_coarse,                 # Parameters
#   sparse_penalty, min_accuracy, expected_amplitude           # Parameters
# ):
#   # Initialise
#   time_index = cuda.threadIdx.x + cuda.blockIdx.x*cuda.blockDim.x
#   if time_index < time_coarse.size:
#     amplitude[time_index] = 0.0
#     for frequency_index in range(frequency.size):
#       # Find the Fourier transform coefficient
#       fourier_transform[frequency_index, time_index] = math.sin(2*math.pi*frequency[frequency_index]*time_coarse[time_index])*time_step_coarse/(time_coarse[time_coarse.size - 1] - time_coarse[0])
#       # # Apply the Moore–Penrose inverse of the Fourier transform, based off its SVD
#       # amplitude[time_index] += fourier_transform[frequency_index, time_index]*(2.0*(time_coarse[time_coarse.size - 1] - time_coarse[0])/(time_step_coarse))*frequency_amplitude[frequency_index]

#   step_size = (time_coarse[time_coarse.size - 1] - time_coarse[0])/time_step_coarse
#   max_iteration_index = math.ceil((((expected_amplitude/(1e3*time_step_coarse))**2)/min_accuracy)/step_size)
#   for iteration_index in range(max_iteration_index):
#     # Prediction
#     cuda.syncthreads()
#     frequency_index = cuda.threadIdx.x + cuda.blockIdx.x*cuda.blockDim.x
#     if frequency_index < frequency_amplitude_prediction.size:
#       frequency_amplitude_prediction[frequency_index] = -frequency_amplitude[frequency_index]
#       for time_index in range(time_coarse.size):
#         frequency_amplitude_prediction[frequency_index] += fourier_transform[frequency_index, time_index]*0.0#amplitude[time_index]

#     if cuda.threadIdx.x + cuda.blockIdx.x*cuda.blockDim.x == 0:
#       print(frequency_amplitude_prediction[frequency_index], frequency_amplitude[frequency_index], amplitude[time_coarse.size - 1])

#     # Linear inverse
#     cuda.syncthreads()
#     time_index = cuda.threadIdx.x + cuda.blockIdx.x*cuda.blockDim.x
#     if time_index < time_coarse.size:
#       for frequency_index in range(frequency_amplitude_prediction.size):
#         amplitude[time_index] -= 2*fourier_transform[frequency_index, time_index]*(frequency_amplitude_prediction[frequency_index])*step_size

#     # Shrinkage
#     amplitude_temporary = math.fabs(amplitude[time_index]) - step_size*sparse_penalty
#     if amplitude_temporary > 0:
#       amplitude[time_index] = math.copysign(amplitude_temporary, amplitude[time_index])  # Apparently normal "sign" doesn't exist, but this weird thing does :P
#     else:
#       amplitude[time_index] = 0


# @cuda.jit()
# def reconstruct_ista_initialisation_step(frequency, frequency_amplitude, time_step_coarse, time_coarse, amplitude, fourier_transform):
#   """
#   Generate the Fourier transform matrix, and use the Moore–Penrose inverse to initialise the
#   reconstruction to an allowed (but not optimal) solution
#   """
#   time_index = cuda.threadIdx.x + cuda.blockIdx.x*cuda.blockDim.x
#   if time_index < time_coarse.size:
#     amplitude[time_index] = 0
#     for frequency_index in range(frequency.size):
#       # Find the Fourier transform coefficient
#       fourier_transform[frequency_index, time_index] = math.sin(2*math.pi*frequency[frequency_index]*time_coarse[time_index])*time_step_coarse/time_coarse[time_coarse.size - 1]
#       # Apply the Moore–Penrose inverse of the Fourier transform, based off its SVD
#       amplitude[time_index] += 2*fourier_transform[frequency_index, time_index]*(time_coarse[time_coarse.size - 1]/(time_step_coarse))*frequency_amplitude[frequency_index]

# @cuda.jit()
# def reconstruct_ista_sparse_step(step_size, amplitude):
#   """
#   Use gradient decent to minimise the one norm of the reconstruction (ie make it sparse)

#   min Z(r),
#   Z(r) = norm1(r) (in time),
#   dZ(r)/dr(t) = sign(r(t))

#   This algorithm is equivalent to
  
#   r = sign(r)*ReLU(abs(r) - step_size)

#   from the paper on FISTA.
#   """
#   time_index = cuda.threadIdx.x + cuda.blockIdx.x*cuda.blockDim.x
#   if time_index < amplitude.size:
#     amplitude_previous = amplitude[time_index]
#     if amplitude_previous > 0:
#       amplitude[time_index] -= step_size
#     elif amplitude_previous < 0:
#       amplitude[time_index] += step_size
#     # Set to zero rather than oscillate
#     if amplitude[time_index]*amplitude_previous < 0:
#       amplitude[time_index] = 0

# @cuda.jit()
# def reconstruct_ista_prediction_step(frequency_amplitude, amplitude, fourier_transform, frequency_amplitude_prediction):
#   """
#   Take the sine Fourier transform of the reconstructed signal, and compare it to the measured frequency components. Returns the difference.
#   """
#   frequency_index = cuda.threadIdx.x + cuda.blockIdx.x*cuda.blockDim.x
#   if frequency_index < frequency_amplitude_prediction.size:
#     frequency_amplitude_prediction[frequency_index] = -frequency_amplitude[frequency_index]
#     for time_index in range(amplitude.size):
#       frequency_amplitude_prediction[frequency_index] += fourier_transform[frequency_index, time_index]*amplitude[time_index]

# @cuda.jit()
# def reconstruct_ista_manifold_step(frequency_amplitude_prediction, step_size, fourier_transform, amplitude):
#   """
#   Use gradient decent to bring the reconstruction closer to having the correct partial sine Fourier transform.

#   min X(r),
#   X(r) = norm2(S r - s) (in frequency)
#   dX(r)/dr = 2 S^T (S r - s)
#   """
#   time_index = cuda.threadIdx.x + cuda.blockIdx.x*cuda.blockDim.x
#   if time_index < amplitude.size:
#     for frequency_index in range(frequency_amplitude_prediction.size):
#       amplitude[time_index] -= fourier_transform[frequency_index, time_index]*(frequency_amplitude_prediction[frequency_index])*step_size