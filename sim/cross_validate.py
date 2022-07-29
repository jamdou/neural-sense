import numpy as np
import math, cmath
import time
import matplotlib.pyplot as plt

import spinsim

import archive as arch
import test_signal
from util import PrettyTritty as C
import reconstruction as recon

class ResultsCompilation:
  def __init__(self, frequency:np.ndarray, frequency_amplitudes:np.ndarray, pulse_times:np.ndarray, dc_detunings:np.ndarray, time_properties:test_signal.TimeProperties, amplitudes:np.ndarray):
    self.frequency = frequency
    self.frequency_amplitudes = frequency_amplitudes
    self.pulse_times = pulse_times
    self.dc_detunings = dc_detunings
    self.time_properties = time_properties
    self.amplitudes = amplitudes
  
  def write_to_file(self, archive:arch.Archive):
    archive_group = archive.archive_file.require_group("cross_validation/results_compilation")
    archive_group["frequency"] = self.frequency
    archive_group["pulse_times"] = self.pulse_times
    archive_group.attrs["time_step"] = self.time_properties.time_step_coarse
    archive_group.attrs["time_end"] = self.time_properties.time_end_points[1]

  @staticmethod
  def generate_inputs(frequency_max = 10e3, frequency_step = 100, time_end = 5e-3, number_of_experiments = 720, number_of_pulses_max = 2, detuning_std = 200, pulse_duration = 1/5e3):
    frequency =  np.arange(frequency_step, frequency_max, frequency_step)
    dc_detunings = np.random.normal(0, detuning_std, (number_of_experiments, frequency.size))
    time_step =time_end/(frequency_max/frequency_step)
    time_properties = test_signal.TimeProperties(
      time_step_coarse = time_step,
      time_end_points = [time_step, time_end]
    )
    time_properties.time_coarse = np.arange(time_step, time_end, time_step)

    pulse_times = np.random.uniform(size = (number_of_experiments, number_of_pulses_max))
    number_of_pulses = np.random.randint(0, number_of_pulses_max + 1, number_of_experiments)
    pulse_times *= time_end/number_of_pulses.reshape((pulse_times.shape[0], 1)) - pulse_duration
    pulse_times[:, 1] += pulse_times[:, 0] + pulse_duration
    pulse_times[number_of_pulses < 2, 1] = -1
    pulse_times[number_of_pulses < 1, 0] = -1
    
    return ResultsCompilation(frequency, np.empty_like(dc_detunings), pulse_times, dc_detunings, time_properties, np.empty_like(dc_detunings))

  @staticmethod
  def read_simulations_from_archive_time(archive:arch.Archive, archive_time):
    archive_previous = arch.Archive(archive.archive_path[:-25], "")
    archive_previous.open_archive_file(archive_time)
    archive_group = archive_previous.archive_file.require_group("cross_validation/results_compilation")

    frequency = np.asarray(archive_group["frequency"])
    pulse_times = np.asarray(archive_group["pulse_times"])
    time_step = archive_group.attrs["time_step"]
    time_end = archive_group.attrs["time_end"]
    time_properties = test_signal.TimeProperties(
      time_step_coarse = time_step,
      time_end_points = [time_step, time_end]
    )
    time_properties.time_coarse = np.arange(time_step, time_end, time_step)

    archive_group_frequency = archive_previous.archive_file.require_group("cross_validation/results_compilation/frequency_amplitudes")
    archive_group_amplitudes = archive_previous.archive_file.require_group("cross_validation/results_compilation/amplitudes")
    frequency_amplitudes = []
    amplitudes = []
    experiment_index = 0
    while f"{experiment_index}" in archive_group_frequency:
      frequency_amplitudes.append(archive_group_frequency[f"{experiment_index}"])
      amplitudes.append(archive_group_amplitudes[f"{experiment_index}"])
      experiment_index += 1
    frequency_amplitudes = np.array(frequency_amplitudes)
    amplitudes = np.array(amplitudes)

    return ResultsCompilation(frequency, frequency_amplitudes, pulse_times, np.empty_like(frequency_amplitudes), time_properties, amplitudes)


  def simulate(self, archive:arch.Archive = None):
    C.starting("cross validation simulations")

    time_end = 5e-3
    readout_amplitude = 20e3
    neural_amplitude = 1000#360
    neural_frequency = 5e3
    bias = 600e3

    atom_count = 10e3

    if archive is not None:
      archive_group = archive.archive_file.require_group("cross_validation/results_compilation")
      archive_group.attrs["time_end"] = time_end
      archive_group.attrs["readout_amplitude"] = readout_amplitude
      archive_group.attrs["neural_amplitude"] = neural_amplitude
      archive_group.attrs["neural_frequency"] = neural_frequency
      archive_group.attrs["bias"] = bias
      archive_group.attrs["atom_count"] = atom_count

    def get_field(time, parameters, field):
      dressing = parameters[0]
      dc_detuning = parameters[1]

      pulse_time_0 = parameters[2]
      pulse_time_1 = parameters[3]

      field[2] = math.tau*(bias + dc_detuning)
      if time > pulse_time_0 and time < pulse_time_0 + 1/neural_frequency:
        field[2] += math.tau*neural_amplitude*math.sin(math.tau*neural_frequency*(time - pulse_time_0))
      if time > pulse_time_1 and time < pulse_time_1 + 1/neural_frequency:
        field[2] += math.tau*neural_amplitude*math.sin(math.tau*neural_frequency*(time - pulse_time_1))
      
      field[0] = 0
      if time < time_end:
        field[0] += math.tau*2*dressing*math.cos(math.tau*bias*time)
      elif time < time_end + (1/readout_amplitude)/4:
        field[0] += math.tau*2*readout_amplitude*math.sin(math.tau*bias*time)

      field[1] = 0
      field[3] = 0

    simulator = spinsim.Simulator(get_field, spinsim.SpinQuantumNumber.ONE, threads_per_block = 256)

    wall_time_start = time.time()
    wall_time_current = 0
    C.print(f"|{'Experiment index':>16s}|{'Shot index':>16s}|{'Time elapsed (s)':>16s}|{'Time step (s)':>16s}|")
    for experiment_index in range(self.frequency_amplitudes.shape[0]):
      # Calculate amplitude
      self.amplitudes[experiment_index, :] = 0
      mask = np.logical_and(self.time_properties.time_coarse > self.pulse_times[experiment_index, 0], self.time_properties.time_coarse < self.pulse_times[experiment_index, 0] + 1/neural_frequency)
      self.amplitudes[experiment_index, mask] = neural_amplitude*np.sin(math.tau*neural_frequency*(self.time_properties.time_coarse[mask] - self.pulse_times[experiment_index, 0]))
      mask = np.logical_and(self.time_properties.time_coarse > self.pulse_times[experiment_index, 1], self.time_properties.time_coarse < self.pulse_times[experiment_index, 1] + 1/neural_frequency)
      self.amplitudes[experiment_index, mask] = neural_amplitude*np.sin(math.tau*neural_frequency*(self.time_properties.time_coarse[mask] - self.pulse_times[experiment_index, 1]))
      if archive is not None:
        archive_group = archive.archive_file.require_group("cross_validation/results_compilation/amplitudes")
        archive_group[f"{experiment_index}"] = self.amplitudes[experiment_index, :]

      # Calculate frequency
      for shot_index in range(self.frequency_amplitudes.shape[1]):
        wall_time_previous = wall_time_current

        result = simulator.evaluate(0, 5.1e-3, 5e-8, 1e-6, spinsim.SpinQuantumNumber.ONE.minus_z, [self.frequency[shot_index], self.dc_detunings[experiment_index, shot_index], self.pulse_times[experiment_index, 0], self.pulse_times[experiment_index, 1]])

        atom_count_plus = np.random.poisson(atom_count*np.abs(result.state[-1, 0]**2))
        atom_count_zero = np.random.poisson(atom_count*np.abs(result.state[-1, 1]**2))
        atom_count_minus = np.random.poisson(atom_count*np.abs(result.state[-1, 2]**2))
        self.frequency_amplitudes[experiment_index, shot_index] = (1/(time_end*math.tau))*(atom_count_minus - atom_count_plus)/(atom_count_minus + atom_count_zero + atom_count_plus)

        wall_time_current = time.time() - wall_time_start
        wall_time_step = wall_time_current - wall_time_previous
        C.print(f"|{experiment_index:16d}|{shot_index:16d}|{wall_time_current:16.6f}|{wall_time_step:16.6f}|", end="\r")
      if archive is not None:
        archive_group = archive.archive_file.require_group("cross_validation/results_compilation/frequency_amplitudes")
        archive_group[f"{experiment_index}"] = self.frequency_amplitudes[experiment_index, :]
    C.print("\n")

    C.finished("cross validation simulations")

  def reconstruct(self, archive:arch.Archive = None, number_of_samples = 50, metric = "rms"):
    C.starting("cross validation reconstructions")
    reconstruction = recon.Reconstruction(self.time_properties)

    self.number_of_experiments = self.frequency_amplitudes.shape[0]
    self.norm_scale_factor = np.geomspace(0.1, 10, number_of_samples)

    if archive is not None:
      archive_group = archive.archive_file.require_group("cross_validation/reconstructions")
      archive_group["norm_scale_factor"] = self.norm_scale_factor
      archive_group.attrs["number_of_experiments"] = self.number_of_experiments
    archive_group = archive.archive_file.require_group("cross_validation/reconstructions/error")
    self.error_array = []
    wall_time_start = time.time()
    wall_time_current = 0
    for experiment_index in range(self.number_of_experiments):
      wall_time_previous = wall_time_current
      if metric == "scaled rms":
        power = np.dot(self.amplitudes[experiment_index, :], self.amplitudes[experiment_index, :])

      reconstruction.read_frequencies_directly(
        self.frequency,
        self.frequency_amplitudes[experiment_index, :],
        60,
        # 100,
        0, 20e3
      )
      error_experiment = []
      for norm_scale_factor_instance in self.norm_scale_factor:
        reconstruction.evaluate_fista_backtracking(
          expected_amplitude = 1000,#360.0,
          expected_frequency = 5000,
          expected_error_measurement = 5,
          norm_scale_factor_modifier = 1,
          is_fast = True,
          norm_scale_factor = norm_scale_factor_instance
        )
        if metric == "rms":
          error_experiment.append(np.sqrt(np.sum((reconstruction.amplitude - self.amplitudes[experiment_index, :])**2)))
        elif metric == "scaled rms":
          if power > 0:
            dot = np.dot(reconstruction.amplitude, self.amplitudes[experiment_index, :])
            # if dot > 0.2:
            error_experiment.append(np.sqrt(np.sum((reconstruction.amplitude - (dot/power)*self.amplitudes[experiment_index, :])**2)) + (np.sqrt(power) - np.sqrt(dot)))
            # else:
            #   error_experiment.append(np.sqrt(np.sum((reconstruction.amplitude - self.amplitudes[experiment_index, :])**2)))
          else:
            error_experiment.append(np.sqrt(np.sum((reconstruction.amplitude)**2)))
        elif metric == "support":
          error_experiment.append(np.sum(np.logical_and(np.abs(reconstruction.amplitude) > 5, np.abs(self.amplitudes[experiment_index, :]) < 5) + 24*np.logical_and(np.abs(reconstruction.amplitude) < 5, np.abs(self.amplitudes[experiment_index, :]) > 5))/25)
          # error_experiment.append(np.sum(np.logical_and(np.abs(reconstruction.amplitude) > 0, np.abs(self.amplitudes[experiment_index, :]) < 100)) + 25*np.sum(np.logical_and(np.abs(reconstruction.amplitude) < 0.7*np.abs(self.amplitudes[experiment_index, :]), np.abs(self.amplitudes[experiment_index, :]) > 100)))
      error_experiment = np.array(error_experiment)
      self.error_array.append(error_experiment)
      archive_group[f"{experiment_index}"] = error_experiment

      wall_time_current = time.time() - wall_time_start
      wall_time_step = wall_time_current - wall_time_previous
      C.print(f"Time: {wall_time_current} s, Time step: {wall_time_step} s")

    self.error_array = np.array(self.error_array)
    C.finished("cross validation reconstructions")

  def read_reconstructions_from_archive_time(self, archive:arch.Archive, archive_time):
    archive_previous = arch.Archive(archive.archive_path[:-25], "")
    archive_previous.open_archive_file(archive_time)

    archive_group = archive_previous.archive_file.require_group("cross_validation/reconstructions")
    self.number_of_experiments = archive_group.attrs["number_of_experiments"]
    self.norm_scale_factor = np.asarray(archive_group["norm_scale_factor"])

    archive_group = archive_previous.archive_file.require_group("cross_validation/reconstructions/error")
    experiment_index = 0
    self.error_array = []
    while f"{experiment_index}" in archive_group:
      self.error_array.append(np.asarray(archive_group[f"{experiment_index}"]))
      experiment_index += 1
    self.error_array = np.array(self.error_array)

  def cross_validate(self, archive:arch.Archive = None, units = "Hz", number_of_folds = 5, metric = "rms"):

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

    C.starting("cross validation error minimisation")

    fold = np.arange(number_of_folds)
    indices = np.arange(self.number_of_experiments)
    error = []
    error_min = []
    norm_scale_factor_min = []
    plt.figure()
    for fold_index in fold:
      mask = np.logical_or(indices*fold.size/indices.size < fold_index, indices*fold.size/indices.size >= fold_index + 1)
      error.append(np.mean(self.error_array[mask, :], axis = 0))
      error_min.append(np.min(error[-1]))
      norm_scale_factor_min.append(self.norm_scale_factor[np.argmin(error[-1])])

      plt.loglog(self.norm_scale_factor*unit_factor, error[-1]*unit_factor, "g", alpha = 1/number_of_folds)
      plt.loglog([norm_scale_factor_min[-1]*unit_factor], [error_min[-1]*unit_factor], "co", alpha = 1/number_of_folds)

    error = np.array(error)
    error_min = np.array(error_min)
    norm_scale_factor_min = np.array(norm_scale_factor_min)

    norm_scale_factor_min_mean = np.mean(norm_scale_factor_min)
    norm_scale_factor_min_std = np.std(norm_scale_factor_min)

    if archive is not None:
      archive_group = archive.archive_file.require_group("cross_validation/cross_validation")
      archive_group["norm_scale_factor"] = self.norm_scale_factor
      archive_group["error"] = error
      archive_group["error_min"] = error_min
      archive_group["norm_scale_factor_min"] = norm_scale_factor_min
      archive_group["norm_scale_factor_min"].attrs["mean"] = norm_scale_factor_min_mean
      archive_group["norm_scale_factor_min"].attrs["std"] = norm_scale_factor_min_std
    C.print(f"λ (trained): {norm_scale_factor_min_mean*unit_factor} +- {norm_scale_factor_min_std*unit_factor} {units}")

    plt.xlabel(f"Regularisation parameter ({units})")
    if metric == "rms":
      plt.ylabel(f"Average RMSE for training set ({units})")
    elif metric == "scaled rms":
      plt.ylabel(f"Average noise RMS for training set ({units})")
    elif metric == "support":
      plt.ylabel(f"Weighted support error for training set")
    plt.legend(["Error curve for training", "Regularisation parameter determined from training set"])
    if archive:
      archive.write_plot(f"Cross validation:\nMinimising regularisation parameter", f"cross_validation_min")
    plt.draw()
    C.finished("cross validation error minimisation")

    C.starting("cross validation validation step")
    error_validation = []
    for fold_index in fold:
      mask = np.logical_and(indices*fold.size/indices.size >= fold_index, indices*fold.size/indices.size < fold_index + 1)
      error_validation.append(np.mean(self.error_array[mask, self.norm_scale_factor == norm_scale_factor_min[fold_index]]))
    error_validation = np.array(error_validation)
    error_validation_mean = np.mean(error_validation)
    error_validation_std = np.std(error_validation)
    if archive is not None:
      archive_group = archive.archive_file.require_group("cross_validation/cross_validation")
      archive_group["error_validation"] = error_validation
      archive_group["error_validation"].attrs["mean"] = error_validation_mean
      archive_group["error_validation"].attrs["std"] = error_validation_std
    C.print(f"Validation noise: {error_validation_mean*unit_factor} +- {error_validation_std*unit_factor} {units}")

    plt.figure()
    plt.plot(fold, error_validation*unit_factor, "yo", label = "Validation error")
    plt.plot(fold, error_min*unit_factor, "co", label = "Training error")
    plt.xlabel("Iteration")
    if metric == "rms":
      plt.ylabel(f"Average validation RMSE ({units})")
    elif metric == "scaled rms":
      plt.ylabel(f"Average validation noise RMS ({units})")
    plt.legend()
    if archive:
      archive.write_plot(f"Cross validation:\nValidation error", f"cross_validation_validation")
    plt.draw()
    C.finished("cross validation validation step")