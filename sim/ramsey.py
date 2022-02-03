import math
import numpy as np
import matplotlib.pyplot as plt
import spinsim

import util
from util import PrettyTritty as C
import archive as arch
import test_signal
import analysis

def simulate_ramsey(scaled:util.ScaledParameters, archive:arch.Archive = None, line_noise_model:test_signal.LineNoiseModel = None):
  bias = 600e3
  duration_sensing = 60e-6
  amplitude_pulse = 20e3
  duration_pulse = 1/(4*amplitude_pulse)
  time_neural = scaled.pulse_time
  frequency_neural = scaled.frequency
  amplitude_neural = scaled.amplitude
  amplitude_line = 500

  if line_noise_model is not None:
    line_noise_amplitudes = line_noise_model.a.copy()
    line_noise_phases = line_noise_model.p.copy()
  else:
    line_noise_amplitudes = np.empty(0)
    line_noise_phases = np.empty(0)

  def ramsey_hamiltonian(time_sample, parameters, field_strength):
    time_sensing = parameters[0]
    if (time_sensing - duration_sensing/2 - duration_pulse <= time_sample) and (time_sample < time_sensing - duration_sensing/2):
      field_strength[0] = 2*math.tau*amplitude_pulse*math.cos(math.tau*bias*(time_sample))
    elif (time_sensing + duration_sensing/2 < time_sample) and (time_sample <= time_sensing + duration_sensing/2 + duration_pulse):
      field_strength[0] = 2*math.tau*amplitude_pulse*math.cos(math.tau*bias*(time_sample) - math.pi/2)
    else:
      field_strength[0] = 0
    field_strength[1] = 0
    field_strength[2] = math.tau*bias
    # field_strength[2] += math.tau*amplitude_line*math.sin(math.tau*50*time_sample)\

    for time_neural_instance in time_neural:
      if time_neural_instance <= time_sample and time_sample < time_neural_instance + 1/frequency_neural:
        field_strength[2] += math.tau*amplitude_neural*math.sin(math.tau*frequency_neural*(time_sample - time_neural_instance))
    for line_noise_index, (line_noise_amplitude, line_noise_phase) in enumerate(zip(line_noise_amplitudes, line_noise_phases)):
      if line_noise_amplitude > 0:
        field_strength[2] += math.tau*line_noise_amplitude*math.cos(math.tau*50*(line_noise_index + 1)*time_sample + line_noise_phase)
    field_strength[3] = 0


  time = np.arange(0, scaled.time_end, scaled.time_step)
  amplitude = np.empty_like(time)
  simulator = spinsim.Simulator(ramsey_hamiltonian, spinsim.SpinQuantumNumber.ONE)
  C.starting("Ramsey simulations")
  C.print(f"|{'Index':8}|{'Perc':8}|")
  for time_index, time_sensing in enumerate(time):
    results = simulator.evaluate(0, scaled.time_end, scaled.time_step/100, scaled.time_step, spinsim.SpinQuantumNumber.ONE.plus_z, [time_sensing])
    amplitude[time_index] = results.spin[-1, 2]/(duration_sensing*math.tau)
    C.print(f"|{time_index:8d}|{(time_index + 1)/time.size*100:7.2f}%|", end = "\r")
  C.print("\n")
  C.finished("Ramsey simulations")
  # if archive:
  #   ramsey_group = archive.archive_file.create_group("simulations_ramsey")
  #   ramsey_group["amplitude"] = amplitude
  #   ramsey_group["time"] = time
  

  return arch.RamseyResults(time = time, amplitude = amplitude, archive_time = archive.execution_time_string, experiment_type = "simulation")

def remove_line_noise_bias(results:arch.RamseyResults, empty_results:arch.RamseyResults):
  time = []
  amplitude = []
  for empty_time_instance, empty_amplitude_instance in zip(empty_results.time, empty_results.amplitude):
    for time_instance, amplitude_instance in zip(results.time, results.amplitude):
      if math.isclose(empty_time_instance, time_instance):
        time.append(time_instance)
        amplitude.append(amplitude_instance - empty_amplitude_instance)
  time = np.array(time)
  amplitude = np.array(amplitude)

  return arch.RamseyResults(time = time, amplitude = amplitude, archive_time = results.archive_time, experiment_type = f"{results.experiment_type}, line noise bias removed")

def compare_to_test_signal(results:arch.RamseyResults, signal:test_signal.TestSignal, archive:arch.Archive = None):
  time = []
  measured_amplitude = []
  original_amplitude = []
  for measured_time_instance, measured_amplitude_instance in zip(results.time, results.amplitude):
    for original_time_instance, original_amplitude_instance in zip(signal.time_properties.time_coarse, signal.amplitude):
      if math.isclose(measured_time_instance, original_time_instance):
        time.append(measured_time_instance)
        measured_amplitude.append(measured_amplitude_instance)
        original_amplitude.append(original_amplitude_instance)
  time = np.array(time)
  measured_amplitude = np.array(measured_amplitude)
  original_amplitude = np.array(original_amplitude)
  error_rms = np.sqrt(np.mean((measured_amplitude - original_amplitude)**2))
  C.print(f"Ramsey error rms: {error_rms}")
  plt.figure()
  plt.plot(time, original_amplitude, "k-", label = "Original")
  plt.plot(time, measured_amplitude, "r-", label = "Measured")
  plt.xlabel("Time (s)")
  plt.ylabel("Amplitude (Hz)")
  plt.legend()
  if archive:
    archive.write_plot("Ramsey comparison", "ramsey_comparison")
  plt.draw()

def mode_filter(results:arch.RamseyResults):
  amplitude = results.amplitude.copy()
  time = results.time.copy()
  time_step = time[1] - time[0]
  frequency_max = 1/(2*time_step)
  d_frequency_min = frequency_max/(time.size + 1)
  frequency = np.arange(d_frequency_min, frequency_max - d_frequency_min/2, d_frequency_min)

  frequency_mesh, time_mesh = np.meshgrid(frequency, time)
  fourier_transform = np.sin(math.tau*frequency_mesh*time_mesh)
  frequency_amplitude = fourier_transform@amplitude

  experiment_results = arch.ExperimentResults(frequency = frequency, frequency_amplitude = frequency_amplitude)
  experiment_results = analysis.mode_filter(experiment_results)

  frequency_amplitude = experiment_results.frequency_amplitude
  amplitude_modified = np.linalg.lstsq(fourier_transform, frequency_amplitude, rcond = None)[0]

  results_modified = arch.RamseyResults(time = time, amplitude = amplitude_modified, archive_time = results.archive_time, experiment_type = f"{results.experiment_type}, mode filter")
  return results_modified
