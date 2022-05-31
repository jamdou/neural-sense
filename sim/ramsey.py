import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import spinsim
from sim.manager import generate_interpolation_sampler

import util
from util import PrettyTritty as C
import archive as arch
import test_signal
import analysis

def simulate_ramsey(scaled:util.ScaledParameters, archive:arch.Archive = None, line_noise_model:test_signal.LineNoiseModel = None, lab_harmonics = None, signal = None):
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

  lab_harmonic_amplitudes = []
  lab_harmonic_frequencies = []
  if lab_harmonics is not None:
    for lab_harmonic in lab_harmonics:
      lab_harmonic_amplitudes.append(lab_harmonic.amplitude[2])
      lab_harmonic_frequencies.append(lab_harmonic.frequency[2])
  lab_harmonic_amplitudes = np.array(lab_harmonic_amplitudes)
  lab_harmonic_frequencies= np.array(lab_harmonic_frequencies)

  interpolation_sampler = generate_interpolation_sampler(amplitude = signal.signal_trace_amplitude, time = signal.signal_trace_time)
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

    # for time_neural_instance in time_neural:
    #   if time_neural_instance <= time_sample and time_sample < time_neural_instance + 1/frequency_neural:
    #     field_strength[2] += math.tau*amplitude_neural*math.sin(math.tau*frequency_neural*(time_sample - time_neural_instance))
    interpolation_sampler(time_sample, parameters, field_strength)
    for line_noise_index, (line_noise_amplitude, line_noise_phase) in enumerate(zip(line_noise_amplitudes, line_noise_phases)):
      if line_noise_amplitude > 0:
        field_strength[2] += math.tau*line_noise_amplitude*math.cos(math.tau*50*(line_noise_index + 1)*time_sample + line_noise_phase)
    
    for lab_harmonic_amplitude, lab_harmonic_frequency in zip(lab_harmonic_amplitudes, lab_harmonic_frequencies):
      field_strength[2] += math.tau*lab_harmonic_amplitude*math.sin(math.tau*lab_harmonic_frequency*time_sample)

    field_strength[3] = 0


  time = np.arange(0, scaled.time_end, scaled.time_step)
  amplitude = np.empty_like(time)
  simulator = spinsim.Simulator(ramsey_hamiltonian, spinsim.SpinQuantumNumber.ONE)
  C.starting("Ramsey simulations")
  C.print(f"|{'Index':8}|{'Perc':8}|")
  for time_index, time_sensing in enumerate(time):
    results = simulator.evaluate(-math.ceil((duration_sensing/2 + duration_pulse)/scaled.time_step)*scaled.time_step, scaled.time_end + math.ceil((duration_sensing/2 + duration_pulse)/scaled.time_step)*scaled.time_step, scaled.time_step/100, scaled.time_step, spinsim.SpinQuantumNumber.ONE.plus_z, [time_sensing])
    amplitude[time_index] = results.spin[-1, 2]/(duration_sensing*math.tau)
    C.print(f"|{time_index:8d}|{(time_index + 1)/time.size*100:7.2f}%|", end = "\r")
  C.print("\n")
  C.finished("Ramsey simulations")

  # amplitude = amplitude[np.logical_and(0 <= time, time < scaled.time_end)]
  # time = time[np.logical_and(0 <= time, time < scaled.time_end)]

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

class RamseyComparisonResults():
  def __init__(self, error_0, error_1, error_2, error_sup, error_snr, error_mf, error_mfd, error_mfp, error_mfpd, error_sensitivity, error_specificity):
    self.error_0 = error_0
    self.error_1 = error_1
    self.error_2 = error_2
    self.error_sup = error_sup
    self.error_snr = error_snr
    self.error_mf = error_mf
    self.error_mfd = error_mfd
    self.error_mfp = error_mfp
    self.error_mfpd = error_mfpd
    self.error_sensitivity = error_sensitivity
    self.error_specificity = error_specificity

  def write_to_file(self, archive:arch.Archive):
    group = archive.archive_file.require_group("ramsey_comparison")
    group.attrs["error_0"] = self.error_0
    group.attrs["error_1"] = self.error_1
    group.attrs["error_2"] = self.error_2
    group.attrs["error_sup"] = self.error_sup
    group.attrs["error_snr"] = self.error_snr
    group.attrs["error_mf"] = self.error_mf
    group.attrs["error_mfd"] = self.error_mfd
    group.attrs["error_mfp"] = self.error_mfp
    group.attrs["error_mfpd"] = self.error_mfpd
    group.attrs["error_sensitivity"] = self.error_sensitivity
    group.attrs["error_specificity"] = self.error_specificity

def compare_to_test_signal(results:arch.RamseyResults, signal:test_signal.TestSignal, archive:arch.Archive = None, units = "Hz"):
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
  amplitude = np.array(measured_amplitude)
  original_amplitude = np.array(original_amplitude)

  expected_signal_power = np.mean(original_amplitude**2)
  mf_cutoff = expected_signal_power/2 # default
  mfp_cutoff = mf_cutoff

  error_0 = np.mean((amplitude == 0)*(original_amplitude != 0) + (amplitude != 0)*(original_amplitude == 0))
  error_1 = np.mean(np.abs(amplitude - original_amplitude))
  error_2 = math.sqrt(np.mean((amplitude - original_amplitude)**2))
  error_sup = np.max(np.abs(amplitude - original_amplitude))
  error_snr = expected_signal_power/np.mean((amplitude - original_amplitude)**2)

  error_mf = np.mean(amplitude*original_amplitude)
  error_mfd = error_mf >= mf_cutoff

  error_mfp = np.max(np.abs(scipy.signal.correlate((amplitude - (error_mf/expected_signal_power)*original_amplitude), original_amplitude)))/original_amplitude.size
  error_mfpd = error_mfp >= mfp_cutoff

  expected_frequency = 5e3
  expected_amplitude = 1e3
  template_time = np.arange(0, 1/expected_frequency, time[1] - time[0])
  template_amplitude = expected_amplitude*np.sin(math.tau*expected_frequency*template_time)
  matched_cutoff = np.sum(template_amplitude**2)/2
  matched_ground_truth = np.abs(scipy.signal.correlate(original_amplitude, template_amplitude)) >= matched_cutoff
  matched_decision = np.abs(scipy.signal.correlate(amplitude, template_amplitude)) >= matched_cutoff
  if np.sum(matched_ground_truth) > 0:
    error_sensitivity = np.sum(np.logical_and(matched_ground_truth, matched_decision))/np.sum(matched_ground_truth)
  else:
    error_sensitivity = 1
  if np.sum(np.logical_not(matched_ground_truth)) > 0:
    error_specificity = np.sum(np.logical_and(np.logical_not(matched_ground_truth), np.logical_not(matched_decision)))/np.sum(np.logical_not(matched_ground_truth))
  else:
    error_specificity = 1

  ramsey_comparison_results = RamseyComparisonResults(error_0, error_1, error_2, error_sup, error_snr, error_mf, error_mfd, error_mfp, error_mfpd, error_sensitivity, error_specificity)
  if archive is not None:
    ramsey_comparison_results.write_to_file(archive)

  
  # if archive is not None:
  #   group = archive.archive_file.require_group("ramsey_comparison")
  #   group.attrs["error_0"] = error_0
  #   group.attrs["error_1"] = error_1
  #   group.attrs["error_2"] = error_2
  #   group.attrs["error_sup"] = error_sup
  #   group.attrs["error_snr"] = error_snr
  #   group.attrs["error_mf"] = error_mf
  #   group.attrs["error_mfd"] = error_mfd
  #   group.attrs["error_mfp"] = error_mfp
  #   group.attrs["error_mfpd"] = error_mfpd

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

  C.print(f"Support error: {error_0*100} %")
  C.print(f"Average error: {error_1*unit_factor} {units}")
  C.print(f"RMS error: {error_2*unit_factor} {units}")
  C.print(f"Maximum error: {error_sup*unit_factor} {units}")
  if error_snr > 0:
    C.print(f"Signal to error ratio: {10*math.log10(error_snr)} dB")
  C.print(f"Matched filter response: {error_mf*unit_factor**2} {units}^2")
  C.print(f"Matched filter threshold: {mf_cutoff*unit_factor**2} {units}^2")
  C.print(f"Detected: {error_mfd}")
  C.print(f"Matched filter false positive response: {error_mfp*unit_factor**2} {units}^2")
  C.print(f"Matched filter false positive threshold: {mfp_cutoff*unit_factor**2} {units}^2")
  C.print(f"Detected: {error_mfpd}")
  C.print(f"Sensitivity: {error_sensitivity*100} %")
  C.print(f"Specificity: {error_specificity*100} %")

  plt.figure()
  plt.plot(time, original_amplitude, "k-", label = "Original")
  plt.plot(time, measured_amplitude, "r-", label = "Measured")
  plt.xlabel("Time (s)")
  plt.ylabel("Amplitude (Hz)")
  plt.legend()
  if archive:
    archive.write_plot("Ramsey comparison", "ramsey_comparison")
  plt.draw()

  return ramsey_comparison_results

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

def remove_dc(results:arch.RamseyResults):
  amplitude = results.amplitude.copy()
  time = results.time.copy()
  amplitude_modified = amplitude - np.mean(amplitude)

  results_modified = arch.RamseyResults(time = time[1:], amplitude = amplitude_modified[1:], archive_time = results.archive_time, experiment_type = f"{results.experiment_type}, removed dc")
  return results_modified
