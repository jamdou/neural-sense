import math
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use("tkagg")
# import sys, getopt    # Command line arguments
from numba import cuda  # GPU code
import colorama     # Colourful terminal
colorama.init()
from cmcrameri import cm

# The different pieces that make up this sensing code
import archive as arch        # Saving results and configurations
from archive import handle_arguments

import test_signal          # The properties of the magnetic signal, used for simulations and reconstructions
import spinsim            # Main simulation package
import reconstruction as recon    # Uses compressive sensing to reconstruct the a magnetic signal
import sim
import util
from util import PrettyTritty as C
import analysis

class CompressivePaper:
  @staticmethod
  def make_comparison(archive):
    experiment_time = "20220517T111439" # One signal, By_aux, Compressive
    scaled = util.ScaledParameters.new_from_experiment_time(experiment_time)

    time_properties_reconstruction = test_signal.TimeProperties(scaled.time_step, scaled.time_step/np.ceil(scaled.time_step/1e-5), 1e-8, [scaled.time_step, scaled.time_end])

    acquired_signal_1 = test_signal.AcquiredSignal.new_from_experiment_time(archive, "20220517T111439")
    acquired_time_1, acquired_amplitude_1 = acquired_signal_1.subsample(scaled.time_step, archive, "Hz")
    # acquired_amplitude_1 /= 0.4
    signal_reconstruction_1 = test_signal.TestSignal(scaled.get_neural_pulses(), [], time_properties_reconstruction, signal_trace_time = acquired_time_1, signal_trace_amplitude = acquired_amplitude_1)

    ramsey_results_1 = arch.RamseyResults.new_from_archive_time(archive, "20220520T143240")
    ramsey_results_1 = sim.ramsey.remove_dc(ramsey_results_1)

    experiment_results_1 = arch.ExperimentResults.new_from_archive_time(archive, "20220517T111439")
    reconstruction_dst_1 = recon.Reconstruction(signal_reconstruction_1.time_properties)
    reconstruction_dst_1.read_frequencies_from_experiment_results(experiment_results_1, number_of_samples = 99, random_seed = util.Seeds.metroid)
    reconstruction_dst_1.evaluate_least_squares()

    reconstruction_fst_1 = recon.Reconstruction(signal_reconstruction_1.time_properties)
    reconstruction_fst_1.read_frequencies_from_experiment_results(experiment_results_1, number_of_samples = 60, random_seed = util.Seeds.metroid)
    reconstruction_fst_1.evaluate_fista_backtracking(
      expected_amplitude = scaled.amplitude,
      expected_frequency = scaled.frequency,
      expected_error_measurement = 1.8,
      norm_scale_factor_modifier = 1,
      is_fast = True,
      norm_scale_factor = 1.0377456768938575#1.434744597683189#1.246014062993185#1.238443214398892#1.291549665014884#2.1544346900318834#0.7488103857590022
    )

    acquired_signal_2 = test_signal.AcquiredSignal.new_from_experiment_time(archive, "20220520T111524")
    acquired_time_2, acquired_amplitude_2 = acquired_signal_2.subsample(scaled.time_step, archive, "Hz")
    # acquired_amplitude_2 /= 0.4
    signal_reconstruction_2 = test_signal.TestSignal(scaled.get_neural_pulses(), [], time_properties_reconstruction, signal_trace_time = acquired_time_2, signal_trace_amplitude = acquired_amplitude_2)

    ramsey_results_2 = arch.RamseyResults.new_from_archive_time(archive, "20220523T141012")
    ramsey_results_2 = sim.ramsey.remove_dc(ramsey_results_2)

    experiment_results_2 = arch.ExperimentResults.new_from_archive_time(archive, "20220520T111524")
    reconstruction_dst_2 = recon.Reconstruction(signal_reconstruction_2.time_properties)
    reconstruction_dst_2.read_frequencies_from_experiment_results(experiment_results_2, number_of_samples = 99, random_seed = util.Seeds.metroid)
    reconstruction_dst_2.evaluate_least_squares()

    reconstruction_fst_2 = recon.Reconstruction(signal_reconstruction_2.time_properties)
    reconstruction_fst_2.read_frequencies_from_experiment_results(experiment_results_2, number_of_samples = 60, random_seed = util.Seeds.metroid)
    reconstruction_fst_2.evaluate_fista_backtracking(
      expected_amplitude = scaled.amplitude,
      expected_frequency = scaled.frequency,
      expected_error_measurement = 1.8,
      norm_scale_factor_modifier = 1,
      is_fast = True,
      norm_scale_factor = 1.0377456768938575#1.434744597683189#1.246014062993185#1.238443214398892#1.291549665014884#2.1544346900318834#0.7488103857590022
    )

    # recon.plot_reconstruction_method_comparison(archive, [ramsey_results_1, ramsey_results_2, reconstruction_dst_1, reconstruction_dst_2, reconstruction_fst_1, reconstruction_fst_2], [acquired_amplitude_1, acquired_amplitude_2])

    CompressivePaper.make_matrix(archive, acquired_time_2, acquired_amplitude_2, reconstruction_fst_2.amplitude, reconstruction_fst_2.frequency, reconstruction_fst_2.frequency_amplitude)

  @staticmethod
  def make_matrix(archive, time, amplitude, amplitude_recover, frequency, frequency_amplitude):
    line_ratio = 4
    # borderline = [1.0, 1.0, 1.0, 1.0]
    # borderline = [0.0, 0.0, 0.0, 1.0]
    borderline = [0.0, 0.0, 0.0, 0.0]
    borderline_true = [0.0, 0.0, 0.0, 1.0]
    colour_map = cm.tokyo

    plt.figure()
    lim = np.max(np.abs(frequency_amplitude))
    frequency_amplitude_colour = []
    frequency_amplitude_colour += [[borderline]]
    for frequency_amplitude_instance in frequency_amplitude[0:16]:
      frequency_amplitude_colour += [[list(colour_map(0.5*(1 + frequency_amplitude_instance/lim)))]]*line_ratio + [[borderline]]
    frequency_amplitude_colour = np.array(frequency_amplitude_colour)

    plt.imshow(frequency_amplitude_colour)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["left"].set_visible(False)
    plt.gca().spines["bottom"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)
    plt.xticks([])
    plt.yticks([])
    plt.gca().set_aspect(1/((line_ratio + 1)))
    plt.gca().set_facecolor(borderline_true)
    if archive:
      archive.write_plot(f"", f"matrix_frequency")
    plt.draw()

    plt.figure()
    frequency_colour = []
    frequency_colour += [[borderline]*(time.size - 1)]
    for frequency_instance in frequency[0:16]:
      frequency_colour += [[list(colour_map(0.5*(1 + np.sin(math.tau*frequency_instance*time_instance)))) for time_instance in time[1:]]]*line_ratio + [[borderline]*(time.size - 1)]
    frequency_colour = np.array(frequency_colour)

    plt.imshow(frequency_colour)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["left"].set_visible(False)
    plt.gca().spines["bottom"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)
    plt.xticks([])
    plt.yticks([])
    plt.gca().set_aspect(3/((line_ratio + 1)))
    plt.gca().set_facecolor(borderline_true)
    if archive:
      archive.write_plot(f"", f"matrix_matrix")
    plt.draw()

    lim = np.max(np.abs(amplitude[1:]))

    def plot_coloured_trace(tm, am, title):
      plt.figure()
      time_fine = np.linspace(0.05e-3, 4.95e-3, 1000)
      amplitude_fine = np.interp(time_fine, tm, am)
      # for time_instance, amplitude_instance, colour_instance in zip(time[1:], amplitude[1:], cm.berlin(0.5*(1 + amplitude[1:]/lim))):
      for time_index in range(1, 1000):
        plt.plot([time_fine[time_index - 1], time_fine[time_index]], [amplitude_fine[time_index - 1], amplitude_fine[time_index]], "-", c = colour_map(0.5*(1 + amplitude_fine[time_index - 1 + np.argmax([abs(amplitude_fine[time_index - 1]), abs(amplitude_fine[time_index])])]/lim)), linewidth = 2)
      # plt.gca().set_facecolor(borderline_true)
      plt.gca().spines["right"].set_visible(False)
      plt.gca().spines["left"].set_visible(False)
      plt.gca().spines["bottom"].set_visible(False)
      plt.gca().spines["top"].set_visible(False)
      plt.xticks([])
      plt.yticks([])
      if archive:
        archive.write_plot(f"", f"matrix_time_plot_{title}")
      plt.draw()

      plt.figure()
      plt.imshow(am.reshape((99, 1)), cmap = colour_map, vmin = -lim, vmax = lim)
      plt.gca().spines["right"].set_visible(False)
      plt.gca().spines["left"].set_visible(False)
      plt.gca().spines["bottom"].set_visible(False)
      plt.gca().spines["top"].set_visible(False)
      plt.xticks([])
      plt.yticks([])
      plt.gca().set_aspect(1/5)
      plt.gca().set_facecolor(borderline_true)
      if archive:
        archive.write_plot(f"", f"matrix_time_{title}")
      plt.draw()

    plot_coloured_trace(time[1:], amplitude[1:], "ground")
    plot_coloured_trace(time[1:], amplitude_recover, "recovered")

  @staticmethod
  def make_unknown(archive):
    experiment_time = "20220517T111439" # One signal, By_aux, Compressive
    scaled = util.ScaledParameters.new_from_experiment_time(experiment_time)

    time_properties_reconstruction = test_signal.TimeProperties(scaled.time_step, scaled.time_step/np.ceil(scaled.time_step/1e-5), 1e-8, [scaled.time_step, scaled.time_end])

    acquired_signal_u1 = test_signal.AcquiredSignal.new_from_experiment_time(archive, "20220607T134852")
    acquired_time_u1, acquired_amplitude_u1 = acquired_signal_u1.subsample(scaled.time_step, archive, "Hz")
    # acquired_amplitude_u1 /= 0.4
    signal_reconstruction_u1 = test_signal.TestSignal(scaled.get_neural_pulses(), [], time_properties_reconstruction, signal_trace_time = acquired_time_u1, signal_trace_amplitude = acquired_amplitude_u1)

    experiment_results_u1 = arch.ExperimentResults.new_from_archive_time(archive, "20220607T134852")
    reconstruction_fst_u1 = recon.Reconstruction(signal_reconstruction_u1.time_properties)
    reconstruction_fst_u1.read_frequencies_from_experiment_results(experiment_results_u1, number_of_samples = 60, random_seed = util.Seeds.metroid)
    reconstruction_fst_u1.evaluate_fista_backtracking(
      expected_amplitude = scaled.amplitude,
      expected_frequency = scaled.frequency,
      expected_error_measurement = 1.8,
      norm_scale_factor_modifier = 1,
      is_fast = True,
      norm_scale_factor = 0.7488103857590022
    )

    acquired_signal_u2 = test_signal.AcquiredSignal.new_from_experiment_time(archive, "20220607T144242")
    acquired_time_u2, acquired_amplitude_u2 = acquired_signal_u2.subsample(scaled.time_step, archive, "Hz")
    # acquired_amplitude_u2 /= 0.4
    signal_reconstruction_u2 = test_signal.TestSignal(scaled.get_neural_pulses(), [], time_properties_reconstruction, signal_trace_time = acquired_time_u2, signal_trace_amplitude = acquired_amplitude_u2)

    experiment_results_u2 = arch.ExperimentResults.new_from_archive_time(archive, "20220607T144242")
    reconstruction_fst_u2 = recon.Reconstruction(signal_reconstruction_u2.time_properties)
    reconstruction_fst_u2.read_frequencies_from_experiment_results(experiment_results_u2, number_of_samples = 60, random_seed = util.Seeds.metroid)
    reconstruction_fst_u2.evaluate_fista_backtracking(
      expected_amplitude = scaled.amplitude,
      expected_frequency = scaled.frequency,
      expected_error_measurement = 1.8,
      norm_scale_factor_modifier = 1,
      is_fast = True,
      norm_scale_factor = 0.7488103857590022
      )

    recon.plot_reconstruction_unknown(archive, [reconstruction_fst_u1, reconstruction_fst_u2])

  @staticmethod
  def make_metrics(archive):
    recon.plot_reconstruction_number_of_samples_sweep_signal_comparison(
      archive,
      # ["20220725T162532", "20220725T164434"] # 360 Hz
      ["20220728T111144", "20220728T120757"] # 1 kHz
    )

class Aip2022:
  @staticmethod
  def make_abstract_plot(archive):
    experiment_time = "20220520T111524" # One signal, By_aux, Compressive
    scaled = util.ScaledParameters.new_from_experiment_time(experiment_time)

    time_properties_reconstruction = test_signal.TimeProperties(scaled.time_step, scaled.time_step/np.ceil(scaled.time_step/1e-5), 1e-8, [scaled.time_step, scaled.time_end])

    acquired_signal_1 = test_signal.AcquiredSignal.new_from_experiment_time(archive, "20220520T111524")
    acquired_time_1, acquired_amplitude_1 = acquired_signal_1.subsample(scaled.time_step, archive, "Hz")
    # acquired_amplitude_1 /= 0.4
    signal_reconstruction_1 = test_signal.TestSignal(scaled.get_neural_pulses(), [], time_properties_reconstruction, signal_trace_time = acquired_time_1, signal_trace_amplitude = acquired_amplitude_1)

    ramsey_results_1 = arch.RamseyResults.new_from_archive_time(archive, "20220523T141012")
    ramsey_results_1 = sim.ramsey.remove_dc(ramsey_results_1)

    experiment_results_1 = arch.ExperimentResults.new_from_archive_time(archive, "20220520T111524")

    reconstruction_fst_1 = recon.Reconstruction(signal_reconstruction_1.time_properties)
    reconstruction_fst_1.read_frequencies_from_experiment_results(experiment_results_1, number_of_samples = 60, random_seed = util.Seeds.metroid)
    reconstruction_fst_1.evaluate_fista_backtracking(
      expected_amplitude = scaled.amplitude,
      expected_frequency = scaled.frequency,
      expected_error_measurement = 1.8,
      norm_scale_factor_modifier = 1,
      is_fast = True,
      norm_scale_factor = 1.434744597683189#0.7488103857590022
    )

    units = "nT"
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

    plt.figure(figsize = [(6.4 + 0.4)*2, 4.8*2/3])
    plt.subplot(1, 3, 1)
    plt.plot(ramsey_results_1.time/1e-3, signal_reconstruction_1.amplitude*unit_factor, "k--", alpha = 0.5)
    plt.plot(ramsey_results_1.time/1e-3, ramsey_results_1.amplitude*unit_factor, "r-")
    plt.text(0.25, 600*unit_factor, f"(a)", size = 16)
    plt.xlabel("Time (ms)", size = 16)
    plt.ylabel(f"Standard Ramsey\nmeasurement ({units})", size = 16)
    plt.xlim([0, 5])
    plt.ylim([-900*unit_factor, 900*unit_factor])
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)

    plt.subplot(1, 3, 2)
    plt.plot(signal_reconstruction_1.frequency/1e3, signal_reconstruction_1.frequency_amplitude*unit_factor, "k--", alpha = 0.5)
    plt.plot(reconstruction_fst_1.frequency/1e3, reconstruction_fst_1.frequency_amplitude*unit_factor, "c.")
    plt.text(0.25, 26*unit_factor, f"(b)", size = 16)
    plt.xlabel("Frequency (kHz)", size = 16)
    plt.ylabel(f"Subsampled\nsine coefficients ({units})", size = 16)
    plt.xlim([0, 10])
    plt.ylim([-40*unit_factor, 40*unit_factor])
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)

    plt.subplot(1, 3, 3)
    plt.plot(reconstruction_fst_1.time_properties.time_coarse/1e-3, signal_reconstruction_1.amplitude*unit_factor, "k--", alpha = 0.5)
    plt.plot(reconstruction_fst_1.time_properties.time_coarse/1e-3, reconstruction_fst_1.amplitude*unit_factor, "c-")
    plt.text(0.25, 600*unit_factor, f"(c)", size = 16)
    plt.xlabel("Time (ms)", size = 16)
    plt.ylabel(f"Compressive\nmeasurement ({units})", size = 16)
    plt.xlim([0, 5])
    plt.ylim([-900*unit_factor, 900*unit_factor])
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)

    plt.subplots_adjust(wspace = 0.4, bottom = 0.2)

    if archive:
      archive.write_plot(f"", f"methods_comparison_abstract")
    plt.draw()