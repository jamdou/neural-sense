import math
import numpy as np
import scipy.fftpack as spf
import matplotlib.pyplot as plt
import spinsim
from cmcrameri import cm

import archive as arch
from util import PrettyTritty as C

class MultiAnalysis:
  @staticmethod
  def avoiding_dressing(archive = None):
    def get_field_multi(time, parameters, field):
      trap_index      = parameters[0]
      number_of_traps = parameters[1]
      gradient_mid    = parameters[2]
      gradient_span   = parameters[3]
      single_shot     = abs(parameters[4] - 1) < 0.1
      dressings       = parameters[5:]

      gradient_here = gradient_mid + gradient_span*(trap_index/(number_of_traps - 1) - 0.5)

      field_x = 0
      for dressing_index in range(number_of_traps):
        if single_shot or abs(dressing_index -  trap_index) < 0.1:
          dressing     = dressings[dressing_index]
          frequency_rf = gradient_mid + gradient_span*(dressing_index/(number_of_traps - 1) - 0.5)
          field_x     += 2*math.tau*dressing*math.cos(math.tau*frequency_rf*time)

      field[0] = math.tau*field_x
      field[1] = 0
      field[2] = math.tau*gradient_here
      field[3] = 0

    simulator = spinsim.Simulator(get_field_multi, spinsim.SpinQuantumNumber.ONE)

    gradient_mid  = 1000e3
    gradient_span = 1200e3

    experiment_duration = 250e-6 #5e-3 #500e-6 #5e-3
    time_step           = 1e-6

    frequencies = np.arange(1000, 10e3, 1000)/5
    parameters  = [0, frequencies.size, gradient_mid, gradient_span, 1] + [frequency for frequency in frequencies]

    shape = (frequencies.size, int(experiment_duration/time_step))
    spins_multi_shot  = np.empty(shape)
    spins_single_shot = np.empty(shape)

    for trap_index in range(shape[0]):
      parameters[0] = trap_index

      parameters[4]                   = 0
      result                          = simulator.evaluate(0, experiment_duration, 1e-7, 1e-6, spinsim.SpinQuantumNumber.ONE.minus_z, parameters)
      spins_multi_shot[trap_index, :] = result.spin[:, 2]

      parameters[4]                    = 1
      result                           = simulator.evaluate(0, experiment_duration, 1e-7, 1e-6, spinsim.SpinQuantumNumber.ONE.minus_z, parameters)
      spins_single_shot[trap_index, :] = result.spin[:, 2]
    time = result.time
    # frequency_dct = 1/(2*(time[1] - time[0]))*np.arange(time.size)
    frequency_dct = spf.fftfreq(time.size, d = time[1] - time[0])

    spins_bleed = spins_single_shot - spins_multi_shot

    gradient = gradient_mid + gradient_span*(np.arange(shape[0])/(shape[0] - 1) - 0.5)

    spins_multi_shot_dct  = abs(spf.fft(spins_multi_shot))
    spins_multi_shot_dct  /= np.max(np.abs(spins_multi_shot_dct))/2
    spins_single_shot_dct = abs(spf.fft(spins_single_shot))
    spins_single_shot_dct /= np.max(np.abs(spins_single_shot_dct))/2
    spins_bleed_dct = abs(spf.fft(spins_bleed))
    spins_bleed_dct /= np.max(np.abs(spins_bleed_dct))

    aspect = 3
    figure_shape = [4.8*(aspect - 0.5), 4.8]
    interpolation = "none"

    plt.figure(figsize = figure_shape)
    plt.imshow(spins_multi_shot, cmap = cm.roma, vmin = -1, vmax = 1, aspect = shape[1]/shape[0]/aspect, interpolation = interpolation)
    xtick_decimate = np.arange(0, shape[1], int(np.round(shape[1]/5)), dtype = np.int)
    plt.gca().axes.xaxis.set_ticks(xtick_decimate)
    plt.gca().axes.xaxis.set_ticklabels([f"{1000*number:.2f}" for number in time[xtick_decimate]])
    plt.xlabel("Time (ms)")
    ytick_decimate = np.arange(0, shape[0], int(np.round(shape[0]/5)), dtype = np.int)
    plt.gca().axes.yaxis.set_ticks(ytick_decimate)
    plt.gca().axes.yaxis.set_ticklabels([f"{number/1000:.0f}" for number in gradient[ytick_decimate]])
    plt.ylabel("Gradient (kHz)")
    plt.colorbar(label = "Expected spin z projection (hbar)")
    plt.subplots_adjust(left = 0.07, right = 0.99, bottom = 0.1, top = 0.9)
    if archive:
      archive.write_plot("Multi shot", "mu_dr_bleed_multi_shot")
    plt.draw()

    plt.figure(figsize = figure_shape)
    plt.imshow(spins_multi_shot_dct, cmap = cm.tokyo, vmin = 0, vmax = 1, aspect = shape[1]/shape[0]/aspect, interpolation = interpolation)
    xtick_decimate = np.arange(0, shape[1], int(np.round(shape[1]/15)), dtype = np.int)
    plt.gca().axes.xaxis.set_ticks(xtick_decimate)
    plt.gca().axes.xaxis.set_ticklabels([f"{number/1000:.2f}" for number in frequency_dct[xtick_decimate]])
    plt.xlabel("Frequency (kHz)")
    ytick_decimate = np.arange(0, shape[0], int(np.round(shape[0]/5)), dtype = np.int)
    plt.gca().axes.yaxis.set_ticks(ytick_decimate)
    plt.gca().axes.yaxis.set_ticklabels([f"{number/1000:.0f}" for number in gradient[ytick_decimate]])
    plt.ylabel("Gradient (kHz)")
    plt.colorbar(label = "abs FFT coefficient (A.U.)")
    plt.subplots_adjust(left = 0.07, right = 0.99, bottom = 0.1, top = 0.9)
    if archive:
      archive.write_plot("Multi shot FFT", "mu_dr_bleed_multi_shot_fft")
    plt.draw()

    plt.figure(figsize = figure_shape)
    plt.imshow(spins_single_shot, cmap = cm.roma, vmin = -1, vmax = 1, aspect = shape[1]/shape[0]/aspect, interpolation = interpolation)
    xtick_decimate = np.arange(0, shape[1], int(np.round(shape[1]/5)), dtype = np.int)
    plt.gca().axes.xaxis.set_ticks(xtick_decimate)
    plt.gca().axes.xaxis.set_ticklabels([f"{1000*number:.2f}" for number in time[xtick_decimate]])
    plt.xlabel("Time (ms)")
    ytick_decimate = np.arange(0, shape[0], int(np.round(shape[0]/5)), dtype = np.int)
    plt.gca().axes.yaxis.set_ticks(ytick_decimate)
    plt.gca().axes.yaxis.set_ticklabels([f"{number/1000:.0f}" for number in gradient[ytick_decimate]])
    plt.ylabel("Gradient (kHz)")
    plt.colorbar(label = "Expected spin z projection (hbar)")
    plt.subplots_adjust(left = 0.07, right = 0.99, bottom = 0.1, top = 0.9)
    if archive:
      archive.write_plot("Single shot", "mu_dr_bleed_single_shot")
    plt.draw()

    plt.figure(figsize = figure_shape)
    plt.imshow(spins_single_shot_dct, cmap = cm.tokyo, vmin = 0, vmax = 1, aspect = shape[1]/shape[0]/aspect, interpolation = interpolation)
    xtick_decimate = np.arange(0, shape[1], int(np.round(shape[1]/15)), dtype = np.int)
    plt.gca().axes.xaxis.set_ticks(xtick_decimate)
    plt.gca().axes.xaxis.set_ticklabels([f"{number/1000:.2f}" for number in frequency_dct[xtick_decimate]])
    plt.xlabel("Frequency (kHz)")
    ytick_decimate = np.arange(0, shape[0], int(np.round(shape[0]/5)), dtype = np.int)
    plt.gca().axes.yaxis.set_ticks(ytick_decimate)
    plt.gca().axes.yaxis.set_ticklabels([f"{number/1000:.0f}" for number in gradient[ytick_decimate]])
    plt.ylabel("Gradient (kHz)")
    plt.colorbar(label = "abs FFT coefficient (A.U.)")
    plt.subplots_adjust(left = 0.07, right = 0.99, bottom = 0.1, top = 0.9)
    if archive:
      archive.write_plot("Single shot FFT", "mu_dr_bleed_single_shot_fft")
    plt.draw()

    # plt.figure(figsize = figure_shape)
    # plt.imshow(spins_bleed, cmap = cm.roma, vmin = -1, vmax = 1, aspect = shape[1]/shape[0]/aspect, interpolation = interpolation)
    # xtick_decimate = np.arange(0, shape[1], int(np.round(shape[1]/5)), dtype = np.int)
    # plt.gca().axes.xaxis.set_ticks(xtick_decimate)
    # plt.gca().axes.xaxis.set_ticklabels([f"{1000*number:.2f}" for number in time[xtick_decimate]])
    # plt.xlabel("Time (ms)")
    # ytick_decimate = np.arange(0, shape[0], int(np.round(shape[0]/5)), dtype = np.int)
    # plt.gca().axes.yaxis.set_ticks(ytick_decimate)
    # plt.gca().axes.yaxis.set_ticklabels([f"{number/1000:.0f}" for number in gradient[ytick_decimate]])
    # plt.ylabel("Gradient (kHz)")
    # plt.colorbar(label = "Expected spin z projection (hbar)")
    # plt.subplots_adjust(left = 0.07, right = 0.99, bottom = 0.1, top = 0.9)
    # if archive:
    #   archive.write_plot("Single - multi", "mu_dr_bleed_difference")
    # plt.draw()

    # plt.figure(figsize = figure_shape)
    # plt.imshow(spins_bleed_dct, cmap = cm.roma, vmin = -1, vmax = 1, aspect = shape[1]/shape[0]/aspect, interpolation = interpolation)
    # xtick_decimate = np.arange(0, shape[1], int(np.round(shape[1]/5)), dtype = np.int)
    # plt.gca().axes.xaxis.set_ticks(xtick_decimate)
    # plt.gca().axes.xaxis.set_ticklabels([f"{number/1000000:.2f}" for number in frequency_dct[xtick_decimate]])
    # plt.xlabel("Frequency (MHz)")
    # ytick_decimate = np.arange(0, shape[0], int(np.round(shape[0]/5)), dtype = np.int)
    # plt.gca().axes.yaxis.set_ticks(ytick_decimate)
    # plt.gca().axes.yaxis.set_ticklabels([f"{number/1000:.0f}" for number in gradient[ytick_decimate]])
    # plt.ylabel("Gradient (kHz)")
    # plt.colorbar(label = "DCT coefficient (A.U.)")
    # plt.subplots_adjust(left = 0.07, right = 0.99, bottom = 0.1, top = 0.9)
    # if archive:
    #   archive.write_plot("Single - multi DCT", "mu_dr_bleed_difference_dct")
    # plt.draw()