import math
import numpy as np
import scipy.fftpack as spf
import matplotlib.pyplot as plt
import matplotlib.colors as mplc
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
        # if single_shot != (abs(dressing_index -  trap_index) < 0.1):
        if single_shot or (abs(dressing_index -  trap_index) < 0.1):
          dressing     = dressings[dressing_index]
          frequency_rf = gradient_mid + gradient_span*(dressing_index/(number_of_traps - 1) - 0.5)
          field_x     += 2*dressing*math.cos(math.tau*frequency_rf*time)

      field[0] = math.tau*field_x
      field[1] = 0
      field[2] = math.tau*gradient_here
      field[3] = 0

    simulator = spinsim.Simulator(get_field_multi, spinsim.SpinQuantumNumber.ONE)

    gradient_mid  = 1000e3
    # gradient_span = 1200e3
    gradient_span = 100e3

    experiment_duration = 250e-6 #5e-3 #500e-6 #5e-3
    time_step           = 1e-6

    # frequencies = np.arange(1000, 10e3, 1000)/2.5
    frequencies = np.array([30e3]*2)
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

      # spins_multi_shot[trap_index, :] = np.sin(math.tau*68e3*result.time)
      # spins_multi_shot[trap_index, :] = np.cos(math.tau*frequencies[trap_index]*result.time)*np.sin(frequencies[trap_index]*np.cos(math.tau*gradient_span*result.time))
      spins_multi_shot[trap_index, :] = np.cos(math.tau*frequencies[trap_index]*(result.time + (1/(math.tau*gradient_span))*np.cos(math.tau*gradient_span*result.time)))
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

    label_steps_y = min(5, shape[0])
    plt.figure(figsize = figure_shape)
    plt.imshow(spins_multi_shot, cmap = cm.roma, vmin = -1, vmax = 1, aspect = shape[1]/shape[0]/aspect, interpolation = interpolation)
    xtick_decimate = np.arange(0, shape[1], int(np.round(shape[1]/5)), dtype = np.int)
    plt.gca().axes.xaxis.set_ticks(xtick_decimate)
    plt.gca().axes.xaxis.set_ticklabels([f"{1000*number:.2f}" for number in time[xtick_decimate]])
    plt.xlabel("Time (ms)")
    ytick_decimate = np.arange(0, shape[0], int(np.round(shape[0]/label_steps_y)), dtype = np.int)
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
    ytick_decimate = np.arange(0, shape[0], int(np.round(shape[0]/label_steps_y)), dtype = np.int)
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
    ytick_decimate = np.arange(0, shape[0], int(np.round(shape[0]/label_steps_y)), dtype = np.int)
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
    ytick_decimate = np.arange(0, shape[0], int(np.round(shape[0]/label_steps_y)), dtype = np.int)
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

  @staticmethod
  def visualise_dynamical_decoupling(archive = None):
    number_of_pulses_max = 256  
    time_sample_points   = 1024

    time              = np.arange(time_sample_points)/time_sample_points
    shape             = (number_of_pulses_max, time_sample_points)
    numbers_of_pulses = np.arange(1, number_of_pulses_max + 1)

    flip_udd   = np.ones(shape)
    flip_cpmg  = np.ones(shape)
    flip_spwm  = np.ones(shape)
    flip_spwmu = np.zeros(shape)
    flip_ideal = np.sin(20*math.tau*time)*np.ones(shape)

    for experiment_index, number_of_pulses in enumerate(numbers_of_pulses):
      pulse_indices   = np.arange(1, number_of_pulses + 1)
      pulse_fractions = (pulse_indices - 0.5)/number_of_pulses
      for pulse_fraction in pulse_fractions:
        flip_cpmg[experiment_index, time >= pulse_fraction] *= -1

      pulse_fractions = np.sin(math.pi*pulse_indices/(2*number_of_pulses + 2))**2
      for pulse_fraction in pulse_fractions:
        flip_udd[experiment_index, time >= pulse_fraction] *= -1

      sine = np.sin(20*math.tau*time)

      triangle = np.arcsin(np.sin(math.tau*(number_of_pulses + 1)*time/2))
      flip_spwm[experiment_index, sine < triangle]  = -1

      triangle = np.arcsin(np.sin(math.tau*(number_of_pulses)*time/4))
      flip_spwmu[experiment_index, np.logical_and(sine > 0, sine > triangle)] = 1
      flip_spwmu[experiment_index, np.logical_and(sine < 0, sine < triangle)] = -1
      # flip_spwmu[experiment_index, :] = triangle
      # flip_spwmu[experiment_index, :] *= np.sign(sine)

    flip_fft_udd   = np.abs(spf.fft(flip_udd))
    flip_fft_udd   /= np.max(flip_fft_udd)
    flip_fft_cpmg  = np.abs(spf.fft(flip_cpmg))
    flip_fft_cpmg  /= np.max(flip_fft_cpmg)
    flip_fft_spwm  = np.abs(spf.fft(flip_spwm))
    flip_fft_spwm  /= np.max(flip_fft_spwm)
    flip_fft_spwmu = np.abs(spf.fft(flip_spwmu))
    flip_fft_spwmu /= np.max(flip_fft_spwmu)
    flip_fft_ideal  = np.abs(spf.fft(flip_ideal))
    flip_fft_ideal  /= np.max(flip_fft_ideal)

    aspect        = 1
    interpolation = "bilinear"
    label_steps_y = min(number_of_pulses_max, 5)

    plt.figure()
    plt.imshow(flip_udd, cmap = cm.hawaii, vmin = -1, vmax = 1, aspect = shape[1]/shape[0]/aspect, interpolation = interpolation)
    xtick_decimate = np.arange(0, shape[1], int(np.round(shape[1]/5)), dtype = np.int)
    plt.gca().axes.xaxis.set_ticks(xtick_decimate)
    plt.gca().axes.xaxis.set_ticklabels([f"{number:.2f}" for number in time[xtick_decimate]])
    plt.xlabel("Time (au)")
    ytick_decimate = np.arange(0, shape[0], int(np.round(shape[0]/label_steps_y)), dtype = np.int)
    plt.gca().axes.yaxis.set_ticks(ytick_decimate)
    plt.gca().axes.yaxis.set_ticklabels([f"{number:.0f}" for number in numbers_of_pulses[ytick_decimate]])
    plt.ylabel("Number of pulses")
    plt.colorbar(label = "Sign of expected Fy")
    # plt.subplots_adjust(left = 0.07, right = 0.99, bottom = 0.1, top = 0.9)
    if archive:
      archive.write_plot("Uhrig dynamical decoupling", "mu_dd_udd")
    plt.draw()

    plt.figure()
    plt.imshow(flip_cpmg, cmap = cm.hawaii, vmin = -1, vmax = 1, aspect = shape[1]/shape[0]/aspect, interpolation = interpolation)
    xtick_decimate = np.arange(0, shape[1], int(np.round(shape[1]/5)), dtype = np.int)
    plt.gca().axes.xaxis.set_ticks(xtick_decimate)
    plt.gca().axes.xaxis.set_ticklabels([f"{number:.2f}" for number in time[xtick_decimate]])
    plt.xlabel("Time (au)")
    ytick_decimate = np.arange(0, shape[0], int(np.round(shape[0]/label_steps_y)), dtype = np.int)
    plt.gca().axes.yaxis.set_ticks(ytick_decimate)
    plt.gca().axes.yaxis.set_ticklabels([f"{number:.0f}" for number in numbers_of_pulses[ytick_decimate]])
    plt.ylabel("Number of pulses")
    plt.colorbar(label = "Sign of expected Fy")
    # plt.subplots_adjust(left = 0.07, right = 0.99, bottom = 0.1, top = 0.9)
    if archive:
      archive.write_plot("CPMG", "mu_dd_cpmg")
    plt.draw()

    plt.figure()
    plt.imshow(flip_spwm, cmap = cm.hawaii, vmin = -1, vmax = 1, aspect = shape[1]/shape[0]/aspect, interpolation = interpolation)
    xtick_decimate = np.arange(0, shape[1], int(np.round(shape[1]/5)), dtype = np.int)
    plt.gca().axes.xaxis.set_ticks(xtick_decimate)
    plt.gca().axes.xaxis.set_ticklabels([f"{number:.2f}" for number in time[xtick_decimate]])
    plt.xlabel("Time (au)")
    ytick_decimate = np.arange(0, shape[0], int(np.round(shape[0]/label_steps_y)), dtype = np.int)
    plt.gca().axes.yaxis.set_ticks(ytick_decimate)
    plt.gca().axes.yaxis.set_ticklabels([f"{number:.0f}" for number in numbers_of_pulses[ytick_decimate]])
    plt.ylabel("Number of pulses")
    plt.colorbar(label = "Sign of expected Fy")
    # plt.subplots_adjust(left = 0.07, right = 0.99, bottom = 0.1, top = 0.9)
    if archive:
      archive.write_plot("Sinusoidal pulse width modulation", "mu_dd_spwm")
    plt.draw()

    plt.figure()
    plt.imshow(flip_spwmu, cmap = cm.hawaii, vmin = -1, vmax = 1, aspect = shape[1]/shape[0]/aspect, interpolation = interpolation)
    xtick_decimate = np.arange(0, shape[1], int(np.round(shape[1]/5)), dtype = np.int)
    plt.gca().axes.xaxis.set_ticks(xtick_decimate)
    plt.gca().axes.xaxis.set_ticklabels([f"{number:.2f}" for number in time[xtick_decimate]])
    plt.xlabel("Time (au)")
    ytick_decimate = np.arange(0, shape[0], int(np.round(shape[0]/label_steps_y)), dtype = np.int)
    plt.gca().axes.yaxis.set_ticks(ytick_decimate)
    plt.gca().axes.yaxis.set_ticklabels([f"{number:.0f}" for number in numbers_of_pulses[ytick_decimate]])
    plt.ylabel("Number of pulses")
    plt.colorbar(label = "Sign of expected Fy")
    # plt.subplots_adjust(left = 0.07, right = 0.99, bottom = 0.1, top = 0.9)
    if archive:
      archive.write_plot("Sinusoidal pulse width modulation (unipolar)", "mu_dd_spwmu")
    plt.draw()

    plt.figure()
    plt.imshow(flip_ideal, cmap = cm.hawaii, vmin = -1, vmax = 1, aspect = shape[1]/shape[0]/aspect, interpolation = interpolation)
    xtick_decimate = np.arange(0, shape[1], int(np.round(shape[1]/5)), dtype = np.int)
    plt.gca().axes.xaxis.set_ticks(xtick_decimate)
    plt.gca().axes.xaxis.set_ticklabels([f"{number:.2f}" for number in time[xtick_decimate]])
    plt.xlabel("Time (au)")
    ytick_decimate = np.arange(0, shape[0], int(np.round(shape[0]/label_steps_y)), dtype = np.int)
    plt.gca().axes.yaxis.set_ticks(ytick_decimate)
    plt.gca().axes.yaxis.set_ticklabels([f"{number:.0f}" for number in numbers_of_pulses[ytick_decimate]])
    plt.ylabel("Number of pulses")
    plt.colorbar(label = "Sign of expected Fy")
    # plt.subplots_adjust(left = 0.07, right = 0.99, bottom = 0.1, top = 0.9)
    if archive:
      archive.write_plot("Ideal signal", "mu_dd_ideal")
    plt.draw()


    plt.figure()
    plt.imshow(flip_fft_udd, cmap = cm.tokyo, vmin = 0, vmax = 1, aspect = shape[1]/shape[0]/aspect, interpolation = interpolation)
    xtick_decimate = np.arange(0, shape[1], int(np.round(shape[1]/5)), dtype = np.int)
    plt.gca().axes.xaxis.set_ticks(xtick_decimate)
    plt.gca().axes.xaxis.set_ticklabels([f"{number:.2f}" for number in time[xtick_decimate]])
    plt.xlabel("Frequency (au)")
    ytick_decimate = np.arange(0, shape[0], int(np.round(shape[0]/label_steps_y)), dtype = np.int)
    plt.gca().axes.yaxis.set_ticks(ytick_decimate)
    plt.gca().axes.yaxis.set_ticklabels([f"{number:.0f}" for number in numbers_of_pulses[ytick_decimate]])
    plt.ylabel("Number of pulses")
    plt.colorbar(label = "Sign of expected Fy")
    # plt.subplots_adjust(left = 0.07, right = 0.99, bottom = 0.1, top = 0.9)
    if archive:
      archive.write_plot("Uhrig dynamical decoupling (FFT)", "mu_dd_fft_udd")
    plt.draw()

    plt.figure()
    plt.imshow(flip_fft_cpmg, cmap = cm.tokyo, vmin = 0, vmax = 1, aspect = shape[1]/shape[0]/aspect, interpolation = interpolation)
    xtick_decimate = np.arange(0, shape[1], int(np.round(shape[1]/5)), dtype = np.int)
    plt.gca().axes.xaxis.set_ticks(xtick_decimate)
    plt.gca().axes.xaxis.set_ticklabels([f"{number:.2f}" for number in time[xtick_decimate]])
    plt.xlabel("Frequency (au)")
    ytick_decimate = np.arange(0, shape[0], int(np.round(shape[0]/label_steps_y)), dtype = np.int)
    plt.gca().axes.yaxis.set_ticks(ytick_decimate)
    plt.gca().axes.yaxis.set_ticklabels([f"{number:.0f}" for number in numbers_of_pulses[ytick_decimate]])
    plt.ylabel("Number of pulses")
    plt.colorbar(label = "Sign of expected Fy")
    # plt.subplots_adjust(left = 0.07, right = 0.99, bottom = 0.1, top = 0.9)
    if archive:
      archive.write_plot("CPMG (FFT)", "mu_dd_fft_cpmg")
    plt.draw()

    plt.figure()
    plt.imshow(flip_fft_spwm, cmap = cm.tokyo, vmin = 0, vmax = 1, aspect = shape[1]/shape[0]/aspect, interpolation = interpolation)
    xtick_decimate = np.arange(0, shape[1], int(np.round(shape[1]/5)), dtype = np.int)
    plt.gca().axes.xaxis.set_ticks(xtick_decimate)
    plt.gca().axes.xaxis.set_ticklabels([f"{number:.2f}" for number in time[xtick_decimate]])
    plt.xlabel("Frequency (au)")
    ytick_decimate = np.arange(0, shape[0], int(np.round(shape[0]/label_steps_y)), dtype = np.int)
    plt.gca().axes.yaxis.set_ticks(ytick_decimate)
    plt.gca().axes.yaxis.set_ticklabels([f"{number:.0f}" for number in numbers_of_pulses[ytick_decimate]])
    plt.ylabel("Number of pulses")
    plt.colorbar(label = "Sign of expected Fy")
    # plt.subplots_adjust(left = 0.07, right = 0.99, bottom = 0.1, top = 0.9)
    if archive:
      archive.write_plot("Sinusoidal pulse width modulation (FFT)", "mu_dd_fft_spwm")
    plt.draw()

    plt.figure()
    plt.imshow(flip_fft_spwmu, cmap = cm.tokyo, vmin = 0, vmax = 1, aspect = shape[1]/shape[0]/aspect, interpolation = interpolation)
    xtick_decimate = np.arange(0, shape[1], int(np.round(shape[1]/5)), dtype = np.int)
    plt.gca().axes.xaxis.set_ticks(xtick_decimate)
    plt.gca().axes.xaxis.set_ticklabels([f"{number:.2f}" for number in time[xtick_decimate]])
    plt.xlabel("Frequency (au)")
    ytick_decimate = np.arange(0, shape[0], int(np.round(shape[0]/label_steps_y)), dtype = np.int)
    plt.gca().axes.yaxis.set_ticks(ytick_decimate)
    plt.gca().axes.yaxis.set_ticklabels([f"{number:.0f}" for number in numbers_of_pulses[ytick_decimate]])
    plt.ylabel("Number of pulses")
    plt.colorbar(label = "Sign of expected Fy")
    # plt.subplots_adjust(left = 0.07, right = 0.99, bottom = 0.1, top = 0.9)
    if archive:
      archive.write_plot("Sinusoidal pulse width modulation (unipolar) (FFT)", "mu_dd_fft_spwmu")
    plt.draw()

    plt.figure()
    plt.imshow(flip_fft_ideal, cmap = cm.tokyo, vmin = 0, vmax = 1, aspect = shape[1]/shape[0]/aspect, interpolation = interpolation)
    xtick_decimate = np.arange(0, shape[1], int(np.round(shape[1]/5)), dtype = np.int)
    plt.gca().axes.xaxis.set_ticks(xtick_decimate)
    plt.gca().axes.xaxis.set_ticklabels([f"{number:.2f}" for number in time[xtick_decimate]])
    plt.xlabel("Frequency (au)")
    ytick_decimate = np.arange(0, shape[0], int(np.round(shape[0]/label_steps_y)), dtype = np.int)
    plt.gca().axes.yaxis.set_ticks(ytick_decimate)
    plt.gca().axes.yaxis.set_ticklabels([f"{number:.0f}" for number in numbers_of_pulses[ytick_decimate]])
    plt.ylabel("Number of pulses")
    plt.colorbar(label = "Sign of expected Fy")
    # plt.subplots_adjust(left = 0.07, right = 0.99, bottom = 0.1, top = 0.9)
    if archive:
      archive.write_plot("Ideal signal (FFT)", "mu_dd_fft_ideal")
    plt.draw()

  @staticmethod
  def difference_addressing(archive = None):
    number_of_traps = 2
    experiment_duration = 5e-3
    time_step = 1e-5
    shape = [number_of_traps, int(math.floor(experiment_duration/time_step))]
    trap_separation = 20e3
    pulse_amplitude = (math.sqrt(3)/2)*trap_separation
    # pulse_amplitude = (math.sqrt(3/5)/2)*trap_separation
    # pulse_amplitude = trap_separation
    # pulse_duration = 4*math.asin(math.pi/(4*math.sqrt(3)))/(math.tau*trap_separation)
    # pulse_duration = 1/(2*pulse_amplitude)
    pulse_duration = math.asin(math.pi*trap_separation/(8*pulse_amplitude))*4/(math.tau*trap_separation)
    # pulse_duration = 2*math.sqrt(3)*math.asin(math.pi*math.sqrt(3)/16)/(math.tau*trap_separation)
    pulse_time = experiment_duration/16
    gradient_centre = 650e3
    gradient = gradient_centre + (np.arange(number_of_traps) - (number_of_traps - 1)/2)*trap_separation

    flips = 0*np.fmod(np.arange(number_of_traps), 2)
    # flips[5] = 1
    # flips[12] = 1
    # flips[int(3*number_of_traps/4):] = 1
    # flips[15:] = 0
    flips_integrated = flips.copy()
    for flip_index in range(flips.size - 1):
      flips_integrated[flip_index + 1] = flips_integrated[flip_index] == flips[flip_index + 1]
    
    # flips_integrated = np.fmod(np.cumsum(flips), 2)
    C.print(flips_integrated)

    def get_field_difference(time, parameters, field):
      trap_index = parameters[0]
      flips_integrated = parameters[1:]

      field[1] = 0
      field[3] = 0
      
      bias = gradient_centre + (trap_index - (number_of_traps - 1)/2)*trap_separation
      field[2] = math.tau*bias

      field[0] = 0
      for pulse_index in range(number_of_traps + 1):
        radio_frequency = gradient_centre + (pulse_index - (number_of_traps)/2)*trap_separation
        radio_frequency_low = radio_frequency - (number_of_traps)*trap_separation
        # radio_phase = trap_separation*pulse_duration/2
        for pulse_time_index in range(1, 16):
          if time > pulse_time_index*pulse_time - pulse_duration/2 and time < pulse_time_index*pulse_time + pulse_duration/2:
            # field[0] += ((-1)**pulse_index)*math.tau*2*pulse_amplitude*math.cos(math.tau*radio_frequency*(time - pulse_time))
            # field[0] += math.tau*2*pulse_amplitude*math.cos(math.tau*radio_frequency*(time - pulse_time))
            # field[0] += ((-1)**pulse_index)*math.tau*2*pulse_amplitude*math.cos(math.tau*radio_frequency*time)
            # field[0] += math.tau*2*pulse_amplitude*math.cos(math.tau*(radio_frequency*(time - (pulse_time - pulse_duration/2)) + pulse_index*trap_separation*pulse_duration/2))
            # field[0] += math.tau*2*pulse_amplitude*math.cos(math.tau*(radio_frequency*time + pulse_index*(trap_separation*pulse_duration/2 + 1/2)))
            # field[0] += math.tau*2*pulse_amplitude*math.cos(math.tau*(radio_frequency*(time - (pulse_time - pulse_duration/2))))
            # field[0] += math.tau*2*pulse_amplitude*math.cos(math.tau*(radio_frequency*(time - (pulse_time - pulse_duration/2)) + pulse_index*(1/2)))
            flip_integrated = 1
            if pulse_index > 0:
              flip_integrated = flips_integrated[pulse_index - 1]
            field[0] += ((-1)**(flip_integrated + pulse_time_index))*math.tau*2*pulse_amplitude*math.cos(math.tau*(radio_frequency*(time - (pulse_time - pulse_duration/2)) - pulse_time*trap_separation*pulse_index*(pulse_time_index - 1)))
            # field[0] += ((-1)**pulse_index)*math.tau*2*pulse_amplitude*math.cos(math.tau*(radio_frequency_low*(time - (pulse_time - pulse_duration/2)) - (pulse_index - number_of_traps)*pulse_time*trap_separation*(pulse_time_index - 1)))
            # if pulse_index > 0:
            #   field[0] += -((-1)**flip_integrated)*math.tau*2*pulse_amplitude*math.cos(math.tau*(radio_frequency_low*(time - (pulse_time - pulse_duration/2)) - pulse_index*pulse_time*trap_separation*(pulse_time_index - 1)))
            # field[0] += ((-1)**flip_integrated)*math.tau*2*pulse_amplitude*math.cos(math.tau*(radio_frequency*(time - pulse_time) + pulse_index*trap_separation*(pulse_time*(pulse_time_index - 1) + pulse_duration/2)))
        # for pulse_index in range(number_of_traps):
        #   radio_frequency = gradient_centre + (pulse_index - (number_of_traps - 1)/2)*trap_separation
        #   if time > pulse_time - pulse_duration/2 and time < pulse_time + pulse_duration/2:
        #     field[0] += math.tau*2*pulse_amplitude*math.cos(math.tau*radio_frequency*(time - pulse_time))

    simulator = spinsim.Simulator(get_field_difference, spinsim.SpinQuantumNumber.ONE)
    spin_map = np.empty(shape)
    parameters = [0] + [flip_integrated for flip_integrated in flips_integrated]
    for trap_index in range(number_of_traps):
      parameters[0] = trap_index
      result = simulator.evaluate(0, experiment_duration, time_step/50, time_step, spinsim.SpinQuantumNumber.ONE.minus_z, parameters)
      spin_map[trap_index, :] = result.spin[:, 2]

    time = result.time

    aspect = 1
    label_steps_y = min(5, shape[0])
    interpolation = "none"

    plt.figure()
    plt.imshow(spin_map, cmap = cm.roma, vmin = -1, vmax = 1, aspect = shape[1]/shape[0]/aspect, interpolation = interpolation)
    xtick_decimate = np.arange(0, shape[1], int(np.round(shape[1]/5)), dtype = np.int)
    plt.gca().axes.xaxis.set_ticks(xtick_decimate)
    plt.gca().axes.xaxis.set_ticklabels([f"{1000*number:.2f}" for number in time[xtick_decimate]])
    plt.xlabel("Time (ms)")
    ytick_decimate = np.arange(0, shape[0], int(np.round(shape[0]/label_steps_y)), dtype = np.int)
    plt.gca().axes.yaxis.set_ticks(ytick_decimate)
    plt.gca().axes.yaxis.set_ticklabels([f"{number/1000:.0f}" for number in gradient[ytick_decimate]])
    plt.ylabel("Gradient (kHz)")
    plt.colorbar(label = "Expected spin z projection (hbar)")
    if archive:
      archive.write_plot("Difference addressing", "mu_da_time")
    plt.draw()

  @staticmethod
  def continuous_multi(archive = None):
    def get_field(time, parameters, field):
      field[0] = 0
      field[1] = 0
      field[3] = 0

      position = parameters[0]
      magnetic_centre = parameters[1]
      magnetic_gradient = parameters[2]

      sign = math.cos(math.tau*10e3*time)
      # sign = -1
      # time_flip = time
      # while time_flip > 0:
      #   time_flip -= 75e-6
      #   sign *= -1
      magnetic_bias = magnetic_centre + sign*magnetic_gradient*position
      field[2] = math.tau*magnetic_bias
      
      pulse_amplitude = parameters[3]
      experiment_duration = parameters[4]
      pulse_duration = 1/(4*pulse_amplitude)

      pulse_time_init = experiment_duration/9
      pulse_time = pulse_time_init
      pulse_time_init -= pulse_duration
      if time > pulse_time - pulse_duration and time < pulse_time:
        field[0] = 2*math.tau*pulse_amplitude*math.cos(math.tau*magnetic_centre*(time - pulse_time_init))

      pulse_time += 200e-6
      if time > pulse_time and time < pulse_time + 2*pulse_duration:
        field[0] = -2*math.tau*pulse_amplitude*math.cos(math.tau*magnetic_centre*(time - pulse_time_init))

      pulse_time += 200e-6 + 2*pulse_duration
      if time > pulse_time and time < pulse_time + 2*pulse_duration:
        field[0] = 2*math.tau*pulse_amplitude*math.cos(math.tau*magnetic_centre*(time - pulse_time_init))

      pulse_time += 200e-6 + 2*pulse_duration
      if time > pulse_time and time < pulse_time + 2*pulse_duration:
        field[0] = -2*math.tau*pulse_amplitude*math.cos(math.tau*magnetic_centre*(time - pulse_time_init))

      pulse_time += 200e-6 + 2*pulse_duration
      if time > pulse_time and time < pulse_time + 2*pulse_duration:
        field[0] = 2*math.tau*pulse_amplitude*math.cos(math.tau*magnetic_centre*(time - pulse_time_init))

      pulse_time += 200e-6 + 2*pulse_duration
      if time > pulse_time and time < pulse_time + pulse_duration:
        field[0] = -2*math.tau*pulse_amplitude*math.cos(math.tau*magnetic_centre*(time - pulse_time_init))

    experiment_separation = 1e-3
    position_step = 1e-6
    positions = np.arange(-experiment_separation/2, experiment_separation/2, position_step)

    experiment_duration = 5e-3
    time_step = 5e-7
    times = np.arange(0, experiment_duration, time_step)

    atom_trace_shape = (positions.size, times.size, 3)
    atom_trace = np.zeros(atom_trace_shape)

    magnetic_centre = 750e3
    magnetic_gradient = 100e3/experiment_separation

    pulse_amplitude = 1e3

    number_of_traps = 9
    trap_radius = 10e-6
    trap_separation = 100e-6
    trap_positions = (np.arange(number_of_traps) - (number_of_traps - 1)/2)*trap_separation
    trap_mask = np.zeros_like(positions)
    
    for trap_position in trap_positions:
      trap_mask += np.exp(-0.5*((positions - trap_position)/trap_radius)**2)
    trap_mask_min = 0.01

    # plt.figure()
    # plt.plot(positions/1e-3, trap_mask, "k-")
    # plt.xlabel("Position (mm)")
    # plt.ylabel("Trap depth (au)")
    # plt.draw()
    def apply_colour_map(spin):
      return np.array(
        [
          np.cos(math.tau*(spin - 1)/8),
          np.cos(math.tau*(spin    )/8),
          np.cos(math.tau*(spin + 1)/8)
        ]
      ).T
    simulator = spinsim.Simulator(get_field, spinsim.SpinQuantumNumber.ONE)
    parameters = np.array([0, magnetic_centre, magnetic_gradient, pulse_amplitude, experiment_duration])
    C.starting("Simulations")
    C.print(f"|{'Index':7s}|{'Progress (%)':12s}|{'Position (um)':13s}|")
    for position_index in range(positions.size):
      if True: #trap_mask[position_index] > trap_mask_min:
        # spin = 2*times/experiment_duration - 1
        parameters[0] = positions[position_index]
        results = simulator.evaluate(0, experiment_duration, time_step/3, time_step, spinsim.SpinQuantumNumber.ONE.minus_z, parameters)
        spin = results.spin[:, 2]
      else:
        spin = np.zeros_like(times)
      # atom_trace[position_index, :, 2] = np.cos(math.tau*(spin + 1)/4)
      # atom_trace[position_index, :, 0] = np.cos(math.tau*(spin - 1)/4)
      atom_trace[position_index, :, :] = apply_colour_map(spin)
      C.print(f"|{position_index:7d}|{100*(position_index + 1)/positions.size:12.4f}|{positions[position_index]/1e-6:13.4f}|", end = "\r")
    C.print("")
    C.finished("Simulations")
    

    # atom_trace[:, :, 1] = 1
    # atom_trace *= np.reshape(trap_mask, (atom_trace_shape[0], 1, 1))
    colourbar_x = np.linspace(1, -1, 49)
    colourbar = np.empty((colourbar_x.size, 1, 3))
    colourbar[:, 0, :] = apply_colour_map(colourbar_x)

    aspect = 1
    label_steps_y = min(5, atom_trace_shape[0])
    interpolation = "bilinear"

    plt.figure(figsize = [8, 4.8])
    plt.subplot(1, 2, 1)
    plt.imshow(atom_trace, aspect = atom_trace_shape[1]/atom_trace_shape[0]/aspect, interpolation = interpolation)
    xtick_decimate = np.arange(0, atom_trace_shape[1], int(np.round(atom_trace_shape[1]/5)), dtype = np.int)
    plt.gca().axes.xaxis.set_ticks(xtick_decimate)
    plt.gca().axes.xaxis.set_ticklabels([f"{number/1e-3:.2f}" for number in times[xtick_decimate]])
    plt.xlabel("Time (ms)")
    ytick_decimate = np.arange(0, atom_trace_shape[0], int(np.round(atom_trace_shape[0]/label_steps_y)), dtype = np.int)
    plt.gca().axes.yaxis.set_ticks(ytick_decimate)
    plt.gca().axes.yaxis.set_ticklabels([f"{number/1e-6:.0f}" for number in positions[ytick_decimate]])
    plt.ylabel("Position (um)")

    plt.subplot(1, 2, 2)
    plt.imshow(colourbar, aspect = 1/5, interpolation = "bilinear")
    plt.gca().axes.xaxis.set_ticks([])
    plt.gca().axes.xaxis.set_ticklabels([])
    plt.xlabel("Colourbar")
    ctick_decimate = np.arange(0, colourbar_x.size, int(np.round(colourbar_x.size/6)), dtype = np.int)
    plt.gca().axes.yaxis.set_ticks(ctick_decimate)
    plt.gca().axes.yaxis.set_ticklabels([f"{number:.2f}" for number in colourbar_x[ctick_decimate]])
    plt.ylabel("Expected spin projection (hbar)")

    plt.tight_layout()

    if archive:
      archive.write_plot("Continuous multi-trap", "mu_co_trace")
    plt.draw()

    atom_trace *= np.reshape(trap_mask, (atom_trace_shape[0], 1, 1))

    plt.figure(figsize = [8, 4.8])
    plt.subplot(1, 2, 1)
    plt.imshow(atom_trace, aspect = atom_trace_shape[1]/atom_trace_shape[0]/aspect, interpolation = interpolation)
    xtick_decimate = np.arange(0, atom_trace_shape[1], int(np.round(atom_trace_shape[1]/5)), dtype = np.int)
    plt.gca().axes.xaxis.set_ticks(xtick_decimate)
    plt.gca().axes.xaxis.set_ticklabels([f"{number/1e-3:.2f}" for number in times[xtick_decimate]])
    plt.xlabel("Time (ms)")
    ytick_decimate = np.arange(0, atom_trace_shape[0], int(np.round(atom_trace_shape[0]/label_steps_y)), dtype = np.int)
    plt.gca().axes.yaxis.set_ticks(ytick_decimate)
    plt.gca().axes.yaxis.set_ticklabels([f"{number/1e-6:.0f}" for number in positions[ytick_decimate]])
    plt.ylabel("Position (um)")

    plt.subplot(1, 2, 2)
    plt.imshow(colourbar, aspect = 1/5, interpolation = "bilinear")
    plt.gca().axes.xaxis.set_ticks([])
    plt.gca().axes.xaxis.set_ticklabels([])
    plt.xlabel("Colourbar")
    ctick_decimate = np.arange(0, colourbar_x.size, int(np.round(colourbar_x.size/6)), dtype = np.int)
    plt.gca().axes.yaxis.set_ticks(ctick_decimate)
    plt.gca().axes.yaxis.set_ticklabels([f"{number:.2f}" for number in colourbar_x[ctick_decimate]])
    plt.ylabel("Expected spin projection (hbar)")

    plt.tight_layout()
    
    if archive:
      archive.write_plot("Continuous multi-trap", "mu_co_trace_mask")
    plt.draw()

