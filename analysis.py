import numpy as np
import math
import matplotlib.pyplot as plt
from numpy.lib.function_base import gradient

import archive as arch
from util import C

def find_neural_signal_size(experiment_results:arch.ExperimentResults, archive:arch.Archive = None):
    print(f"{C.y}Starting fit...{C.d}")
    scaled_frequency = 1002.6*5
    scaled_density = 1/25
    scaled_samples = 10
    scaled_amplitude = 8000
    scaled_sweep = [scaled_frequency/5, 14000]
    scaled_time_step = 1/(scaled_frequency*scaled_samples)
    scaled_time_end = 1/(scaled_frequency*scaled_density)
    scaled_pulse_time = 0.2333333*scaled_time_end
    scaled_frequency_step = scaled_density*scaled_frequency/2
    
    frequency = np.arange(scaled_sweep[0], scaled_sweep[1], scaled_frequency_step)
    frequency = np.copy(experiment_results.frequency)
    time = np.arange(0, scaled_time_end, scaled_time_step)
    
    amplitude_approximated_normalised = np.zeros_like(time)
    for time_index in range(time.size):
        if time[time_index] >= scaled_pulse_time and time[time_index] < scaled_pulse_time + 1/scaled_frequency:
            amplitude_approximated_normalised[time_index] = np.sin(math.tau*scaled_frequency*(time[time_index] - scaled_pulse_time))
    frequency_mesh, time_mesh = np.meshgrid(frequency, time)
    frequency_mesh, amplitude_approximated_normalised_mesh = np.meshgrid(frequency, amplitude_approximated_normalised)
    frequency_amplitude_approximated_normalised = np.sum(np.sin(math.tau*frequency_mesh*time_mesh)*amplitude_approximated_normalised_mesh, axis = 0)*scaled_time_step/scaled_time_end

    frequency_amplitude_measured = np.copy(experiment_results.frequency_amplitude)

    scaled_amplitude = 100
    decent_step_size = 1e-6
    mean_squared_errors = []
    for epoch in range(1000000):
        frequency_amplitude_predict = (1/(math.tau*scaled_time_end))*np.sin(math.tau*scaled_time_end*scaled_amplitude*frequency_amplitude_approximated_normalised)
        frequency_amplitude_residual = frequency_amplitude_predict - frequency_amplitude_measured
        mean_squared_error = np.mean(frequency_amplitude_residual**2)
        frequency_amplitude_gradient = np.sum(2*frequency_amplitude_residual*np.cos(math.tau*scaled_time_end*scaled_amplitude*frequency_amplitude_approximated_normalised)*scaled_amplitude*frequency_amplitude_approximated_normalised)
        scaled_amplitude -= decent_step_size*frequency_amplitude_gradient
        mean_squared_errors += [mean_squared_error]
    print(f"\tAmp: {scaled_amplitude}, Squared error: {mean_squared_error}, RMS Error: {np.sqrt(mean_squared_error)}")
    print(f"{C.g}Done!{C.d}")
    
    analysis_group = archive.archive_file.require_group("analysis")
    analysis_group["pulse_amplitude_fit"] = np.zeros(0)
    analysis_group["pulse_amplitude_fit"].attrs["scaled_frequency"] = scaled_frequency
    analysis_group["pulse_amplitude_fit"].attrs["scaled_density"] = scaled_density
    analysis_group["pulse_amplitude_fit"].attrs["scaled_samples"] = scaled_samples
    analysis_group["pulse_amplitude_fit"].attrs["scaled_amplitude"] = scaled_amplitude
    analysis_group["pulse_amplitude_fit"].attrs["scaled_sweep"] = np.asarray(scaled_sweep)
    analysis_group["pulse_amplitude_fit"].attrs["scaled_time_step"] = scaled_time_step
    analysis_group["pulse_amplitude_fit"].attrs["scaled_time_end"] = scaled_time_end
    analysis_group["pulse_amplitude_fit"].attrs["scaled_pulse_time"] = scaled_pulse_time
    analysis_group["pulse_amplitude_fit"].attrs["scaled_frequency_step"] = scaled_frequency_step
    analysis_group["pulse_amplitude_fit"].attrs["mean_squared_error"] = mean_squared_error
    analysis_group["pulse_amplitude_fit"].attrs["root_mean_squared_error"] = np.sqrt(mean_squared_error)
    analysis_group["pulse_amplitude_fit"].attrs["old_archive_time"] = experiment_results.archive_time

    plt.figure()
    plt.plot(frequency, frequency_amplitude_measured, "r.")
    plt.plot(frequency, frequency_amplitude_predict, "b.")
    plt.plot(frequency, frequency_amplitude_residual, "y.")
    plt.legend(["Measured", "Modelled", "Residual"])
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude (Hz)")
    if archive:
        archive.write_plot("Neural pulse amplitude (best fit)", "analysis_pulse_amplitude_fit_result")
    plt.draw()

    plt.figure()
    plt.plot(mean_squared_errors)
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    if archive:
        archive.write_plot("Neural pulse amplitude\nerror per epoch", "analysis_pulse_amplitude_fit_error")
    plt.draw()

    plt.show()