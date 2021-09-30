import numpy as np
# from scipy.optimize import curve_fit
import math
import matplotlib.pyplot as plt
from numpy.lib.function_base import gradient

import archive as arch
import util
from util import C

def find_neural_signal_size(experiment_results:arch.ExperimentResults, scaled:util.ScaledParameters, archive:arch.Archive = None):
    print(f"{C.y}Starting fit...{C.d}")

    frequency = np.arange(scaled.sweep[0], scaled.sweep[1], scaled.frequency_step)
    frequency = np.copy(experiment_results.frequency)
    time = np.arange(0, scaled.time_end, scaled.time_step)
    
    amplitude_approximated_normalised = np.zeros_like(time)
    for time_index in range(time.size):
        if time[time_index] >= scaled.pulse_time and time[time_index] < scaled.pulse_time + 1/scaled.frequency:
            amplitude_approximated_normalised[time_index] = np.sin(math.tau*scaled.frequency*(time[time_index] - scaled.pulse_time))
    frequency_mesh, time_mesh = np.meshgrid(frequency, time)
    frequency_mesh, amplitude_approximated_normalised_mesh = np.meshgrid(frequency, amplitude_approximated_normalised)
    frequency_amplitude_approximated_normalised = np.sum(np.sin(math.tau*frequency_mesh*time_mesh)*amplitude_approximated_normalised_mesh, axis = 0)*scaled.time_step/scaled.time_end

    frequency_amplitude_measured = np.copy(experiment_results.frequency_amplitude)

    scaled.amplitude = 100
    decent_step_size = 1e-6
    mean_squared_errors = []
    for epoch in range(1000000):
        frequency_amplitude_predict = (1/(math.tau*scaled.time_end))*np.sin(math.tau*scaled.time_end*scaled.amplitude*frequency_amplitude_approximated_normalised)
        frequency_amplitude_residual = frequency_amplitude_predict - frequency_amplitude_measured
        mean_squared_error = np.mean(frequency_amplitude_residual**2)
        frequency_amplitude_gradient = np.sum(2*frequency_amplitude_residual*np.cos(math.tau*scaled.time_end*scaled.amplitude*frequency_amplitude_approximated_normalised)*scaled.amplitude*frequency_amplitude_approximated_normalised)
        scaled.amplitude -= decent_step_size*frequency_amplitude_gradient
        mean_squared_errors += [mean_squared_error]
    print(f"\tAmp: {scaled.amplitude}\n\tSquared error: {mean_squared_error}\n\tRMS Error: {np.sqrt(mean_squared_error)}")
    print(f"{C.g}Done!{C.d}")
    
    analysis_group = archive.archive_file.require_group("analysis")
    analysis_group["pulse_amplitude_fit"] = np.zeros(0)
    analysis_group["pulse_amplitude_fit"].attrs["scaled_amplitude"] = scaled.amplitude
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

def find_line_noise_size(experiment_results:arch.ExperimentResults, scaled:util.ScaledParameters, archive:arch.Archive = None):
    print(f"{C.y}Starting fit (line noise)...{C.d}")

    frequency = np.arange(scaled.sweep[0], scaled.sweep[1], scaled.frequency_step)
    frequency = np.copy(experiment_results.frequency)
    time = np.arange(0, scaled.time_end, scaled.time_step)
    
    amplitude_approximated_normalised = np.cos(math.tau*50*time)
    frequency_mesh, time_mesh = np.meshgrid(frequency, time)
    frequency_mesh, amplitude_approximated_normalised_mesh = np.meshgrid(frequency, amplitude_approximated_normalised)
    frequency_amplitude_approximated_normalised = np.sum(np.sin(math.tau*frequency_mesh*time_mesh)*amplitude_approximated_normalised_mesh, axis = 0)*scaled.time_step/scaled.time_end

    frequency_amplitude_measured = np.copy(experiment_results.frequency_amplitude)

    amplitude = 1000
    decent_step_size = 1e-6
    mean_squared_errors = []
    for epoch in range(1000000):
    # for epoch in range(1):
        frequency_amplitude_predict = (1/(math.tau*scaled.time_end))*np.sin(math.tau*scaled.time_end*amplitude*frequency_amplitude_approximated_normalised)
        frequency_amplitude_residual = frequency_amplitude_predict - frequency_amplitude_measured
        mean_squared_error = np.mean(frequency_amplitude_residual**2)
        frequency_amplitude_gradient = np.sum(2*frequency_amplitude_residual*np.cos(math.tau*scaled.time_end*amplitude*frequency_amplitude_approximated_normalised)*amplitude*frequency_amplitude_approximated_normalised)
        amplitude -= decent_step_size*frequency_amplitude_gradient
        mean_squared_errors += [mean_squared_error]
    print(f"\tAmp: {amplitude}\n\tSquared error: {mean_squared_error}\n\tRMS Error: {np.sqrt(mean_squared_error)}")
    print(f"{C.g}Done!{C.d}")
    
    analysis_group = archive.archive_file.require_group("analysis")
    analysis_group["line_noise_amplitude_fit"] = -frequency_amplitude_residual
    analysis_group["line_noise_amplitude_fit"].attrs["line_noise_amplitude"] = amplitude
    analysis_group["line_noise_amplitude_fit"].attrs["mean_squared_error"] = mean_squared_error
    analysis_group["line_noise_amplitude_fit"].attrs["root_mean_squared_error"] = np.sqrt(mean_squared_error)
    if experiment_results.archive_time:
        analysis_group["line_noise_amplitude_fit"].attrs["old_archive_time"] = experiment_results.archive_time

    modified_experiment_results = arch.ExperimentResults(frequency = 1*experiment_results.frequency, frequency_amplitude = -(1/(math.tau*scaled.time_end))*np.arcsin(math.tau*scaled.time_end*frequency_amplitude_residual), archive_time = experiment_results.archive_time, experiment_type = f"{experiment_results.experiment_type}, 50Hz corrected")

    plt.figure()
    plt.plot(frequency, frequency_amplitude_measured, "r.")
    plt.plot(frequency, frequency_amplitude_predict, "b.")
    plt.plot(frequency, -frequency_amplitude_residual, "y.")
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

    return modified_experiment_results

def find_line_noise_size_from_tilt(experiment_results:arch.ExperimentResults, scaled:util.ScaledParameters, archive:arch.Archive = None, frequency_line_noise = 50):
    print(f"{C.y}Starting fit (line noise from tilt)...{C.d}")

    frequency = np.copy(experiment_results.frequency)
    time = np.arange(0, scaled.time_end, scaled.time_step)
    
    frequency_amplitude_approximated_normalised = (1/(math.tau*scaled.time_end))*(1/(2*frequency))*np.sin(math.tau*frequency*scaled.time_end)*np.sin(math.tau*frequency_line_noise*scaled.time_end)
    # frequency_amplitude_approximated_normalised = (1/(math.tau*scaled.time_end))*(1/2)*np.sin(math.tau*frequency*scaled.time_end)*np.sin(math.tau*frequency_line_noise*scaled.time_end)
    frequency_amplitude_measured = np.copy(experiment_results.frequency_amplitude)

    amplitude = np.sum(frequency_amplitude_approximated_normalised*frequency_amplitude_measured)/np.sum(frequency_amplitude_approximated_normalised**2)
    frequency_amplitude_predict = math.tau*scaled.time_end*amplitude*frequency_amplitude_approximated_normalised
    frequency_amplitude_residual = frequency_amplitude_measured - frequency_amplitude_predict
    mean_squared_error = np.mean(frequency_amplitude_residual**2)

    print(f"\tAmp: {amplitude}\n\tSquared error: {mean_squared_error}\n\tRMS Error: {np.sqrt(mean_squared_error)}")
    print(f"{C.g}Done!{C.d}")
    
    analysis_group = archive.archive_file.require_group("analysis")
    analysis_group["line_noise_amplitude_fit"] = frequency_amplitude_residual
    analysis_group["line_noise_amplitude_fit"].attrs["line_noise_amplitude"] = amplitude
    analysis_group["line_noise_amplitude_fit"].attrs["mean_squared_error"] = mean_squared_error
    analysis_group["line_noise_amplitude_fit"].attrs["root_mean_squared_error"] = np.sqrt(mean_squared_error)
    if experiment_results.archive_time:
        analysis_group["line_noise_amplitude_fit"].attrs["old_archive_time"] = experiment_results.archive_time

    modified_experiment_results = arch.ExperimentResults(frequency = 1*experiment_results.frequency, frequency_amplitude = 1*frequency_amplitude_residual, archive_time = experiment_results.archive_time, experiment_type = f"{experiment_results.experiment_type}, 50Hz corrected")

    plt.figure()
    plt.plot(frequency, frequency_amplitude_measured, "r.")
    plt.plot(frequency, frequency_amplitude_predict, "b.")
    plt.plot(frequency, frequency_amplitude_residual, "y.")
    plt.legend(["Measured", "Modelled", "Residual"])
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude (Hz)")
    if archive:
        archive.write_plot("Neural pulse line tilt (best fit)", "analysis_line_tilt_fit_result")
    plt.draw()

    return modified_experiment_results

def find_noise_size_from_rabi(experiment_results:arch.ExperimentResults, scaled:util.ScaledParameters, archive:arch.Archive = None, frequency_line_noise = 50):
    C.starting("finding line noise (Rabi model)")

    frequency = experiment_results.frequency.copy()
    frequency_amplitude_measured = experiment_results.frequency_amplitude.copy()

    def noise_model(rabi_frequency, noise_amplitude):
        rabi_frequency_readout = 2e4
        time_end = scaled.time_end

        noise_end_sample = noise_amplitude*np.sin(math.tau*frequency_line_noise*time_end)

        true_rabi = np.sqrt(rabi_frequency**2 + noise_end_sample**2)
        cos_tilt = rabi_frequency/true_rabi
        sin_tilt = noise_end_sample/true_rabi

        true_rabi_readout = np.sqrt(rabi_frequency_readout**2 + noise_end_sample**2)
        cos_tilt_readout = rabi_frequency_readout/true_rabi_readout
        # sin_tilt_readout = noise_end_sample/true_rabi_readout

        rabi_phase = math.tau*(rabi_frequency + (noise_amplitude**2)/(4*rabi_frequency))*time_end# - (noise_amplitude**2)/(8*rabi_frequency*frequency_line_noise)*np.cos(2*math.tau*frequency_line_noise*time_end))
        cos_rabi_phase = np.cos(rabi_phase)
        # sin_rabi_phase = np.sin(rabi_phase)

        readout = math.pi/(2*cos_tilt_readout)
        cos_readout = np.cos(readout)
        sin_readout = np.sin(readout)

        return 1/(math.tau*time_end)*cos_rabi_phase*(cos_tilt*cos_readout - sin_tilt*sin_readout)

        # return 1/(math.tau*time_end)*(cos_rabi_phase*(cos_tilt*(cos_tilt_readout**2*cos_readout + sin_tilt_readout**2) - sin_tilt*cos_tilt_readout*sin_readout) + sin_rabi_phase*cos_tilt_readout*sin_tilt_readout*(cos_readout - 1))

    noise_amplitude_fitted = None
    frequency_amplitude_fitted = None
    frequency_amplitude_residual = None
    mean_squared_error = None
    for starting_noise_amplitude in range(-2000, 2100, 100):
        try:
            noise_amplitude_fitted_start = curve_fit(noise_model, frequency, frequency_amplitude_measured, starting_noise_amplitude, method = "trf")[0][0]
        except:
            noise_amplitude_fitted_start = 0
        frequency_amplitude_fitted_start = noise_model(frequency, noise_amplitude_fitted_start)
        frequency_amplitude_residual_start = frequency_amplitude_measured - frequency_amplitude_fitted_start
        mean_squared_error_start = np.mean(frequency_amplitude_residual_start**2)
        if mean_squared_error is None or mean_squared_error_start < mean_squared_error:
            noise_amplitude_fitted = noise_amplitude_fitted_start
            frequency_amplitude_fitted = frequency_amplitude_fitted_start
            frequency_amplitude_residual = frequency_amplitude_residual_start
            mean_squared_error = mean_squared_error_start

    C.print(f"Noise = {noise_amplitude_fitted}")
    C.print(f"RMSE = {np.sqrt(mean_squared_error):.2f}")
    C.finished("finding line noise (Rabi model)")
    
    analysis_group = archive.archive_file.require_group("analysis")
    analysis_group["line_noise_amplitude_fit"] = frequency_amplitude_residual
    # analysis_group["line_noise_amplitude_fit"].attrs["line_noise_amplitude"] = noise_amplitude_fitted
    analysis_group["line_noise_amplitude_fit"].attrs["mean_squared_error"] = mean_squared_error
    analysis_group["line_noise_amplitude_fit"].attrs["root_mean_squared_error"] = np.sqrt(mean_squared_error)
    if experiment_results.archive_time:
        analysis_group["line_noise_amplitude_fit"].attrs["old_archive_time"] = experiment_results.archive_time

    modified_experiment_results = arch.ExperimentResults(frequency = 1*experiment_results.frequency, frequency_amplitude = frequency_amplitude_residual.copy(), archive_time = experiment_results.archive_time, experiment_type = f"{experiment_results.experiment_type}, 50Hz corrected")

    plt.figure()
    plt.plot(frequency, frequency_amplitude_measured, "r.")
    plt.plot(frequency, frequency_amplitude_fitted, "b.")
    plt.plot(frequency, frequency_amplitude_residual, "y.")
    plt.legend(["Measured", "Fitted", "Residual"])
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude (Hz)")
    if archive:
        archive.write_plot("Neural pulse line Rabi (best fit)", "analysis_line_rabi_fit_result")
    plt.draw()
    return modified_experiment_results

def find_noise_size_from_fourier_transform(experiment_results:arch.ExperimentResults, scaled:util.ScaledParameters, archive:arch.Archive = None, frequency_line_noise = 50):
    C.starting("finding line noise (FT model)")

    frequency = experiment_results.frequency.copy()
    frequency_amplitude_measured = experiment_results.frequency_amplitude.copy()

    def noise_model(rabi_frequency, noise_amplitude):
        rabi_frequency_readout = 2e4
        # readout_overshoot = 10*math.pi*((math.tau*noise_amplitude)**2)/(8*((math.tau*rabi_frequency_readout)**2))*(1 - np.cos(2*math.tau*frequency_line_noise*scaled.time_end))
        readout_overshoot = math.pi/2*(np.sqrt((math.tau*rabi_frequency_readout)**2 + (math.tau*noise_amplitude*np.sin(math.tau*frequency_line_noise*scaled.time_end))**2)/(math.tau*rabi_frequency_readout) - 1)
        # C.print(f"{readout_overshoot}")
        fourier_transform_line_noise = math.tau*noise_amplitude/2*(np.sin(math.tau*(rabi_frequency + frequency_line_noise)*scaled.time_end)/(math.tau*(rabi_frequency + frequency_line_noise)) - np.sin(math.tau*(rabi_frequency - frequency_line_noise)*scaled.time_end)/(math.tau*(rabi_frequency - frequency_line_noise)))
        return -1/(math.tau*scaled.time_end)*(np.sin(fourier_transform_line_noise)*np.cos(readout_overshoot) - np.cos(fourier_transform_line_noise)*np.cos(math.tau*rabi_frequency*scaled.time_end)*np.sin(readout_overshoot))
        # return 1/(math.tau*scaled.time_end)*( - np.cos(fourier_transform_line_noise)*np.cos(math.tau*rabi_frequency*scaled.time_end)*np.sin(readout_overshoot))

    noise_amplitude_fitted = curve_fit(noise_model, frequency, frequency_amplitude_measured, 500)[0][0]

    frequency_amplitude_fitted = noise_model(frequency, 500)
    frequency_amplitude_residual = frequency_amplitude_measured - frequency_amplitude_fitted

    mean_squared_error = np.mean(frequency_amplitude_residual**2)

    C.print(f"Noise = {noise_amplitude_fitted:.2f}")
    C.print(f"RMSE = {np.sqrt(mean_squared_error):.2f}")
    C.finished("finding line noise (FT model)")
    
    analysis_group = archive.archive_file.require_group("analysis")
    analysis_group["line_noise_amplitude_fit"] = frequency_amplitude_residual
    analysis_group["line_noise_amplitude_fit"].attrs["line_noise_amplitude"] = noise_amplitude_fitted
    analysis_group["line_noise_amplitude_fit"].attrs["mean_squared_error"] = mean_squared_error
    analysis_group["line_noise_amplitude_fit"].attrs["root_mean_squared_error"] = np.sqrt(mean_squared_error)
    if experiment_results.archive_time:
        analysis_group["line_noise_amplitude_fit"].attrs["old_archive_time"] = experiment_results.archive_time

    modified_experiment_results = arch.ExperimentResults(frequency = experiment_results.frequency.copy(), frequency_amplitude = frequency_amplitude_residual.copy(), archive_time = experiment_results.archive_time, experiment_type = f"{experiment_results.experiment_type}, 50Hz corrected")

    plt.figure()
    plt.plot(frequency, frequency_amplitude_measured, "r.")
    plt.plot(frequency, frequency_amplitude_fitted, "b.")
    plt.plot(frequency, frequency_amplitude_residual, "y.")
    plt.legend(["Measured", "Fitted", "Residual"])
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude (Hz)")
    if archive:
        archive.write_plot("Neural pulse line FT (best fit)", "analysis_line_ft_fit_result")
    plt.draw()
    return modified_experiment_results

def find_time_blind_spots(scaled:util.ScaledParameters, archive:arch.Archive = None):
    time = np.arange(0, scaled.time_end, scaled.time_step)
    frequencies = scaled.sample_frequencies

    blind_spots = np.zeros_like(time)
    for frequency in frequencies:
        blind_spots += np.abs(np.sin(math.tau*frequency*time))
    blind_spots = (max(blind_spots) - blind_spots)/max(blind_spots)
    
    if archive:
        analysis_group = archive.archive_file.require_group("analysis")
        analysis_group["blind_spots"] = blind_spots

    plt.figure()
    plt.plot(time, blind_spots, "k-")
    plt.xlabel("Time (s)")
    plt.ylabel("Blindness")
    if archive:
        archive.write_plot("Blind spots", "time_blind_spots")
    plt.draw()