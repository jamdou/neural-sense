import numpy as np
import scipy.optimize
import math
import matplotlib.pyplot as plt
from numpy.lib.function_base import gradient

import archive as arch
import util
from util import PrettyTritty as C
import test_signal
import reconstruction as recon

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

    scaled.amplitude = 1000
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

    def noise_model(rabi_frequency, noise_amplitude, time_extra_cycle):
        rabi_frequency_readout = 2e4
        larmor_frequency = 840e3
        # if time_extra_cycle < 0:
        #     time_extra_cycle = 0
        # if time_extra_cycle > 0.25:
        #     time_extra_cycle = 0.25
        time_extra_cycle = 0#0.25
        time_end = scaled.time_end + time_extra_cycle/rabi_frequency_readout

        noise_end_sample = noise_amplitude*np.sin(math.tau*frequency_line_noise*time_end)

        true_rabi = np.sqrt(rabi_frequency**2 + noise_end_sample**2)
        cos_tilt = rabi_frequency/true_rabi
        sin_tilt = noise_end_sample/true_rabi

        true_rabi_readout = np.sqrt(rabi_frequency_readout**2 + (noise_end_sample + 0.25*(rabi_frequency_readout**2)/larmor_frequency)**2)
        cos_tilt_readout = rabi_frequency_readout/true_rabi_readout
        sin_tilt_readout = (noise_end_sample + 0.25*(rabi_frequency_readout**2)/larmor_frequency)/true_rabi_readout

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
            fit = scipy.optimize.curve_fit(noise_model, frequency, frequency_amplitude_measured, [starting_noise_amplitude, 0], method = "trf")[0]
            noise_amplitude_fitted_start = fit[0]
            time_extra_cycle = fit[1]
        except:
            noise_amplitude_fitted_start = 0
            time_extra_cycle = 0
        # frequency_amplitude_fitted_start = noise_model(frequency, noise_amplitude_fitted_start, time_extra_cycle)
        frequency_amplitude_fitted_start = noise_model(frequency, 0, 0)
        frequency_amplitude_residual_start = frequency_amplitude_measured - frequency_amplitude_fitted_start
        mean_squared_error_start = np.mean(frequency_amplitude_residual_start**2)
        if mean_squared_error is None or mean_squared_error_start < mean_squared_error:
            noise_amplitude_fitted = noise_amplitude_fitted_start
            frequency_amplitude_fitted = frequency_amplitude_fitted_start
            frequency_amplitude_residual = frequency_amplitude_residual_start
            mean_squared_error = mean_squared_error_start

    C.print(f"Noise = {noise_amplitude_fitted:.2f}")
    C.print(f"Extra cycle = {time_extra_cycle:.2f}")
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

    noise_amplitude_fitted = scipy.optimize.curve_fit(noise_model, frequency, frequency_amplitude_measured, 500)[0][0]

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

def add_shot_noise_to_projections(projections, atom_count, noise_modifier):
    noisy_projections = np.empty_like(projections)
    populations = np.empty(3)
    for projection_index, projection in enumerate(projections):
        populations[0] = (projection**2 + 2*projection + 1)/4    # +
        populations[1] = (1 - projection**2)/2                   # 0
        populations[2] = (projection**2 - 2*projection + 1)/4    # -
        
        populations *= atom_count
        noisy_populations = np.random.poisson(populations/(noise_modifier**2))
        noisy_populations *= noise_modifier**2
        
        noisy_projections[projection_index] = (noisy_populations[0] - noisy_populations[2])/np.sum(noisy_populations)
    return noisy_projections

def add_shot_noise(experiment_results:arch.ExperimentResults, scaled:util.ScaledParameters, archive:arch.Archive = None, atom_count = 1e6, noise_modifier = 1) -> arch.ExperimentResults:
    C.starting("addition of simulated shot noise")
    fourier_scale = -1/(math.tau*scaled.time_end)
    projections = experiment_results.frequency_amplitude/fourier_scale

    noisy_projections = add_shot_noise_to_projections(projections, atom_count, noise_modifier)

    modified_experiment_results = arch.ExperimentResults(frequency = experiment_results.frequency.copy(), frequency_amplitude = noisy_projections*fourier_scale, archive_time = experiment_results.archive_time, experiment_type = f"{experiment_results.experiment_type}, Shot noise added")

    error = fourier_scale*(noisy_projections - projections)
    error_rms = np.sqrt(np.mean(error**2))
    error_average = np.mean(np.abs(error))
    error_max = np.max(np.abs(error))
    C.print(f"Average error: {error_average:.4g}Hz")
    C.print(f"RMS error: {error_rms:.4g}Hz")
    C.print(f"Max error: {error_max:.4g}Hz")

    C.finished("addition of simulated shot noise")

    shot_noise_group = archive.archive_file.require_group("added_shot_noise")
    shot_noise_group["unmodified"] = experiment_results.frequency_amplitude
    shot_noise_group["noise_added"] = modified_experiment_results.frequency_amplitude
    shot_noise_group["residual"] = error
    shot_noise_group["residual"].attrs["average"] = error_average
    shot_noise_group["residual"].attrs["rms"] = error_rms
    shot_noise_group["residual"].attrs["max"] = error_max
    
    plt.figure()
    plt.plot(experiment_results.frequency, experiment_results.frequency_amplitude, "r--", label = "Unmodified")
    plt.plot(modified_experiment_results.frequency, modified_experiment_results.frequency_amplitude, "bx", label = "Noise added")
    plt.plot(modified_experiment_results.frequency, error, "y+", label = "Residual")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Frequency amplitude (Hz)")
    plt.legend()
    if archive:
        archive.write_plot("Added shot noise", "added_shot_noise")
    plt.draw()

    return modified_experiment_results

def remove_line_noise_from_model(experiment_results:arch.ExperimentResults, scaled:util.ScaledParameters, line_noise_model:test_signal.LineNoiseModel, archive:arch.Archive = None):
    rabi = experiment_results.frequency

    tilt = np.sum(line_noise_model.a*np.cos(math.tau*line_noise_model.h*scaled.time_end + line_noise_model.p))/rabi
    
    phase = np.zeros_like(rabi)
    time_step = scaled.time_step/100
    time = np.arange(0, scaled.time_end, time_step)
    for time_sample in time:
        phase += math.tau*np.sqrt(rabi**2 + (np.sum(line_noise_model.a*np.cos(math.tau*line_noise_model.h*time_sample + line_noise_model.p)))**2)*time_step

    error = -1/(math.tau*scaled.time_end)*np.cos(phase)*np.sin(tilt)*-1
    modified_experiment_results = arch.ExperimentResults(frequency = experiment_results.frequency.copy(), frequency_amplitude = experiment_results.frequency_amplitude - error, archive_time = experiment_results.archive_time, experiment_type = f"{experiment_results.experiment_type}, 50Hz corrected feed forward")

    plt.figure()
    plt.plot(experiment_results.frequency, experiment_results.frequency_amplitude, "r--", label = "Unmodified")
    plt.plot(modified_experiment_results.frequency, modified_experiment_results.frequency_amplitude, "bx", label = "Modified")
    plt.plot(modified_experiment_results.frequency, error, "y+", label = "\"Noise\" removed")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Frequency amplitude (Hz)")
    plt.legend()
    if archive:
        archive.write_plot("Removed line noise", "removed_line_noise")
    plt.draw()

    return modified_experiment_results

def remove_line_noise_from_evaluation(experiment_results:arch.ExperimentResults, scaled:util.ScaledParameters, evaluation_experiment_results:arch.ExperimentResults, archive:arch.Archive = None):

    error_raw = evaluation_experiment_results.frequency_amplitude
    error = []
    for error_instance, frequency_instance in zip(error_raw, evaluation_experiment_results.frequency):
        if frequency_instance in experiment_results.frequency:
            error.append(error_instance)
    error = np.array(error)
    
    modified_experiment_results = arch.ExperimentResults(frequency = experiment_results.frequency.copy(), frequency_amplitude = experiment_results.frequency_amplitude - error, archive_time = experiment_results.archive_time, experiment_type = f"{experiment_results.experiment_type}, 50Hz corrected feed forward evaluation")

    plt.figure()
    plt.plot(experiment_results.frequency, experiment_results.frequency_amplitude, "r--", label = "Unmodified")
    plt.plot(modified_experiment_results.frequency, modified_experiment_results.frequency_amplitude, "bx", label = "Modified")
    plt.plot(modified_experiment_results.frequency, error, "y+", label = "\"Noise\" removed")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Frequency amplitude (Hz)")
    plt.legend()
    if archive:
        archive.write_plot("Removed line noise", "removed_line_noise")
    plt.draw()

    return modified_experiment_results

def sweep_sensing_coherence(archive:arch.Archive, time_properties:test_signal.TimeProperties, sweep_parameters = [30, None, 10]):
    sweep_parameters[1] = sweep_parameters[1] or (time_properties.time_coarse.size)
    sampleses = np.arange(sweep_parameters[1], sweep_parameters[0] - 1, -sweep_parameters[2])
    frequency_max = 1/(2*time_properties.time_step_coarse)
    d_frequency_min = frequency_max/(time_properties.time_coarse.size + 1)
    coherences_equidistant = np.empty_like(sampleses, dtype = np.float64)
    coherences_lowpass = np.empty_like(sampleses, dtype = np.float64)
    coherences_highpass = np.empty_like(sampleses, dtype = np.float64)
    coherences_random_on_axis = np.empty_like(sampleses, dtype = np.float64)
    coherences_random_on_axis_min = np.empty_like(sampleses, dtype = np.float64)
    coherences_random_on_axis_max = np.empty_like(sampleses, dtype = np.float64)
    coherences_random_off_axis = np.empty_like(sampleses, dtype = np.float64)
    coherences_random_off_axis_min = np.empty_like(sampleses, dtype = np.float64)
    coherences_random_off_axis_max = np.empty_like(sampleses, dtype = np.float64)
    reconstruction = recon.Reconstruction(time_properties)

    for samples_index, samples in enumerate(sampleses):
        # Equidistant
        d_frequency = frequency_max/(samples + 1)
        frequency = np.arange(d_frequency, frequency_max - d_frequency/2, d_frequency)
        reconstruction.read_frequencies_directly(frequency = frequency, frequency_amplitude = frequency.copy(), frequency_cutoff_low = 0, frequency_cutoff_high = 1e6, random_seed = util.Seeds.metroid, number_of_samples = samples)
        coherence = reconstruction.evaluate_coherence()
        coherences_equidistant[samples_index] = coherence

        # Random (on axis)
        frequency = np.arange(d_frequency_min, frequency_max - d_frequency_min/2, d_frequency_min)
        coherence_max = 0
        coherence_min = 1
        for repeat_index in range(50):
            reconstruction.read_frequencies_directly(frequency = frequency, frequency_amplitude = frequency.copy(), frequency_cutoff_low = 0, frequency_cutoff_high = 1e6, random_seed = util.Seeds.metroid*repeat_index, number_of_samples = samples)
            coherence = reconstruction.evaluate_coherence()
            if repeat_index == 0:
                coherences_random_on_axis[samples_index] = coherence
            if coherence < coherence_min:
                coherence_min = coherence
            if coherence > coherence_max:
                coherence_max = coherence
        coherences_random_on_axis_min[samples_index] = coherence_min
        coherences_random_on_axis_max[samples_index] = coherence_max

        # Random (off axis)
        coherence_max = 0
        coherence_min = 1
        for repeat_index in range(10):
            np.random.seed(util.Seeds.metroid*repeat_index)
            frequency = np.random.uniform(0, frequency_max, samples)
            reconstruction.read_frequencies_directly(frequency = frequency, frequency_amplitude = frequency.copy(), frequency_cutoff_low = 0, frequency_cutoff_high = 1e6, random_seed = util.Seeds.metroid*repeat_index, number_of_samples = samples)
            coherence = reconstruction.evaluate_coherence()
            if repeat_index == 0:
                coherences_random_off_axis[samples_index] = coherence
            if coherence < coherence_min:
                coherence_min = coherence
            if coherence > coherence_max:
                coherence_max = coherence
        coherences_random_off_axis_min[samples_index] = coherence_min
        coherences_random_off_axis_max[samples_index] = coherence_max

        # Low pass
        frequency = np.arange(d_frequency_min, d_frequency_min*(samples + 1) - d_frequency_min/2, d_frequency_min)
        reconstruction.read_frequencies_directly(frequency = frequency, frequency_amplitude = frequency.copy(), frequency_cutoff_low = 0, frequency_cutoff_high = 1e6, random_seed = util.Seeds.metroid, number_of_samples = samples)
        coherence = reconstruction.evaluate_coherence()
        coherences_lowpass[samples_index] = coherence

        # High pass
        frequency += frequency_max - d_frequency_min*(samples + 1)
        reconstruction.read_frequencies_directly(frequency = frequency, frequency_amplitude = frequency.copy(), frequency_cutoff_low = 0, frequency_cutoff_high = 1e6, random_seed = util.Seeds.metroid, number_of_samples = samples)
        coherence = reconstruction.evaluate_coherence()
        coherences_highpass[samples_index] = coherence

    # print(time_properties.time_coarse)
    # print(time_properties.time_coarse.size)
    # print(sweep_parameters[1])
    # print(d_frequency_min)
    # print(frequency_max)
    print(frequency)

    plt.figure()
    plt.plot(sampleses, coherences_equidistant, "k-", label = "Equidistant")
    plt.plot(sampleses, coherences_random_on_axis, "m-", label = "Random (on axis)")
    plt.fill_between(sampleses, coherences_random_on_axis_min, coherences_random_on_axis_max, color = (1, 0, 1, 0.25))
    plt.plot(sampleses, coherences_random_off_axis, "y-", label = "Random (off axis)")
    plt.fill_between(sampleses, coherences_random_off_axis_min, coherences_random_off_axis_max, color = (1, 1, 0, 0.25))
    plt.plot(sampleses, coherences_lowpass, "b-", label = "Low pass")
    plt.plot(sampleses, coherences_highpass, "r-", label = "High pass")
    plt.legend()
    plt.xlabel("Number of samples")
    plt.ylabel("Sampling coherence")
    if archive:
        archive.write_plot("Sampling coherence", "coherence_sweep")
    plt.draw()

    plt.figure()
    plt.subplot(2, 1, 2)
    plt.plot(sampleses, np.clip(np.floor(1/(3*coherences_equidistant)), 0, sweep_parameters[1]), "k-", label = "Equidistant")
    plt.plot(sampleses, np.clip(np.floor(1/(3*coherences_random_on_axis)), 0, sweep_parameters[1]), "m-", label = "Random (on axis)")
    plt.fill_between(sampleses, np.clip(np.floor(1/(3*coherences_random_on_axis_min)), 0, sweep_parameters[1]), np.clip(np.floor(1/(3*coherences_random_on_axis_max)), 0, sweep_parameters[1]), color = (1, 0, 1, 0.25))
    plt.plot(sampleses, np.clip(np.floor(1/(3*coherences_random_off_axis)), 0, sweep_parameters[1]), "y-", label = "Random (off axis)")
    plt.fill_between(sampleses, np.clip(np.floor(1/(3*coherences_random_off_axis_min)), 0, sweep_parameters[1]), np.clip(np.floor(1/(3*coherences_random_off_axis_max)), 0, sweep_parameters[1]), color = (1, 1, 0, 0.25))
    plt.plot(sampleses, np.clip(np.floor(1/(3*coherences_lowpass)), 0, sweep_parameters[1]), "b-", label = "Low pass")
    plt.plot(sampleses, np.clip(np.floor(1/(3*coherences_highpass)), 0, sweep_parameters[1]), "r-", label = "High pass")
    # plt.legend()
    plt.xlabel("Number of samples")
    # plt.ylabel("Maximum guaranteed sparsity")
    plt.ylim([0, 15])

    plt.subplot(2, 1, 1)
    plt.plot(sampleses, np.clip(np.floor(1/(3*coherences_equidistant)), 0, sweep_parameters[1]), "k-", label = "Equidistant")
    plt.plot(sampleses, np.clip(np.floor(1/(3*coherences_random_on_axis)), 0, sweep_parameters[1]), "m-", label = "Random (on axis)")
    plt.fill_between(sampleses, np.clip(np.floor(1/(3*coherences_random_on_axis_min)), 0, sweep_parameters[1]), np.clip(np.floor(1/(3*coherences_random_on_axis_max)), 0, sweep_parameters[1]), color = (1, 0, 1, 0.25))
    plt.plot(sampleses, np.clip(np.floor(1/(3*coherences_random_off_axis)), 0, sweep_parameters[1]), "y-", label = "Random (off axis)")
    plt.fill_between(sampleses, np.clip(np.floor(1/(3*coherences_random_off_axis_min)), 0, sweep_parameters[1]), np.clip(np.floor(1/(3*coherences_random_off_axis_max)), 0, sweep_parameters[1]), color = (1, 1, 0, 0.25))
    plt.plot(sampleses, np.clip(np.floor(1/(3*coherences_lowpass)), 0, sweep_parameters[1]), "b-", label = "Low pass")
    plt.plot(sampleses, np.clip(np.floor(1/(3*coherences_highpass)), 0, sweep_parameters[1]), "r-", label = "High pass")
    plt.legend()
    plt.ylabel("Maximum guaranteed sparsity")
    if archive:
        archive.write_plot("Maximum guaranteed sparsity", "guaranteed_sparsity_sweep")
    plt.draw()

def sweep_sensing_coherence_lowpass(archive:arch.Archive, time_properties:test_signal.TimeProperties, sweep_parameters = [30, None, 10]):
    sweep_parameters[1] = sweep_parameters[1] or (time_properties.time_coarse.size)
    sampleses = np.arange(sweep_parameters[1], sweep_parameters[0], -sweep_parameters[2])
    frequency_max = 1/(2*time_properties.time_step_coarse)
    d_frequency = frequency_max/(time_properties.time_coarse.size + 1)
    coherences = np.empty_like(sampleses)
    reconstruction = recon.Reconstruction(time_properties)
    for samples_index, samples in enumerate(sampleses):
        frequency = np.arange(d_frequency, d_frequency*sampleses + d_frequency/2, d_frequency)
        reconstruction.read_frequencies_directly(frequency = frequency, frequency_amplitude = 0*frequency, frequency_cutoff_low = 0, frequency_cutoff_high = 1e6, random_seed = util.Seeds.metroid, number_of_samples = samples)
        coherence = reconstruction.evaluate_coherence()
        coherences[samples_index] = coherence

    plt.figure()
    plt.plot(sampleses, coherences, "-k")
    plt.xlabel("Number of samples")
    plt.ylabel("Sampling coherence")
    if archive:
        archive.write_plot("Coherence sweep (lowpass)", "coherence_sweep_lowpass")
    plt.draw()