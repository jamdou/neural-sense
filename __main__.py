import math
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use("tkagg")
# import sys, getopt      # Command line arguments
from numba import cuda  # GPU code
import colorama         # Colourful terminal
colorama.init()

# The different pieces that make up this sensing code
import archive as arch              # Saving results and configurations
from archive import handle_arguments

import test_signal                  # The properties of the magnetic signal, used for simulations and reconstructions
import spinsim                      # Main simulation package
import reconstruction as recon      # Uses compressive sensing to reconstruct the a magnetic signal
import sim
import util
from util import PrettyTritty as C
import analysis

if __name__ == "__main__":
    # This will be recorded in the HDF5 file to give context for what was being tested
    description_of_test = ""

    # Check to see if there is a compatible GPU
    if cuda.list_devices():
        C.print(f"{C.g}Using cuda device {cuda.list_devices()[0].name.decode('UTF-8')}{C.d}")
    else:
        print(f"{C.r}No cuda devices found. System is incompatible. Exiting...{C.d}")
        exit()

    profile_state, archive_path = handle_arguments()

    # === Initialise ===
    # cuda.profile_start()
    np.random.seed()

    # === Make archive ===
    archive = arch.Archive(archive_path, description_of_test, profile_state)
    if profile_state != arch.ProfileState.ARCHIVE:
        archive.new_archive_file()

        # === Scaled protocol ===
        experiment_time = "20211117T155508"#"20211117T123323"#"20210429T125734"
        scaled = util.ScaledParameters.new_from_experiment_time(experiment_time)
        # scaled = util.ScaledParameters(
        #     scaled_frequency = 5000,
        #     scaled_density = 1/50,
        #     scaled_samples = 10,
        #     scaled_amplitude = 500,#995.5/2,
        #     scaled_sweep = [5000/5, 14000],
        #     scaled_pulse_time_fraction = 0.2333333,
        #     # scaled_stagger_constant = math.sqrt(7)
        # )
        scaled.print()
        scaled.write_to_file(archive)

        # # print(f"{'freq_n':>10s} {'period_n':>10s} {'time_n':>10s} {'sig_dense':>10s} {'samp_num':>10s} {'freq_d_s':>10s} {'freq_d_e':>10s} {'dfreq_d':>10s} {'time_e':>10s}")
        # # print(f"{scaled.frequency:10.4e} {1/scaled.frequency:10.4e} {scaled.pulse_time:10.4e} {scaled.density:10.4e} {scaled.samples:10.4e} {scaled.sweep[0]:10.4e} {scaled.sweep[1]:10.4e} {scaled.frequency_step:10.4e} {scaled.time_end:10.4e}")

        # === Make signal ===
        # time_properties = test_signal.TimeProperties(5e-7, 1e-8, 1e-8, [0, 0.0001])
        # time_properties_reconstruction = test_signal.TimeProperties(5e-7, 1e-8, 1e-8, [0, 0.0001])

        # # time_properties = test_signal.TimeProperties(5e-7, 1e-7, 1e-8, [0, 0.1])
        # time_properties = test_signal.TimeProperties(5e-7, 1e-7, 1e-8, [0, 0.11])
        # time_properties_reconstruction = test_signal.TimeProperties(5e-7, 1e-7, 1e-8, [0, 0.1])

        time_properties = test_signal.TimeProperties(scaled.time_step, scaled.time_step/np.ceil(scaled.time_step/1e-7), 1e-8, [0, scaled.time_end + 0.02])
        time_properties_reconstruction = test_signal.TimeProperties(scaled.time_step, scaled.time_step/np.ceil(scaled.time_step/1e-7), 1e-8, [0, scaled.time_end])

        signal = test_signal.TestSignal(
            # [],
            # [test_signal.NeuralPulse(0.02333333, 70.0, 1000), test_signal.NeuralPulse(0.0444444444, 70.0, 1000)],
            # [test_signal.NeuralPulse(0.02333333, 70.0, 1000)],
            [test_signal.NeuralPulse(scaled.pulse_time, scaled.amplitude, scaled.frequency)],
            # [],
            [test_signal.SinusoidalNoise.new_line_noise([0.0, 0.0, 500])],
            # [test_signal.PeriodicNoise(amplitude = [0, 0, 1000], resolution = 3)],
            # [test_signal.PeriodicNoise.new_line_noise_sawtooth(amplitude = [0, 0, 1000], resolution = 3)],
            time_properties
        )
        signal_reconstruction = test_signal.TestSignal(
            # [],
            # [test_signal.NeuralPulse(0.02333333, 70.0, 1000), test_signal.NeuralPulse(0.0444444444, 70.0, 1000)],
            # [test_signal.NeuralPulse(0.02333333, 70.0, 1000)],
            [test_signal.NeuralPulse(scaled.pulse_time, scaled.amplitude, scaled.frequency)],
            [],
            # [test_signal.SinusoidalNoise.new_line_noise([0.0, 0.0, 500.0])],
            time_properties_reconstruction
        )

        # # === Make state ===
        # # [0.5, 1/np.sqrt(2), 0.5]
        # state_properties = sim.manager.StateProperties(spinsim.SpinQuantumNumber.ONE)
        # # state_properties = sim.manager.StateProperties(spinsim.SpinQuantumNumber.HALF)

        # cuda.profile_start()
        # # === Run simulations ===
        # # frequency = np.arange(70, 3071, 30)
        # # frequency = np.arange(250, 2251, 3)
        # # frequency = np.arange(250, 2000, 10.0)
        # # frequency = np.arange(250, 2251, 50)
        # # frequency = np.arange(250, 2251, 460e3/1e5)
        # # frequency = np.arange(990, 1010, 0.02)
        # # frequency = np.arange(253, 3251, 30)
        # # frequency = np.arange(1000, 1003, 1)
        # # frequency = np.arange(1000, 1001, 1)
        # # frequency = np.arange(0, 1000000, 1)
        # # frequency = np.arange(scaled.sweep[0], min(max(scaled.sweep[1], 0), scaled.samples*scaled.frequency/2), scaled.frequency_step) # ---- Scaled
        # frequency = scaled.sample_frequencies

        # simulation_manager = sim.manager.SimulationManager(signal, frequency, archive, state_properties = state_properties, measurement_method = sim.manager.MeasurementMethod.HARD_PULSE, signal_reconstruction = signal_reconstruction)
        # simulation_manager.evaluate(False, False)

        # === Experiment results ===
        # experiment_results = arch.ExperimentResults.new_from_simulation_manager(simulation_manager)
        # "20210429T125734"
        experiment_results = arch.ExperimentResults.new_from_archive_time(archive, experiment_time[0:15])
        # experiment_results.write_to_archive(archive)
        # experiment_results.plot(archive, signal_reconstruction)

        # === Make reconstructions ===
        reconstruction = recon.Reconstruction(signal_reconstruction.time_properties)
        # experiment_results = analysis.find_noise_size_from_rabi(experiment_results, scaled, archive)
        # experiment_results = analysis.add_shot_noise(experiment_results, scaled, archive, atom_count = 10e3, noise_modifier = 3)
        experiment_results.plot(archive, signal_reconstruction)
        # experiment_results.frequency_amplitude /= np.sqrt(2)
        experiment_results.write_to_archive(archive)
        reconstruction.read_frequencies_from_experiment_results(experiment_results, number_of_samples = min(10000, experiment_results.frequency.size), frequency_cutoff_low = 0, frequency_cutoff_high = 15e3, random_seed = util.Seeds.metroid)
        # reconstruction.read_frequencies_from_experiment_results(experiment_results, number_of_samples = min(150, experiment_results.frequency.size), frequency_cutoff_low = 0, frequency_cutoff_high = 14000, random_seed = util.Seeds.metroid)
        # reconstruction.read_frequencies_from_test_signal(signal_reconstruction, number_of_samples = 139)
        # reconstruction.evaluate_ista(
        #     expected_amplitude = scaled.amplitude,
        #     expected_frequency = scaled.frequency,
        #     expected_error_measurement = 4,#0.4,#0.25,#11.87e-2,
        #     norm_scale_factor_modifier = 3#0.0001
        # )
        # reconstruction.evaluate_ista_backtracking(
        #     expected_amplitude = scaled.amplitude,
        #     expected_frequency = scaled.frequency,
        #     expected_error_measurement = 11.87
        # )
        reconstruction.evaluate_fista_backtracking(
            expected_amplitude = scaled.amplitude,
            expected_frequency = scaled.frequency,
            expected_error_measurement = 6,#0.25,#11.87e-2,
            norm_scale_factor_modifier = 1#0.0001
        )
        # reconstruction.evaluate_fista_ayanzadeh(
        #     expected_amplitude = scaled.amplitude,
        #     expected_frequency = scaled.frequency,
        #     expected_error_measurement = 11.87/3
        # )
        # reconstruction.evaluate_fista_fit(
        #     expected_amplitude = scaled.amplitude,
        #     expected_frequency = scaled.frequency,
        #     expected_error_measurement = 11.87,
        #     norm_scale_factor_modifier = 0.0015,#0.002
        # )
        # reconstruction.evaluate_least_squares()
        # reconstruction.evaluate_fista()
        # reconstruction.evaluateISTAComplete()
        # reconstruction.evaluate_frequency_amplitude(signal_reconstruction)
        reconstruction.plot(archive, signal_reconstruction)
        reconstruction.write_to_file(archive.archive_file)


        # # === ===                       === ===
        # # === === Reconstruction sweeps === ===
        # # === ===                       === ===

        # # recon.run_reconstruction_subsample_sweep(
        # #     expected_signal = signal_reconstruction,
        # #     experiment_results = experiment_results,
        # #     sweep_parameters = (0, 10000, 10),
        # #     archive = archive,
        # #     random_seeds = np.arange(10)*util.Seeds.metroid,
        # #     evaluation_methods = ["least_squares"]
        # # )
        # # experiment_results = analysis.add_shot_noise(experiment_results, scaled, archive, atom_count = 10e3, noise_modifier = 3)
        # recon.run_reconstruction_subsample_sweep(
        #     expected_signal = signal_reconstruction,
        #     experiment_results = experiment_results,
        #     sweep_parameters = (0, 10000, 10),
        #     archive = archive,
        #     random_seeds = np.arange(10)*util.Seeds.metroid,
        #     evaluation_methods = ["least_squares", "fista_backtracking"],
        #     expected_amplitude = scaled.amplitude,
        #     expected_frequency = scaled.frequency,
        #     expected_error_measurement = 0.40,#0.25,#0.05,#0.2,#11.87,
        #     norm_scale_factor_modifier = 3,#0.001,
        #     frequency_line_noise = 50,
        #     rabi_frequency_readout = 2e3
        # )
        experiment_results = analysis.add_shot_noise(experiment_results, scaled, archive, atom_count = 10e3, noise_modifier = 3)
        recon.run_reconstruction_subsample_sweep(
            expected_signal = signal_reconstruction,
            experiment_results = experiment_results,
            sweep_parameters = (0, 10000, 10),
            archive = archive,
            random_seeds = np.arange(10)*util.Seeds.metroid,
            evaluation_methods = [
                "least_squares",
                "fista_backtracking"
            ],
            expected_amplitude = scaled.amplitude,
            expected_frequency = scaled.frequency,
            expected_error_measurement = 0.4, #4,#0.40,#0.25,#0.05,#0.2,#11.87,
            norm_scale_factor_modifier = 3,#3,#0.001,
            frequency_line_noise = 50,
            rabi_frequency_readout = 2e3
        )

        # # === ===          === ===
        # # === === Analysis === ===
        # # === ===          === ===

        # # analysis.find_time_blind_spots(scaled, archive)
        # # analysis.find_neural_signal_size(experiment_results, scaled, archive)
        # # analysis.find_line_noise_size(experiment_results, scaled, archive)
        # analysis.find_noise_size_from_rabi(experiment_results, scaled, archive)
        # # analysis.find_noise_size_from_fourier_transform(experiment_results, scaled, archive)

        # # === ===                 === ===
        # # === === Non compressive === ===
        # # === ===                 === ===
        # sim.ramsey.simulate_ramsey(scaled, archive)

        # === ===                      === ===
        # === === Benchmarks and tests === ===
        # === ===                      === ===

        # # === Time step fine test ===
        # # time_step_fine = [5e-9, 1e-8, 2e-8, 2.5e-8, 4e-8, 5e-8, 1e-7, 2e-7, 2.5e-7, 4e-7, 5e-7, 1e-6, 2e-6, 2.5e-6, 5e-6]
        # # time_step_fine = time_properties.time_step_coarse/np.floor(np.logspace(np.log10(200), np.log10(1), 50))
        # # frequency = np.arange(50, 3051, 300)
        # # time_step_fine = time_properties.time_step_coarse/np.floor(np.logspace(np.log10(200), np.log10(1), 10))
        # # frequency = np.arange(50, 3051, 300)
        # # frequency = np.arange(1000, 1003, 5)
        # # sim.benchmark.new_benchmark_time_step_fine(archive, signal, frequency, time_step_fine, state_properties)

        # # === Fit detuning ===
        # frequency = np.logspace(2.3, 5.3, 50)*0 + 10000
        # # util.fit_frequency_shift(archive, signal, frequency, state_properties)
        # detuning = np.linspace(-25, -15, frequency.size)
        # util.fit_frequency_detuning(archive, signal, frequency, detuning, state_properties)

        # # === Time test ===
        # frequency = np.arange(50, 3051, 1000)
        # sim.benchmark.new_benchmark_device(archive, signal, frequency, state_properties)

        # # === Device aggregate ===
        # # sim.benchmark.new_benchmark_device_aggregate(archive, ["20201208T132324", "20201214T183902", "20210521T131221"])
        # sim.benchmark.new_benchmark_device_aggregate(archive, ["20201208T132324", "20201214T183902", "20210414T111231", "20210521T131221", "20210531T170344"])

        # # === Time step source test ===
        # time_step_source = np.logspace(-9, -6, 50)
        # frequency = np.arange(50, 3051, 300)
        # # frequency = np.arange(1000, 1003, 5)
        # sim.benchmark.new_benchmark_time_step_source(archive, signal, frequency, state_properties, time_step_source)

        # === Benchmark comparison ===
        # # spinsim.benchmark.plot_benchmark_comparison(archive, ["20201113T173915", "20201113T202948", "20201113T204017", "20201113T205415", "20201113T210439", "20201113T211136"], ["CF4 RF", "CF4 LF", "HS RF", "HS LF", "MP RF", "MP LF"], "Effect of integration method on fine timestep benchmark (spin one)")

        # sim.benchmark.plot_benchmark_comparison(archive, ["20201116T110647", "20201116T111313", "20201116T111851", "20201116T112430", "20201116T112932", "20201116T113330"], ["CF4 RF", "CF4 LF", "HS RF", "HS LF", "MP RF", "MP LF"], "Effect of integration method on fine timestep benchmark\n(spin half, lie trotter)")

        # sim.benchmark.plot_benchmark_comparison(archive, ["20201119T181459", "20201119T181809", "20201119T182040", "20201119T182334", "20201119T182612", "20201119T182817"], ["CF4 RF", "CF4 LF", "HS RF", "HS LF", "MP RF", "MP LF"], "Effect of integration method on fine timestep benchmark\n(spin half, analytic)")

        # # === Trotter test ===
        # sim.benchmark.new_benchmark_trotter_cutoff_matrix(archive, np.arange(50, 0, -2), 1)
        # # frequency = np.arange(50, 3051, 300)
        # # frequency = np.arange(50, 3051, 30)
        # # newBenchmark_trotter_cutoff(archive, signal, frequency, np.arange(60, 0, -4))

        # === ===                     === ===
        # === === External benchmarks === ===
        # === ===                     === ===

        # time_step_fines = time_properties.time_step_coarse/np.floor(time_properties.time_step_coarse/np.asarray([1.e-9, 1.26896e-9, 1.61026e-9, 2.04336e-9, 2.59294e-9, 3.29034e-9, 4.17532e-9, 5.29832e-9, 6.72336e-9, 8.53168e-9, 1.08264e-8, 1.37382e-8, 1.74333e-8, 2.21222e-8, 2.80722e-8, 3.56225e-8, 4.52035e-8, 5.73615e-8, 7.27895e-8, 9.23671e-8, 1.1721e-7, 1.48735e-7, 1.88739e-7, 2.39503e-7, 3.0392e-7, 3.85662e-7, 4.8939e-7, 6.21017e-7, 7.88046e-7, 1.e-6]))
        # time_step_fines = [1.00200401e-09, 1.26903553e-09, 1.61290323e-09, 2.04918033e-09, 2.60416667e-09, 3.31125828e-09, 4.20168067e-09, 5.31914894e-09, 6.75675676e-09, 8.62068966e-09, 1.08695652e-08, 1.38888889e-08, 1.78571429e-08, 2.27272727e-08, 2.94117647e-08, 3.57142857e-08, 4.54545455e-08, 6.25000000e-08, 8.33333333e-08, 1.00000000e-07, 1.25000000e-07, 1.66666667e-07, 2.50000000e-07]
        # time_step_fines = [1.00200401e-09]
        # time_step_fines = [2.50000000e-07]
        # frequency = np.asarray([1000], dtype = np.float64)

        # # === SciPy Benchmark ===
        # # time_step_fine = time_properties.time_step_coarse/np.floor(np.logspace(np.log10(200), np.log10(1), 20))
        # frequency = np.asarray([1000], dtype = np.float64)
        # sim.benchmark.new_benchmark_external_scipy(archive, signal, frequency, time_step_fines, state_properties)

        # # === QuTip Benchmark ===
        # # time_step_fine = time_properties.time_step_coarse/np.floor(np.logspace(np.log10(200), np.log10(1), 20))
        # frequency = np.asarray([1000], dtype = np.float64)
        # sim.benchmark.new_benchmark_external_qutip(archive, signal, frequency, time_step_fines, state_properties)

        # # === Mathematica Benchmark ===
        # time_step_fines = [1.e-9, 1.26896e-9, 1.61026e-9, 2.04336e-9, 2.59294e-9, 3.29034e-9, 4.17532e-9, 5.29832e-9, 6.72336e-9, 8.53168e-9, 1.08264e-8, 1.37382e-8, 1.74333e-8, 2.21222e-8, 2.80722e-8, 3.56225e-8, 4.52035e-8, 5.73615e-8, 7.27895e-8, 9.23671e-8, 1.1721e-7, 1.48735e-7, 1.88739e-7, 2.39503e-7, 3.0392e-7, 3.85662e-7, 4.8939e-7, 6.21017e-7, 7.88046e-7, 1.e-6]
        # errors = [0, 1.27648e-7, 1.71067e-7, 3.52869e-7, 1.10268e-7, 4.21367e-7, 4.86085e-7, 2.79164e-7, 2.02119e-6, 1.54471e-6, 1.62444e-6, 3.49334e-6, 0.0000156213, 0.0000836096, 0.000454502, 0.00249248, 0.00503788, 0.00503797, 0.00510717, 0.0051067, 0.00510602, 0.00510511, 0.00510396, 0.00510275, 0.0051019, 0.0051023, 0.00510479, 0.00510821, 0.00510659, 0.00510244]
        # execution_times = [631.373, 810.259, 553.893, 394.797, 290.008, 216.053, 163.82, 145.432, 115.548, 90.8332, 72.04, 56.8771, 44.3481, 35.5904, 28.4812, 22.2169, 20.1843, 20.2888, 20.1585, 20.2421, 20.1293, 20.0051, 20.0887, 20.2273, 20.0593, 20.1271, 20.2015, 20.1939, 20.1278, 20.1355]
        # sim.benchmark.new_benchmark_mathematica(archive, time_step_fines, errors, execution_times)

        # # === Spinsim Benchmark ===
        # frequency = np.asarray([1000], dtype = np.float64)
        # sim.benchmark.new_benchmark_true_external_spinsim(archive, frequency, time_step_fines, device = spinsim.Device.CPU)
        # # sim.benchmark.new_benchmark_external_spinsim(archive, signal, frequency, time_step_fines, state_properties)
        # # sim.benchmark.new_benchmark_external_spinsim(archive, signal, frequency, time_step_fines, state_properties, device = spinsim.Device.CPU)
        # # sim.benchmark.new_benchmark_true_external_internal_spinsim(archive, frequency, time_step_fines)
        # # sim.benchmark.new_benchmark_internal_spinsim(signal, frequency, time_step_fines, state_properties)
        # # sim.benchmark.new_benchmark_internal_trotter_spinsim(signal, frequency, time_step_fines, state_properties)
        # # sim.benchmark.new_benchmark_time_step_fine(archive, signal, frequency, time_step_fines, state_properties)
        # # sim.benchmark.new_benchmark_spinsim(archive, signal, frequency, time_step_fines, state_properties)

        # # === Comparison ===
        # # sim.benchmark.new_benchmark_external_evaluation(archive, ["20210720T145449", "20210720T150101", "20210617T163718", "20210507T140423", "20210618T090427", "20210507T124849"], reference_name = "SciPy", title = "Comparison to alternative software")
        # sim.benchmark.new_benchmark_external_evaluation(archive, ["20210720T145449", "20210720T150101", "20210617T163718", "20210507T140423", "20210618T090427"], reference_name = "SciPy", title = "Comparison to alternative software")
        # # # ---- Trotter test ----:
        # # sim.benchmark.new_benchmark_external_evaluation(archive, ["20210510T152208", "20210510T152349", "20210510T152537", "20210510T152730", "20210510T152930", "20210510T153133", "20210510T153342", "20210510T153554", "20210510T153811", "20210510T154034", "20210510T154302", "20210510T154531", "20210510T154807", "20210510T155046", "20210510T155330", "20210510T155620", "20210504T175150"], reference_name = "SciPy", is_external = False)
        # # ---- Spin one ----:
        # sim.benchmark.new_benchmark_external_evaluation(archive, ["20210720T140531", "20210720T140811", "20210720T141042", "20210720T141225", "20210720T141401", "20210720T141647", "20210504T175150"], reference_name = "SciPy", is_external = False, title = "spinsim options")
        # # ---- Spin half (analytic) ----
        # sim.benchmark.new_benchmark_external_evaluation(archive, ["20210720T141916", "20210720T142003", "20210720T142042", "20210720T142120", "20210720T142151", "20210720T142236", "20210511T115900"], reference_name = "SciPy", is_external = False, title = "spinsim options")
        # # # ---- Spin half (trotter) ----
        # # sim.benchmark.new_benchmark_external_evaluation(archive, ["20210607T170836", "20210607T170949", "20210607T171054", "20210607T171208", "20210607T171312", "20210607T171412", "20210511T115900"], reference_name = "SciPy", is_external = False, title = "spinsim options (Lie Trotter)")
        # # # # sim.benchmark.new_benchmark_external_evaluation(archive, ["20210507T165913", "20210507T170105", "20210507T170256", "20210507T170456", "20210507T170646", "20210507T170822", "20210504T175150"], reference_name = "SciPy", is_external = False)
        # # # # sim.benchmark.plot_benchmark_comparison(archive, ["20210423T181745", "20210422T091436", "20210422T090233"], ["ss", "sp", "sp (h)", "mm", "mm (h)"], "Comparison of alternative integration packages")

        # === Clean up ===
        archive.close_archive_file()
        cuda.profile_stop()
        cuda.close()
        plt.show()