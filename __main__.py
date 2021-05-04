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

if __name__ == "__main__":
    # This will be recorded in the HDF5 file to give context for what was being tested
    description_of_test = "Adiabatic sweep"

    # Check to see if there is a compatible GPU
    if cuda.list_devices():
        print("\033[32mUsing cuda device {}\033[0m".format(cuda.list_devices()[0].name.decode('UTF-8')))
    else:
        print("\033[31mNo cuda devices found. System is incompatible. Exiting...\033[0m")
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
        scaled_frequency = 5000
        scaled_density = 1/100
        scaled_samples = 10
        scaled_amplitude = 800
        # scaled_sweep = [scaled_frequency/5, 14000]
        scaled_sweep = [2000, 7000]
        scaled_time_step = 1/(scaled_frequency*scaled_samples)
        # scaled_sweep = [0, scaled_samples*scaled_density/2]

        scaled_time_end = 1/(scaled_frequency*scaled_density)
        scaled_pulse_time = 0.2333333*scaled_time_end
        scaled_frequency_step = scaled_density*scaled_frequency/2
        # scaled_frequency_step = scaled_density*scaled_frequency*2

        print(f"{'freq_n':>10s} {'period_n':>10s} {'time_n':>10s} {'sig_dense':>10s} {'samp_num':>10s} {'freq_d_s':>10s} {'freq_d_e':>10s} {'dfreq_d':>10s} {'time_e':>10s}")
        print(f"{scaled_frequency:10.4e} {1/scaled_frequency:10.4e} {scaled_pulse_time:10.4e} {scaled_density:10.4e} {scaled_samples:10.4e} {scaled_sweep[0]:10.4e} {scaled_sweep[1]:10.4e} {scaled_frequency_step:10.4e} {scaled_time_end:10.4e}")

        # === Make signal ===
        # time_properties = test_signal.TimeProperties(5e-7, 1e-8, 1e-8, [0, 0.0001])
        # time_properties_reconstruction = test_signal.TimeProperties(5e-7, 1e-8, 1e-8, [0, 0.0001])

        # time_properties = test_signal.TimeProperties(5e-7, 1e-7, 1e-8, [0, 0.1])
        # time_properties_reconstruction = test_signal.TimeProperties(5e-7, 1e-7, 1e-8, [0, 0.1])

        time_properties = test_signal.TimeProperties(5e-7, 1e-7, 1e-8, [0, scaled_time_end + 0.02])
        time_properties_reconstruction = test_signal.TimeProperties(scaled_time_step, 1e-7, 1e-8, [0, scaled_time_end])

        signal = test_signal.TestSignal(
            # [],
            # [test_signal.NeuralPulse(0.02333333, 70.0, 1000), test_signal.NeuralPulse(0.0444444444, 70.0, 1000)],
            # [test_signal.NeuralPulse(0.02333333, 70.0, 1000)],
            [test_signal.NeuralPulse(scaled_pulse_time, scaled_amplitude, scaled_frequency)],
            # [],
            [test_signal.SinusoidalNoise.new_line_noise([0.0, 0.0, 500.0])],
            # [test_signal.PeriodicNoise(amplitude = [0, 0, 1000], resolution = 200)],
            time_properties
        )
        signal_reconstruction = test_signal.TestSignal(
            # [],
            # [test_signal.NeuralPulse(0.02333333, 70.0, 1000), test_signal.NeuralPulse(0.0444444444, 70.0, 1000)],
            # [test_signal.NeuralPulse(0.02333333, 70.0, 1000)],
            [test_signal.NeuralPulse(scaled_pulse_time, scaled_amplitude, scaled_frequency)],
            [],
            # [test_signal.SinusoidalNoise.new_line_noise([0.0, 0.0, 500.0])],
            time_properties_reconstruction
        )

        # === Make state ===
        # [0.5, 1/np.sqrt(2), 0.5]
        state_properties = sim.manager.StateProperties(spinsim.SpinQuantumNumber.ONE)

        cuda.profile_start()
        
        # === Run simulations ===
        # frequency = np.arange(70, 3071, 30)
        # frequency = np.arange(250, 2251, 3)
        # frequency = np.arange(250, 2000, 10.0)
        # frequency = np.arange(250, 2251, 50)
        # frequency = np.arange(250, 2251, 460e3/1e5)
        # frequency = np.arange(990, 1010, 0.02)
        # frequency = np.arange(253, 3251, 30)
        # frequency = np.arange(1000, 1003, 1)
        # frequency = np.arange(1000, 1001, 1)
        # frequency = np.arange(0, 1000000, 1)
        frequency = np.arange(scaled_sweep[0], min(max(scaled_sweep[1], 0), scaled_samples*scaled_frequency/2), scaled_frequency_step)

        simulation_manager = sim.manager.SimulationManager(signal, frequency, archive, state_properties = state_properties, measurement_method = sim.manager.MeasurementMethod.HARD_PULSE, signal_reconstruction = signal_reconstruction)
        simulation_manager.evaluate(False, False)

        # === Experiment results ===
        experiment_results = arch.ExperimentResults.new_from_simulation_manager(simulation_manager)
        # experiment_results = arch.ExperimentResults.new_from_archive_time(archive, "20210504T111910")
        experiment_results.write_to_archive(archive)
        experiment_results.plot(archive, signal_reconstruction)

        # === Make reconstructions ===
        reconstruction = recon.Reconstruction(signal_reconstruction.time_properties)
        reconstruction.read_frequencies_from_experiment_results(experiment_results, number_of_samples = min(130, experiment_results.frequency.size), frequency_cutoff_low = 0, frequency_cutoff_high = 14000)
        # reconstruction.read_frequencies_from_test_signal(signal_reconstruction, number_of_samples = 139)
        reconstruction.evaluate_ista()
        # reconstruction.evaluate_fista()
        # reconstruction.evaluateISTAComplete()
        reconstruction.plot(archive, signal_reconstruction)
        reconstruction.write_to_file(archive.archive_file)

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
        # sim.benchmark.new_benchmark_device_aggregate(archive, ["20201208T132324", "20201214T183902"])

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
        # sim.benchmark.new_benchmark_trotter_cutoff_matrix(archive, np.arange(80, 0, -4), 1e1)
        # # frequency = np.arange(50, 3051, 300)
        # # frequency = np.arange(50, 3051, 30)
        # # newBenchmark_trotter_cutoff(archive, signal, frequency, np.arange(60, 0, -4))

        # === ===                     === ===
        # === === External benchmarks === ===
        # === ===                     === ===

        # time_step_fines = time_properties.time_step_coarse/np.floor(time_properties.time_step_coarse/np.asarray([1.e-9, 1.26896e-9, 1.61026e-9, 2.04336e-9, 2.59294e-9, 3.29034e-9, 4.17532e-9, 5.29832e-9, 6.72336e-9, 8.53168e-9, 1.08264e-8, 1.37382e-8, 1.74333e-8, 2.21222e-8, 2.80722e-8, 3.56225e-8, 4.52035e-8, 5.73615e-8, 7.27895e-8, 9.23671e-8, 1.1721e-7, 1.48735e-7, 1.88739e-7, 2.39503e-7, 3.0392e-7, 3.85662e-7, 4.8939e-7, 6.21017e-7, 7.88046e-7, 1.e-6]))
        # frequency = np.asarray([1000], dtype = np.float64)

        # # === SciPy Benchmark ===
        # # time_step_fine = time_properties.time_step_coarse/np.floor(np.logspace(np.log10(200), np.log10(1), 20))
        # sim.benchmark.new_benchmark_scipy(archive, signal, frequency, time_step_fines, state_properties)

        # # === Mathematica Benchmark ===
        # time_step_fines = [1.e-9, 1.26896e-9, 1.61026e-9, 2.04336e-9, 2.59294e-9, 3.29034e-9, 4.17532e-9, 5.29832e-9, 6.72336e-9, 8.53168e-9, 1.08264e-8, 1.37382e-8, 1.74333e-8, 2.21222e-8, 2.80722e-8, 3.56225e-8, 4.52035e-8, 5.73615e-8, 7.27895e-8, 9.23671e-8, 1.1721e-7, 1.48735e-7, 1.88739e-7, 2.39503e-7, 3.0392e-7, 3.85662e-7, 4.8939e-7, 6.21017e-7, 7.88046e-7, 1.e-6]
        # errors = [0, 1.27648e-7, 1.71067e-7, 3.52869e-7, 1.10268e-7, 4.21367e-7, 4.86085e-7, 2.79164e-7, 2.02119e-6, 1.54471e-6, 1.62444e-6, 3.49334e-6, 0.0000156213, 0.0000836096, 0.000454502, 0.00249248, 0.00503788, 0.00503797, 0.00510717, 0.0051067, 0.00510602, 0.00510511, 0.00510396, 0.00510275, 0.0051019, 0.0051023, 0.00510479, 0.00510821, 0.00510659, 0.00510244]
        # execution_times = [631.373, 810.259, 553.893, 394.797, 290.008, 216.053, 163.82, 145.432, 115.548, 90.8332, 72.04, 56.8771, 44.3481, 35.5904, 28.4812, 22.2169, 20.1843, 20.2888, 20.1585, 20.2421, 20.1293, 20.0051, 20.0887, 20.2273, 20.0593, 20.1271, 20.2015, 20.1939, 20.1278, 20.1355]
        # sim.benchmark.new_benchmark_mathematica(archive, time_step_fines, errors, execution_times)

        # # === Spinsim Benchmark ===
        # frequency = np.asarray([1000, 1000], dtype = np.float64)
        # # sim.benchmark.new_benchmark_time_step_fine(archive, signal, frequency, time_step_fines, state_properties)
        # sim.benchmark.new_benchmark_spinsim(archive, signal, frequency, time_step_fines, state_properties)

        # === Comparison ===
        # sim.benchmark.plot_benchmark_comparison(archive, ["20210423T181745", "20210422T091436", "20210422T090233"], ["ss", "sp", "sp (h)", "mm", "mm (h)"], "Comparison of alternative integration packages")

        # === Clean up ===
        archive.close_archive_file()
        cuda.profile_stop()
        cuda.close()
        plt.show()