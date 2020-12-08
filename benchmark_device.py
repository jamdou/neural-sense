import numpy as np
from numba import cuda  # GPU code
import colorama         # Colourful terminal
colorama.init()

# The different pieces that make up this sensing code
import archive as arch              # Saving results and configurations
from archive import handle_arguments

import test_signal                  # The properties of the magnetic signal, used for simulations and reconstructions
import spinsim                      # Main simulation package

import sim                          # Simulation code

if __name__ == "__main__":
    # This will be recorded in the HDF5 file to give context for what was being tested
    description_of_test = "CPU Benchmark"

    profile_state, archive_path = handle_arguments()

    # Initialise
    np.random.seed()

    # Make archive
    archive = arch.Archive(archive_path, description_of_test, profile_state)
    if profile_state != arch.ProfileState.ARCHIVE:
        archive.new_archive_file()

        # Make signal
        time_properties = test_signal.TimeProperties(5e-7, 1e-7, 1e-8, [0, 0.1])
        signal = test_signal.TestSignal(
            [test_signal.NeuralPulse(0.02333333, 10.0, 1000), test_signal.NeuralPulse(0.0444444444, 10.0, 1000)],
            [],
            time_properties
        )
        signal.write_to_file(archive.archive_file)

        # Make state
        state_properties = sim.manager.StateProperties(spinsim.SpinQuantumNumber.ONE)

        cuda.profile_start()

        # Time test
        frequency = np.arange(50, 3051, 30)
        sim.benchmark.new_benchmark_device(archive, signal, frequency, state_properties)

        # Clean up
        archive.close_archive_file()
        cuda.profile_stop()
        cuda.close()