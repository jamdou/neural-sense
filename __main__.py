import numpy as np
import matplotlib.pyplot as plt
# import sys, getopt      # Command line arguments
from numba import cuda  # GPU code
import colorama         # Colourful terminal
colorama.init()

# The different pieces that make up this sensing code
import archive as arch              # Saving results and configurations
from archive import handleArguments

import testSignal                   # The properties of the magnetic signal, used for simulations and reconstructions
import spinsim                      # Main simulation package
import reconstruction as recon      # Uses compressive sensing to reconstruct the a magnetic signal

if __name__ == "__main__":
    # This will be recorded in the HDF5 file to give context for what was being tested
    descriptionOfTest = "Arbitrary input"

    # Check to see if there is a compatible GPU
    if cuda.list_devices():
        print("\033[32mUsing cuda device " + str(cuda.list_devices()[0].name) + "\033[0m")
    else:
        print("\033[31mNo cuda devices found. System is incompatible. Exiting...\033[0m")
        exit()

    profileState, archivePath = handleArguments()

    # Initialise
    # cuda.profile_start()
    np.random.seed()

    # Make archive
    archive = arch.Archive(archivePath, descriptionOfTest, profileState)
    if profileState != arch.ProfileState.ARCHIVE:
        archive.newArchiveFile()

        # Make signal
        timeProperties = testSignal.TimeProperties(5e-8, 1e-8, [0, 0.01])
        signal = testSignal.TestSignal(
            [],
            # [testSignal.NeuralPulse(0.02333333, 10.0, 1000), testSignal.NeuralPulse(0.0444444444, 10.0, 1000)],
            # [NeuralPulse(0.02333333, 10.0, 1000)],
            [],
            # [SinusoidalNoise.newDetuningNoise(10)],
            timeProperties
        )
        signal.writeToFile(archive.archiveFile)

        # Make state
        # [0.5, 1/math.sqrt(2), 0.5]
        stateProperties = spinsim.simulation.StateProperties(spinsim.simulation.SpinQuantumNumber.ONE)

        cuda.profile_start()

        # # Time step test
        # # timeStepFine = [5e-9, 1e-8, 2e-8, 2.5e-8, 4e-8, 5e-8, 1e-7, 2e-7, 2.5e-7, 4e-7, 5e-7, 1e-6, 2e-6, 2.5e-6, 5e-6]
        # timeStepFine = timeProperties.timeStepCoarse/np.floor(np.logspace(np.log10(200), np.log10(1), 50))
        # frequency = np.arange(50, 3051, 300)
        # # frequency = np.arange(1000, 1003, 5)
        # newBenchmarkTimeStepFine(archive, signal, frequency, timeStepFine)

        # plotBenchmarkComparison(archive, ["20201016T120914", "20201016T121113", "20201016T120556", "20201016T121414", "20201016T113809", "20201016T121721", "20201016T122146"], ["tc = 12", "tc = 16", "tc = 20", "tc = 24", "tc = 28", "tc = 32", "tc = 36"], "Effect of trotter cutoff on timestep benchmark")

        # Trotter Test
        # newBenchmarkTrotterCutoffMatrix(archive, np.arange(80, 0, -4), 1e1)
        # frequency = np.arange(50, 3051, 300)
        # frequency = np.arange(50, 3051, 30)
        # newBenchmarkTrotterCutoff(archive, signal, frequency, np.arange(60, 0, -4))
        
        # Run simulations
        # frequency = np.arange(50, 3051, 3)
        frequency = np.arange(1000, 1003, 1)
        simulationManager = spinsim.simulationManager.SimulationManager(signal, frequency, archive, stateProperties)
        simulationManager.evaluate(True, False)
        # experimentResults = ExperimentResults(simulationManager.frequency, simulationManager.frequencyAmplitude)
        experimentResults = spinsim.experimentResults.ExperimentResults.newFromSimulationManager(simulationManager)
        experimentResults.writeToArchive(archive)
        experimentResults.plot(archive, signal)

        # # Make reconstructions
        # reconstruction = recon.Reconstruction(signal.timeProperties)
        # reconstruction.readFrequenciesFromExperimentResults(experimentResults)
        # # reconstruction.readFrequenciesFromTestSignal(signal)
        # reconstruction.evaluateFISTA()
        # # reconstruction.evaluateISTAComplete()
        # reconstruction.plot(archive, signal)
        # reconstruction.writeToFile(archive.archiveFile)

        # Clean up
        archive.closeArchiveFile()
        cuda.profile_stop()
        cuda.close()