import numpy as np
import matplotlib.pyplot as plt
import sys, getopt      # Command line arguments
from numba import cuda  # GPU code

# The different pieces that make up this sensing code
from archive import *                   # Saving results and configurations
from testSignal import *                # The properties of the magnetic signal, used for simulations and reconstructions

from simulation import *                # Simulates an 3 state atom interacting with a magnetic signal
from simulationUtilities import *       # Tools used in the simulation
from simulationManager import *         # Controls the running of multiple simulations
from experimentResults import *         # A list of frequency coefficients from a simulation or actual experiment

from reconstruction import *            # Uses compressive sensing to reconstruct the a magnetic signal

from benchmarkManager import *          # Runs benchmarks to test accuracy of code in different configurations
from benchmarkResults import *          # Stores the results of, and plots benchmarks

# Making the terminal all colourful because that's what I live for
import colorama

if __name__ == "__main__":
    colorama.init()

    # This will be recorded in the HDF5 file to give context for what was being tested
    descriptionOfTest = "Spin 1/2"

    helpMessage = """
    \033[36m-h  --help      \033[33mShow this message
    \033[36m-a  --archive   \033[33mSelect an alternate archive path.
                    \033[32mDefault:
                        \033[36m.\\archive\033[0m
    \033[36m-p  --profile   \033[33mSelect what type of nvprof profiling to be
                    done, from:
                        \033[36mnone \033[32m(default)  \033[33mRun normally
                        \033[36mtimeline            \033[33mSave timeline
                        \033[36mmetric              \033[33mSave metrics
                        \033[36minstructionlevel    \033[33mSave per instruction
                                            metrics
                        \033[36marchive             \033[33mArchive results,
                                            don't run anything
                                            else
                    \033[35mOnly used for automation with profiling, if
                    you're not doing this, then leave this blank.\033[0m
    """
    # Check to see if there is a compatible GPU
    if cuda.list_devices():
        print("\033[32mUsing cuda device " + str(cuda.list_devices()[0].name) + "\033[0m")
    else:
        print("\033[31mNo cuda devices found. System is incompatible. Exiting...\033[0m")
        exit()

    # Command line arguments. Probably don't worry too much about these. Mostly used for profiling.
    profileState = ProfileState.NONE
    archivePath = ".\\archive\\"
    options, arguments = getopt.getopt(sys.argv[1:], "hpa", ["help", "profile=", "archive="])
    for option, argument in options:
        if option in ("--help", "-h"):
            print(helpMessage)
            exit()
        elif option in ("--profile", "-p"):
            if argument == "timeline":
                profileState = ProfileState.TIME_LINE
            elif argument == "metric":
                profileState = ProfileState.METRIC
            elif argument == "instructionlevel":
                profileState = ProfileState.INSTRUCTION_LEVEL
            elif argument == "archive":
                profileState = ProfileState.ARCHIVE
        elif option in ("--archive", "-a"):
            archivePath = argument + "\\"

    # Initialise
    # cuda.profile_start()
    np.random.seed()

    # Make archive
    archive = Archive(archivePath, descriptionOfTest, profileState)
    if profileState != ProfileState.ARCHIVE:
        archive.newArchiveFile()

        # Make signal
        timeProperties = TimeProperties(5e-7, 1e-7)
        signal = TestSignal(
            [NeuralPulse(0.02333333, 10.0, 1000), NeuralPulse(0.0444444444, 10.0, 1000)],
            # [NeuralPulse(0.02333333, 10.0, 1000)],
            [],
            # [SinusoidalNoise.newDetuningNoise(10)],
            timeProperties
        )
        signal.writeToFile(archive.archiveFile)

        # Make state
        stateProperties = StateProperties(SpinQuantumNumber.HALF)

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
        frequency = np.arange(50, 3051, 3)
        # frequency = np.arange(1000, 1003, 1)
        simulationManager = SimulationManager(signal, frequency, archive, stateProperties)
        simulationManager.evaluate(False, False)
        # experimentResults = ExperimentResults(simulationManager.frequency, simulationManager.frequencyAmplitude)
        experimentResults = ExperimentResults.newFromSimulationManager(simulationManager)
        experimentResults.writeToArchive(archive)
        experimentResults.plot(archive, signal)

        # Make reconstructions
        reconstruction = Reconstruction(signal.timeProperties)
        reconstruction.readFrequenciesFromExperimentResults(experimentResults)
        # reconstruction.readFrequenciesFromTestSignal(signal)
        reconstruction.evaluateFISTA()
        # reconstruction.evaluateISTAComplete()
        reconstruction.plot(archive, signal)
        reconstruction.writeToFile(archive.archiveFile)

        # Clean up
        archive.closeArchiveFile()
        cuda.profile_stop()
        cuda.close()