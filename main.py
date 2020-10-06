import numpy as np
import matplotlib.pyplot as plt
import sys, getopt      # Command line arguments
from numba import cuda  # GPU code

# The different pieces that make up this sensing code
from archive import *
from testSignal import *
from simulation import *
from simulationManager import *
from experimentResults import *
from reconstruction import *

# Making the terminal all colourful because that's what I live for
import colorama
colorama.init()

# This will be recorded in the HDF5 file to give context for what was being tested
descriptionOfTest = "Making readable for others"

#===============================================================#

"""
=== A QUICK EXPLANATION OF GPUS AND NUMBA ===

So I learnt this all pretty recently, so I may still have some
misunderstandings here, but hopefully it gives a good, dense
summary of the important parts.

GPU functions that are called from a CPU, are called KERNELS.
NUMBA is a Just In Time (JIT) compiler that can compile python
code into kernels. GPUs as a whole are also known as STREAMING
MULTIPROCESSORS (SMs). Also, the CPU is sometimes called the
HOST, with the GPU it controls called the DEVICE.

An instance of a kernel program running on a GPU is called a
THREAD. There can be thousands of threads of the same kernel
running on a SM at once, each running the same code. Each
thread is given a unique index, which lets it know which part
of a larger array it should work on (you decide what that array
is). Groups of 32 SM hardware CORES run threads in sync with each
other. These groups are called WARPS. All threads in a warp are
finished executing before they are switched out for more threads.
In software, threads with similar indices are organised into
groups called BLOCKS. Warps are filled with threads of the same
block. The array of all blocks to be executed is called a GRID.

When executing a kernel, you must specify the size of blocks
(in terms of threads) and the size of the grid (in terms of
blocks). In numba, these are written in square brackets after the
function name and before the function arguments.

kernel[blocksPerGrid, threadsPerBlock](arguments)

If the size of the array to be worked on does not fit in nicely
with the block and grid sizes, then you need to check whether the
kernel index is accessing somewhere outside the array boundary or
not, or else other memory might be overwritten.

Kernels cannot return values, but any arrays that are passed in
as arguments can be modified by the kernel.

Each SM has a set number of REGISTERS (fast memory) that can be
shared between all cores. The number of occupied cores can be
increased by limiting the number of registers used per thread
with the max_registers option in the cuda.jit decorator. However,
Doing so too much may also slow down code since it will be using
slower memory more often. Threads each have their own LOCAL
MEMORY that only they can access, and SHARED MEMORY that is
available to all other threads. The arguments to the kernel are
all shared.

DEVICE FUNCTIONS can also be written. These are functions that
are executed within a kernel thread (ie on the device), and
cannot be called by the CPU (ie the host).

Numba can also be used to write JIT compiled code for the CPU.
prange can be used as an alternative to range in for loops to be
able to take advantage of CPU parallel processing such as
multiple cores, or vector coprocessors.
"""

#===============================================================#
helpMessage = """
\033[36m-h  --help      \033[33mShow this message
\033[36m-a  --archive   \033[33mSelect an alternate archive path.
                \033[32mDefault:
                    \033[36m.\\archive\033[0m
\033[36m-p  --profile   \033[33mSelect what type of nvprof profiling to be done,
                from:
                    \033[36mnone \033[32m(default)  \033[33mRun normally
                    \033[36mtimeline        \033[33mSave timeline
                    \033[36mmetric          \033[33mSave metrics
                    \033[36marchive         \033[33mArchive results,
                                    don't run anything
                                    else
                \033[35mOnly used for automation with profiling, if
                you're not doing this, then leave this blank.\033[0m
"""
#===============================================================#

if __name__ == "__main__":
    # Check to see if there is a compatible GPU
    if cuda.list_devices():
        print("\033[32mUsing cuda device " + str(cuda.list_devices()[0].name) + "\033[0m")
    else:
        print("\033[31mNo cuda devices found. System is incompatible. Exiting...\033[0m")
        exit()

    # Command line arguments. Probably don't worry too much about these. Mostly used for profiling.
    profileState = "None"
    archivePath = ".\\archive\\"
    options, arguments = getopt.getopt(sys.argv[1:], "hpa", ["help", "profile=", "archive="])
    for option, argument in options:
        if option in ("--help", "-h"):
            print(helpMessage)
            exit()
        elif option in ("--profile", "-p"):
            if argument == "timeline":
                profileState = "TimeLine"
            elif argument == "metric":
                profileState = "Metric"
            elif argument == "archive":
                profileState = "Archive"
        elif option in ("--archive", "-a"):
            archivePath = argument + "\\"

    # Initialise
    cuda.profile_start()
    np.random.seed()

    # Make archive
    archive = Archive(archivePath, descriptionOfTest, profileState)
    if profileState != "Archive":
        archive.newArchiveFile()

        # Make signal
        signal = TestSignal(
            [NeuralPulse(0.02333333, 10.0, 1000), NeuralPulse(0.0444444444, 10.0, 1000)],
            # [NeuralPulse(0.02333333, 10.0, 1000)],
            TimeProperties(1e-6)
        )
        signal.writeToFile(archive.archiveFile)

        # Run simulations
        frequency = np.arange(50, 3051, 3)
        # frequency = np.arange(1000, 1003, 1)
        simulationManager = SimulationManager(signal, frequency, archive)
        simulationManager.evaluate()
        experimentResults = ExperimentResults(simulationManager.frequency, simulationManager.frequencyAmplitude)
        experimentResults.writeToArchive(archive)
        experimentResults.plot(archive, signal)

        # Make reconstructions
        reconstruction = Reconstruction(signal.timeProperties)
        reconstruction.readFrequenciesFromExperimentResults(experimentResults)
        # reconstruction.readFrequenciesFromTestSignal(signal)
        reconstruction.evaluateFISTA()
        reconstruction.plot(archive, signal)
        reconstruction.writeToFile(archive.archiveFile)

        # Clean up
        archive.closeArchiveFile()
        cuda.profile_stop()
        cuda.close()