import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
import h5py

class BenchmarkType(Enum):
    """
    An enum (label) to define the type of benchmark being done.
    Each benchmark type has different labels and titles for plots,
    for example, but they are all so similar that otherwise I'd be
    copying and pasting the same code over and over again.
    """
    def __init__(self, value, xLabel, yLabel, title, xScale):
        super().__init__()
        self._value_ = value    # Label for HDF5 archive
        self.xLabel = xLabel    # Plot horizontal label
        self.yLabel = yLabel    # Plot vertical label
        self.title = title      # Plot title
        self.xScale = xScale    # Whether the x axis has a linear or log scale

    NONE = (
        "none",
        "Nothing (rad)",
        "RMS error",
        "Nothing",
        "log"
    )
    TROTTER_CUTOFF = (
        "trotterCutoff",
        "Trotter cutoff",
        "RMS error",
        "Effect of trotter cutoff on RMS error",
        "linear"
    )
    TROTTER_CUTOFF_MATRIX = (
        "trotterCutoffMatrix",
        "Trotter cutoff",
        "RMS error",
        "Effect of trotter cutoff on RMS error",
        "linear"
    )
    TIME_STEP_FINE = (
        "timeStepFine",
        "Fine time step (s)",
        "RMS error",
        "Effect of fine time step size on RMS error",
        "log"
    )
    TIME_STEP_FINE_FREQUENCY_DRIFT = (
        "timeStepFineFrequencyDrift",
        "Fine time step (s)",
        "Frequency shift (Hz)",
        "Effect of fine time step size on frequency shift",
        "log"
    )

class BenchmarkResults:
    def __init__(self, benchmarkType = BenchmarkType.NONE, parameter = None, error = None):
        self.benchmarkType = benchmarkType
        self.parameter = parameter
        self.error = error

    @staticmethod
    def readFromArchive(archive):
        """
        Open a benchmark from a hdf5 file
        """
        archiveGroupBenchmark = archive.archiveFile["benchmarkResults"]
        for benchmarkType in BenchmarkType:
            if benchmarkType.value in archiveGroupBenchmark.keys():
                archiveGroupBenchmarkResults = archiveGroupBenchmark[benchmarkType.value]
                benchmarkResults = BenchmarkResults(
                    benchmarkType,
                    archiveGroupBenchmarkResults[benchmarkType.value],
                    archiveGroupBenchmarkResults["error"]
                )
                return benchmarkResults

    def writeToArchive(self, archive):
        """
        Save a benchmark to a hdf5 file
        """
        archiveGroupBenchmarkResults = archive.archiveFile.require_group("benchmarkResults/" + self.benchmarkType.value)
        archiveGroupBenchmarkResults[self.benchmarkType.value] = self.parameter
        archiveGroupBenchmarkResults["error"] = self.error

    def plot(self, archive, doShowPlot = True):
        if doShowPlot:
            plt.figure()
            plt.plot(self.parameter[1:], self.error[1:], "rx--")
        else:
            plt.plot(self.parameter[1:], self.error[1:], "x--")
        plt.grid()
        plt.yscale("log")
        plt.xscale(self.benchmarkType.xScale)
        plt.xlabel(self.benchmarkType.xLabel)
        plt.ylabel(self.benchmarkType.yLabel)
        if doShowPlot:
            plt.title(archive.executionTimeString + "\n" + self.benchmarkType.title)
            plt.savefig(archive.plotPath + "benchmark" + self.benchmarkType.value[0].capitalize() + self.benchmarkType.value[1:] + ".pdf")
            plt.savefig(archive.plotPath + "benchmark" + self.benchmarkType.value[0].capitalize() + self.benchmarkType.value[1:] + ".png")
            plt.show()
