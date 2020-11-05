import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
import h5py

class BenchmarkType(Enum):
    """
    An enum to define the type of benchmark being done. Each gives labels and plot parameters to allow for the plotting and arching code to be modular.

    Parameters
    ----------
    _value_ : `string`
        Label for the archive.
    xLabel : `string`
        Horizontal label for when plotting.
    yLabel : `string`
        Vertical label for when plotting.
    title : `string`
        Title for when plotting.
    xScale : `string`
        The type of scaling to apply to the x axis for when plotting. Either `"linear"` for a linear scale, or `"log"` for a log scale.
    """
    def __init__(self, value, xLabel, yLabel, title, xScale):
        super().__init__()
        self._value_ = value
        self.xLabel = xLabel
        self.yLabel = yLabel
        self.title = title
        self.xScale = xScale

    NONE = (
        "none",
        "Nothing (rad)",
        "RMS error",
        "Nothing",
        "log"
    )
    """
    No benchmark has been defined.
    """

    TIME_STEP_SOURCE = (
        "timeStepSource",
        "Source time step (s)",
        "RMS error",
        "Effect of source time step size on RMS error",
        "log"
    )
    """
    The results of :func:`benchmark.manager.newBenchmarkTrotterCutoff()`.
    """

    TROTTER_CUTOFF = (
        "trotterCutoff",
        "Trotter cutoff",
        "RMS error",
        "Effect of trotter cutoff on RMS error",
        "linear"
    )
    """
    The results of :func:`benchmark.manager.newBenchmarkTrotterCutoff()`.
    """

    TROTTER_CUTOFF_MATRIX = (
        "trotterCutoffMatrix",
        "Trotter cutoff",
        "RMS error",
        "Effect of trotter cutoff on RMS error",
        "linear"
    )
    """
    The results of :func:`benchmark.manager.newBenchmarkTrotterCutoffMatrix()`.
    """

    TIME_STEP_FINE = (
        "timeStepFine",
        "Fine time step (s)",
        "RMS error",
        "Effect of fine time step size on RMS error",
        "log"
    )
    """
    The results of :func:`benchmark.manager.newBenchmarkTimeStepFine()`.
    """

    TIME_STEP_FINE_FREQUENCY_DRIFT = (
        "timeStepFineFrequencyDrift",
        "Fine time step (s)",
        "Frequency shift (Hz)",
        "Effect of fine time step size on frequency shift",
        "log"
    )
    """
    The results of :func:`benchmark.manager.newBenchmarkTimeStepFineFrequencyDrift()`.
    """

class BenchmarkResults:
    """
    A class that holds the results of an arbitrary benchmark, and has the ability to plot them.

    Attributes
    ----------
    benchmarkType : :class:`BenchmarkType`
        The benchmark that this was the result of. Also contains information used to archive and plot the results.
    parameter : :class:`numpy.ndarray`
        The value of the parameter being varied during the benchmark.
    error : :class:`numpy.ndarray`
        The error recorded during the benchmark.
    """
    def __init__(self, benchmarkType = BenchmarkType.NONE, parameter = None, error = None):
        """
        Parameters
        ----------
        benchmarkType : :class:`BenchmarkType`, optional
            The benchmark that this was the result of. Also contains information used to archive and plot the results. Defaults to :obj:`BenchmarkType.NONE`.
        parameter : :class:`numpy.ndarray`
            The value of the parameter being varied during the benchmark. Defaults to `None`.
        error : :class:`numpy.ndarray`
            The error recorded during the benchmark. Defaults to `None`.
        """
        self.benchmarkType = benchmarkType
        self.parameter = parameter
        self.error = error

    @staticmethod
    def readFromArchive(archive):
        """
        A constructor that reads a new benchmark result from a hdf5 file.

        Parameters
        ----------
        archive : :class:`archive.Archive`
            The archive object to read the benchmark from.
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
        Save a benchmark to a hdf5 file.

        Parameters
        ----------
        archive : :class:`archive.Archive`
            The archive object to write the benchmark to.
        """
        archiveGroupBenchmarkResults = archive.archiveFile.require_group("benchmarkResults/" + self.benchmarkType.value)
        archiveGroupBenchmarkResults[self.benchmarkType.value] = self.parameter
        archiveGroupBenchmarkResults["error"] = self.error

    def plot(self, archive = None, doShowPlot = True):
        """
        Plots the benchmark results.

        Parameters
        ----------
        archive : :class:`archive.Archive`, optional
            If specified, will save plots to the archive's `plotPath`.
        doShowPlot : `boolean`, optional
            If `True`, will attempt to show and save the plots. Can be set to false to overlay multiple archive results to be plotted later, as is done with :func:`benchmarkManager.plotBenchmarkComparison()`.
        """
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
            if archive:
                plt.title(archive.executionTimeString + "\n" + self.benchmarkType.title)
                plt.savefig(archive.plotPath + "benchmark" + self.benchmarkType.value[0].capitalize() + self.benchmarkType.value[1:] + ".pdf")
                plt.savefig(archive.plotPath + "benchmark" + self.benchmarkType.value[0].capitalize() + self.benchmarkType.value[1:] + ".png")
            plt.show()
