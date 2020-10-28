import numpy as np
import matplotlib.pyplot as plt
import h5py

class ExperimentResults:
    """
    A class that contains information from the results of a complete experiment, whether it was done via simulations, or in the lab.

    Attributes
    ----------
    frequency : :class:`numpy.ndarray` of :class:`numpy.double`
        The dressing Rabi frequencies at which the experiments were run. In units of Hz.
    frequencyAmplitude : :class:`numpy.ndarray` of :class:`numpy.double`
        The Fourier sine coefficient of the signal being measured, as determined by the experiment. In units of Hz.
    """
    def __init__(self, frequency = None, frequencyAmplitude = None):
        """
        Parameters
        ----------
        frequency : :class:`numpy.ndarray` of :class:`numpy.double`, optional
            The dressing Rabi frequencies at which the experiments were run. In units of Hz.
        frequencyAmplitude : :class:`numpy.ndarray` of :class:`numpy.double`, optional
            The Fourier sine coefficient of the signal being measured, as determined by the experiment. In units of Hz.
        """
        self.frequency = frequency
        self.frequencyAmplitude = frequencyAmplitude

    @staticmethod
    def newFromSimulationManager(simulationManager):
        """
        A constructor that creates a new :class:`ExperimentResults` object from an already evaluated :class:`simulationManager.SimulationManager` object. That is, make an experiment results object based off of simulation results.

        Parameters
        ----------
        simulationManager : :class:`simulationManager.SimulationManager`
            The simulation manager object to read the results of.

        Returns
        -------
        experimentResults : :class:`ExperimentResults`
            A new object containing the results of `simulationManager`.
        """
        frequency = 1*simulationManager.frequency
        frequencyAmplitude = 1*simulationManager.frequencyAmplitude[:frequency.size]

        return ExperimentResults(frequency, frequencyAmplitude)

    def plot(self, archive = None, testSignal = None):
        """
        Plots the experiment results contained in the object. Optionally saves the plots, as well as compares the results to numerically calculated values.

        Parameters
        ----------
        archive : :class:`archive.Archive`, optional
            If specified, will save the generated plot to the archive's :attr:`archive.Archive.plotPath`.
        testSignal : :class:`testSignal:TestSignal`, optional
            The signal that was being measured during the experiment. If specified, this function will plot the sine Fourier transform of the signal behind the measured coefficients of the experiment results.
        """
        plt.figure()
        if testSignal:
            plt.plot(testSignal.frequency, testSignal.frequencyAmplitude, "-k")
        plt.plot(self.frequency, self.frequencyAmplitude, "xr")
        if testSignal:
            plt.legend(["Fourier Transform", "Measured"])
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude (Hz)")
        plt.xlim([0, 2000])
        # plt.ylim([-0.08, 0.08])
        plt.grid()
        if archive:
            plt.title(archive.executionTimeString + "Measured Frequency Amplitude")
            plt.savefig(archive.plotPath + "measuredFrequencyAmplitude.pdf")
            plt.savefig(archive.plotPath + "measuredFrequencyAmplitude.png")
        plt.show()

    def writeToArchive(self, archive):
        """
        Saves the contents of the experiment results to an archive.

        Parameters
        ----------
        archive : :class:`archive.archive`
            The archive object to save the results to.
        """
        archiveGroupExperimentResults = archive.archiveFile.require_group("experimentResults")
        archiveGroupExperimentResults["frequency"] = self.frequency
        archiveGroupExperimentResults["frequencyAmplitude"] = self.frequencyAmplitude