import numpy as np
import matplotlib.pyplot as plt
import time as tm
from numba import cuda

from simulation import *
from benchmarkResults import *

class SimulationManager:
    """
    Controls a set of simulations running for different dressing parameters.

    Attributes
    ----------
    signal : `list` of :class:`testSignal.TestSignal`
        A list of a signal objects containing information describing the magnetic environment of the simulation, as well as timing information for the simulation. :func:`evaluate()` will run simulations for all of these values.
    trotterCutoff : `list` of `int`
        A list of the number of squares to be used in the matrix exponentiation algorithm :func:`simulationUtilities.matrixExponentialLieTrotter()` during the simulation. :func:`evaluate()` will run simulations for all of these values.
    frequency : :class:`numpy.ndarray` of :class:`numpy.double`, (frequencyIndex)
        A list of dressing rabi frequencies for the spin system. In units of Hz. :func:`evaluate()` will run simulations for all of these values.
    frequencyAmplitude : :class:`numpy.ndarray` of :class:`numpy.double`, (simulationIndex)
        The Fourier coefficient of the signal as measured from the simulation `simulationIndex`. Filled in after the simulation `simulationIndex` has been completed.
    archive : :class:`archive.Archive`
        The archive object to save the simulation results and parameters to.
    stateOutput : `list` of (`numpy.ndarray` of  `numpy.cdouble`, (timeIndex, stateIndex))
        An optional `list` to directly write the state of the simulation to. Used for benchmarks. Reference can be optionally passed in on construction.
    stateProperties : :class:`StateProperties`
            The :class:`StateProperties` initial conditions for the wavefunction of the quantum system.
    """
    def __init__(self, signal, frequency, archive, stateProperties = None, stateOutput = None, trotterCutoff = [28]):
        """
        Parameters
        ----------
        signal : :class:`testSignal.TestSignal` or `list` of :class:`testSignal.TestSignal`
            A list of a signal objects containing information describing the magnetic environment of the simulation, as well as timing information for the simulation. :func:`evaluate()` will run simulations for all of these values.
        frequency : :class:`numpy.ndarray` of :class:`numpy.double`, (frequencyIndex)
            A list of dressing rabi frequencies for the spin system. In units of Hz. :func:`evaluate()` will run simulations for all of these values.
        archive : :class:`archive.Archive`
            The archive object to save the simulation results and parameters to.
        stateProperties : :class:`StateProperties`
            The :class:`StateProperties` initial conditions for the wavefunction of the quantum system.
        stateOutput : `list` of (`numpy.ndarray` of  `numpy.cdouble`, (timeIndex, stateIndex)), optional
            An optional `list` to directly write the state of the simulation to. Used for benchmarks.
        trotterCutoff : `list` of `int`, optional
            A list of the number of squares to be used in the matrix exponentiation algorithm :func:`simulationUtilities.matrixExponentialLieTrotter()` during the simulation. :func:`evaluate()` will run simulations for all of these values.
        """
        self.signal = signal
        if not isinstance(self.signal, list):
            self.signal = [self.signal]
        self.trotterCutoff = np.asarray(trotterCutoff)
        self.frequency = np.asarray(frequency, dtype = np.float64)
        self.frequencyAmplitude = np.empty(self.frequency.size*len(self.signal)*self.trotterCutoff.size, dtype = np.float64)
        self.archive = archive
        self.stateOutput = stateOutput
        self.stateProperties = stateProperties

    def evaluate(self, doPlot = False, doWriteEverything = False):
        """
        Evaluates the prepared set of simulations. Fills out the :class:`numpy.ndarray`, :attr:`frequencyAmplitude`. The simulation `simulationIndex` will be run with the frequency given by `frequencyIndex` mod :attr:`frequency.size`, the signal given by floor(`signalIndex` / `frequency.size`) mod len(`signal`), and the trotter cutoff given by floor(`signalIndex` / `frequency.size` / `trotterCutoff.size`).
        doWriteEverything : `boolean`, optional
            If `True`, then save all time series data to file as well as parametric data. Defaults to `False` to reduce archive file size.
        Parameters
        ----------
        doPlot : `boolean`, optional
            If `True`, plots time series of the expected spin values in each direction during execution.
        """
        print("\033[33mStarting simulations...\033[0m")
        executionTimeEndPoints = np.empty(2)
        executionTimeEndPoints[0] = tm.time()
        executionTimeEndPoints[1] = executionTimeEndPoints[0]
        print("Idx\tCmp\tTm\tdTm")
        archiveGroupSimulations = self.archive.archiveFile.require_group("simulations")
        for trotterCutoffIndex, trotterCutoffInstance in enumerate(self.trotterCutoff):
            for signalIndex, signalInstance in enumerate(self.signal):
                for frequencyIndex in range(self.frequency.size):
                    simulationIndex = frequencyIndex + (signalIndex + trotterCutoffIndex*len(self.signal))*self.frequency.size
                    frequencyValue = self.frequency[frequencyIndex]
                    simulation = Simulation(signalInstance, frequencyValue, self.stateProperties, trotterCutoffInstance)
                    simulation.getFrequencyAmplitudeFromDemodulation([0.09, 0.1], doPlot)
                    simulation.writeToFile(archiveGroupSimulations.require_group("simulation" + str(simulationIndex)), doWriteEverything)
                    self.frequencyAmplitude[simulationIndex] = simulation.simulationResults.sensedFrequencyAmplitude
                    if self.stateOutput is not None:
                        self.stateOutput += [simulation.simulationResults.state]
                    print("{:4d}\t{:3.0f}%\t{:3.0f}s\t{:2.3f}s".format(simulationIndex, 100*(simulationIndex + 1)/(self.frequency.size*len(self.signal)*self.trotterCutoff.size), tm.time() - executionTimeEndPoints[0], tm.time() - executionTimeEndPoints[1]))
                    executionTimeEndPoints[1] = tm.time()
        print("\033[32mDone!\033[0m")