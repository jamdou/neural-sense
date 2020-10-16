import numpy as np
import matplotlib.pyplot as plt
import time as tm
from numba import cuda

from simulation import *
from benchmarkResults import *

class SimulationManager:
    """
    Controls a set of simulations running, for different dressing parameters
    """
    def __init__(self, signal, frequency, archive, stateOutput = None, trotterCutoff = [36]):
        self.signal = signal
        if not isinstance(self.signal, list):
            self.signal = [self.signal]
        self.trotterCutoff = np.asarray(trotterCutoff)
        self.frequency = np.asarray(frequency, dtype = np.float64)
        self.frequencyAmplitude = np.empty(self.frequency.size*len(self.signal)*self.trotterCutoff.size, dtype = np.float64)
        self.archive = archive
        self.stateOutput = stateOutput

    def evaluate(self, doPlot = False):
        """
        Evaluates the prepared set of simulations
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
                    simulation = Simulation(signalInstance, frequencyValue, trotterCutoff = trotterCutoffInstance)
                    simulation.getFrequencyAmplitudeFromDemodulation([0.09, 0.1], doPlot)
                    simulation.writeToFile(archiveGroupSimulations.require_group("simulation" + str(simulationIndex)))
                    self.frequencyAmplitude[simulationIndex] = simulation.simulationResults.sensedFrequencyAmplitude
                    if self.stateOutput is not None:
                        self.stateOutput += [simulation.simulationResults.state]
                    print("{:4d}\t{:3.0f}%\t{:3.0f}s\t{:2.3f}s".format(simulationIndex, 100*(simulationIndex + 1)/(self.frequency.size*len(self.signal)*self.trotterCutoff.size), tm.time() - executionTimeEndPoints[0], tm.time() - executionTimeEndPoints[1]))
                    executionTimeEndPoints[1] = tm.time()
        print("\033[32mDone!\033[0m")