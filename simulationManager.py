import numpy as np
import matplotlib.pyplot as plt
import time as tm
from simulation import *
from numba import cuda

class SimulationManager:
    """
    Controls a set of simulations running, for different dressing parameters
    """
    def __init__(self, signal, frequency, archive):
        self.signal = signal
        self.frequency = np.asarray(frequency, dtype = np.float64)
        self.frequencyAmplitude = np.empty_like(frequency, dtype = np.float64)
        self.archive = archive

    def evaluate(self):
        """
        Evaluates the prepared set of simulations
        """
        print("\033[33mStarting simulations...\033[0m")
        executionTimeEndPoints = np.empty(2)
        executionTimeEndPoints[0] = tm.time()
        executionTimeEndPoints[1] = executionTimeEndPoints[0]
        print("Idx\tCmp\tTm\tdTm")
        archiveGroupSimulations = self.archive.archiveFile.create_group("simulations")
        for frequencyIndex in range(self.frequency.size):
            frequencyValue = self.frequency[frequencyIndex]
            simulation = Simulation(self.signal, frequencyValue)
            simulation.getFrequencyAmplitudeFromDemodulation([0.09, 0.1])
            simulation.writeToFile(archiveGroupSimulations.create_group("simulation" + str(frequencyIndex)))
            self.frequencyAmplitude[frequencyIndex] = simulation.simulationResults.sensedFrequencyAmplitude
            print("{:4d}\t{:3.0f}%\t{:3.0f}s\t{:2.3f}s".format(frequencyIndex, 100*(frequencyIndex + 1)/self.frequency.size, tm.time() - executionTimeEndPoints[0], tm.time() - executionTimeEndPoints[1]))
            executionTimeEndPoints[1] = tm.time()
        print("\033[32mDone!\033[0m")