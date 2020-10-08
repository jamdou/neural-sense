import numpy as np
import matplotlib.pyplot as plt
import time as tm
from simulation import *
from numba import cuda

class SimulationManager:
    """
    Controls a set of simulations running, for different dressing parameters
    """
    def __init__(self, signal, frequency, archive, stateOutput = None, trotterCutoff = [48]):
        self.signal = signal
        if not isinstance(self.signal, list):
            self.signal = [self.signal]
        self.trotterCutoff = np.asarray(trotterCutoff)
        self.frequency = np.asarray(frequency, dtype = np.float64)
        self.frequencyAmplitude = np.empty(self.frequency.size*len(self.signal)*self.trotterCutoff.size, dtype = np.float64)
        self.archive = archive
        self.stateOutput = stateOutput

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
        for trotterCutoffIndex, trotterCutoffInstance in enumerate(self.trotterCutoff):
            for signalIndex, signalInstance in enumerate(self.signal):
                for frequencyIndex in range(self.frequency.size):
                    simulationIndex = frequencyIndex + (signalIndex + trotterCutoffIndex*len(self.signal))*self.frequency.size
                    frequencyValue = self.frequency[frequencyIndex]
                    simulation = Simulation(signalInstance, frequencyValue, trotterCutoff = trotterCutoffInstance)
                    simulation.getFrequencyAmplitudeFromDemodulation([0.09, 0.1])
                    simulation.writeToFile(archiveGroupSimulations.create_group("simulation" + str(simulationIndex)))
                    self.frequencyAmplitude[simulationIndex] = simulation.simulationResults.sensedFrequencyAmplitude
                    if self.stateOutput is not None:
                        self.stateOutput += [simulation.simulationResults.state]
                    print("{:4d}\t{:3.0f}%\t{:3.0f}s\t{:2.3f}s".format(simulationIndex, 100*(simulationIndex + 1)/(self.frequency.size*len(self.signal)*self.trotterCutoff.size), tm.time() - executionTimeEndPoints[0], tm.time() - executionTimeEndPoints[1]))
                    executionTimeEndPoints[1] = tm.time()
        print("\033[32mDone!\033[0m")

    @staticmethod
    def newTrotterCutoffComparison(signal, frequency, archive, trotterCutoff):
        stateOutput = []
        error = []
        simulationManager = SimulationManager(signal, frequency, archive, stateOutput, trotterCutoff)
        simulationManager.evaluate()
        for trotterCutoffIndex in range(trotterCutoff.size):
            errorTemp = 0
            for frequencyIndex in range(frequency.size):
                stateDifference = stateOutput[frequencyIndex + trotterCutoffIndex*frequency.size] - stateOutput[frequencyIndex]
                errorTemp += np.sum(np.sqrt(np.real(np.conj(stateDifference)*stateDifference)))
            error += [errorTemp/(frequency.size*stateOutput[0].size)]
        
        trotterCutoff = np.asarray(trotterCutoff)
        error = np.asarray(error)

        archiveGroupErrorFromTrotterCutoff = archive.archiveFile.create_group("errorFromTrotterCutoff")
        archiveGroupErrorFromTrotterCutoff["cutoff"] = trotterCutoff
        archiveGroupErrorFromTrotterCutoff["error"] = error

        plt.figure()
        plt.grid()
        plt.plot(trotterCutoff, error, "rx")
        plt.yscale("log")
        plt.xlabel("Trotter cutoff")
        plt.ylabel("RMS error (sqrt(probability))")
        plt.title(archive.executionTimeString + " Effect of trotter cutoff on RMS error")
        plt.savefig(archive.plotPath + "errorFromTrotterCutoff.pdf")
        plt.savefig(archive.plotPath + "errorFromTrotterCutoff.png")
        plt.show()

    @staticmethod
    def newFineStepComparisonFrequencyDrift(archive, signalTemplate, timeStepFines, dressingFrequency = np.arange(500, 1500, 10)):
        dressingFrequency = np.asarray(dressingFrequency)
        signalTemplate.getAmplitude()
        signalTemplate.getFrequencyAmplitude()

        signals = []
        for timeStepFine in timeStepFines:
            timeProperties = TimeProperties(signalTemplate.timeProperties.timeStepCoarse, timeStepFine)
            signal = TestSignal(signalTemplate.neuralPulses, timeProperties)
            signals += [signal]

        simulationManager = SimulationManager(signals, dressingFrequency, archive)
        simulationManager.evaluate()

        frequencyDrift = np.zeros(len(signals))
        for signalIndex in range(len(signals)):
            for frequencyIndex, frequency in enumerate(dressingFrequency):
                frequencyDrift[signalIndex] += np.abs(simulationManager.frequencyAmplitude[frequencyIndex + signalIndex*dressingFrequency.size] - signalTemplate.frequencyAmplitude[signalTemplate.frequency == frequency])
            frequencyDrift[signalIndex] /= dressingFrequency.size
        archiveGroupTimeStepDrift = archive.archiveFile.create_group("timeStepDrift")
        archiveGroupTimeStepDrift["timeStepFines"] = np.asarray(timeStepFines)
        archiveGroupTimeStepDrift["frequencyDrift"] = frequencyDrift

        plt.figure()
        plt.plot(timeStepFines, frequencyDrift, "rx")
        plt.xscale("log")
        plt.xlabel("Time step size (s)")
        plt.ylabel("Drift of measured frequency (Hz)")
        plt.title(archive.executionTimeString + "Drift from time step")
        plt.grid()
        plt.savefig(archive.plotPath + "driftFromTimeStep.pdf")
        plt.savefig(archive.plotPath + "driftFromTimeStep.png")
        plt.show()