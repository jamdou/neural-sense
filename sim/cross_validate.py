import numpy as np
import archive as arch
import spinsim

class ResultsCompilation:
    def __init__(self, frequency:np.ndarray, frequency_amplitudes:np.ndarray, pulse_times:np.ndarray, dc_detunings:np.ndarray):
        self.frequency = frequency.copy()
        self.frequency_amplitudes = frequency_amplitudes.copy()
        self.pulse_times = pulse_times
        self.dc_detunings = dc_detunings
    
    def write_to_file(self, archive:arch.Archive):
        archive_group = archive.archive_file.require_group("cross_validation/results_compilation")
        archive_group["frequency"] = self.frequency
        archive_group["frequency_amplitudes"] = self.frequency_amplitudes
        archive_group["pulse_times"] = self.pulse_times

    def generate_inputs(self, frequency_max = 10e3, frequency_step = 100, time_end = 5e-3, number_of_experiments = 720, number_of_pulses_max = 2, detuning_std = 100):
        frequency = frequency_step + np.arange(0, frequency_max, frequency_step)