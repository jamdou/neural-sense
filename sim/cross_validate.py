import numpy as np
import math, cmath
import archive as arch
import spinsim

import test_signal

class ResultsCompilation:
    def __init__(self, frequency:np.ndarray, frequency_amplitudes:np.ndarray, pulse_times:np.ndarray, dc_detunings:np.ndarray, time_properties:test_signal.TimeProperties):
        self.frequency = frequency
        self.frequency_amplitudes = frequency_amplitudes
        self.pulse_times = pulse_times
        self.dc_detunings = dc_detunings
        self.time_properties = time_properties
    
    def write_to_file(self, archive:arch.Archive):
        archive_group = archive.archive_file.require_group("cross_validation/results_compilation")
        archive_group["frequency"] = self.frequency
        archive_group["frequency_amplitudes"] = self.frequency_amplitudes
        archive_group["pulse_times"] = self.pulse_times

    @staticmethod
    def generate_inputs(frequency_max = 10e3, frequency_step = 100, time_end = 5e-3, number_of_experiments = 720, number_of_pulses_max = 2, detuning_std = 100, pulse_duration = 1/5e3):
        frequency = frequency_step + np.arange(0, frequency_max, frequency_step)
        dc_detunings = np.random.normal(0, detuning_std, (number_of_experiments, frequency.size))
        time_step =time_end/(frequency_max/frequency_step)
        time_properties = test_signal.TimeProperties(
            time_step_coarse = time_step,
            time_end_points = [time_step, time_end]
        )

        pulse_times = np.random.uniform(size = (number_of_experiments, number_of_pulses_max))
        number_of_pulses = np.random.randint(0, number_of_pulses_max + 1, number_of_experiments)
        pulse_times *= time_end/number_of_pulses.reshape((pulse_times.shape[0], 1)) - pulse_duration
        pulse_times[:, 1] += pulse_times[:, 0] + pulse_duration
        pulse_times[number_of_pulses < 2, 1] = -1
        pulse_times[number_of_pulses < 1, 0] = -1
        
        return ResultsCompilation(frequency, np.empty_like(dc_detunings), pulse_times, dc_detunings, time_properties)

    def simulate(self):
        def get_field(time, parameters, field):
            dressing = parameters[0]
            dc_detuning = parameters[1]

            pulse_time_0 = parameters[2]
            pulse_time_1 = parameters[3]

            time_end = 5e-3
            readout_amplitude = 20e3
            neural_amplitude = 400.0
            neural_frequency = 5e3


            field[0] = math.tau*(600e3 + dc_detuning)
            if time > pulse_time_0 and time < pulse_time_0 + 1/neural_frequency:
                field[0] += math.tau*neural_amplitude*math.sin(math.tau*neural_frequency*(time - pulse_time_0))
            if time > pulse_time_1 and time < pulse_time_1 + 1/neural_frequency:
                field[0] += math.tau*neural_amplitude*math.sin(math.tau*neural_frequency*(time - pulse_time_1))
            
            field[1] = 0
            if time < time_end:
                field[1] += math.tau*2*dressing*math.cos(math.tau*600e3*time)
            elif time < time_end + (1/readout_amplitude)/4:
                field[1] += math.tau*2*readout_amplitude*math.sin(math.tau*600e3*time)

            field[2] = 0
            field[3] = 0

        simulator = spinsim.Simulator(get_field, spinsim.SpinQuantumNumber.ONE, threads_per_block = 256)
        result = simulator.evaluate(0, 5.1e-3, 1e-7, 1e-6, spinsim.SpinQuantumNumber.ONE.minus_z, [self.frequency[0], self.dc_detunings[0, 0], self.pulse_times[0, 0], self.pulse_times[0, 1]])
        self.frequency_amplitudes[0, 0] = (1/(5e-3*math.tau))*(np.abs(result.state[-1, 2]**2) - np.abs(result.state[-1, 0]**2))

        print(self.frequency_amplitudes[0, 0])
