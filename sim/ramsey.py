import math
import numpy as np
import matplotlib.pyplot as plt
import spinsim

import util
from util import PrettyTritty as C
import archive as arch

def simulate_ramsey(scaled:util.ScaledParameters, archive:arch.Archive = None):
    bias = 600e3
    duration_sensing = 60e-6
    amplitude_pulse = 20e3
    duration_pulse = 1/(4*amplitude_pulse)
    time_neural = scaled.pulse_time
    frequency_neural = scaled.frequency
    amplitude_neural = scaled.amplitude
    amplitude_line = 500
    # amplitude_line = 0
    def ramsey_hamiltonian(time_sample, parameters, field_strength):
        time_sensing = parameters[0]
        if (time_sensing - duration_sensing/2 - duration_pulse <= time_sample) and (time_sample < time_sensing - duration_sensing/2):
            field_strength[0] = 2*math.tau*amplitude_pulse*math.cos(math.tau*bias*(time_sample))
        elif (time_sensing + duration_sensing/2 < time_sample) and (time_sample <= time_sensing + duration_sensing/2 + duration_pulse):
            field_strength[0] = 2*math.tau*amplitude_pulse*math.cos(math.tau*bias*(time_sample) + math.pi/2)
        else:
            field_strength[0] = 0
        field_strength[1] = 0
        field_strength[2] = math.tau*bias
        field_strength[2] += math.tau*amplitude_line*math.sin(math.tau*50*time_sample)
        if time_neural <= time_sample and time_sample < time_neural + 1/frequency_neural:
            field_strength[2] += math.tau*amplitude_neural*math.sin(math.tau*frequency_neural*(time_sample - time_neural))
        field_strength[3] = 0
    time = np.arange(0, scaled.time_end, scaled.time_step)
    amplitude = np.empty_like(time)
    simulator = spinsim.Simulator(ramsey_hamiltonian, spinsim.SpinQuantumNumber.ONE)
    C.starting("Ramsey simulations")
    C.print(f"|{'Index':8}|{'Perc':8}|")
    for time_index, time_sensing in enumerate(time):
        results = simulator.evaluate(0, scaled.time_end, scaled.time_step/100, scaled.time_step, spinsim.SpinQuantumNumber.ONE.plus_z, [time_sensing])
        amplitude[time_index] = results.spin[-1, 2]/(duration_sensing*math.tau)
        C.print(f"|{time_index:8d}|{(time_index + 1)/time.size*100:7.2f}%|", end = "\r")
    C.print("\n")
    C.finished("Ramsey simulations")
    if archive:
        ramsey_group = archive.archive_file.create_group("simulations_ramsey")
        ramsey_group["amplitude"] = amplitude
        ramsey_group["time"] = time
    plt.figure()
    plt.plot(time, amplitude, "k-")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (Hz)")
    if archive:
        archive.write_plot("Ramsey sampling", "ramsey_sampling")
    plt.draw()
