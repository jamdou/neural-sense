import numpy as np
import numba.cuda as cuda
import math
import matplotlib.pyplot as plt
import test_signal
from sim import manager

class C:
    d = "\033[0m"
    y = "\033[33m"
    g = "\033[32m"

def fit_frequency_shift(archive, signal, frequency, state_properties, do_plot = True, do_plot_individuals = False):
    """

    """
    spin_output = []
    error = []
    time_coarse = signal.time_properties.time_coarse

    simulation_manager = manager.SimulationManager(signal, frequency, archive, state_properties, spin_output = spin_output)
    simulation_manager.evaluate(False)

    shift = 0*frequency
    shift_predict = 0*frequency

    print(f"{C.y}Starting shift analysis...{C.d}")  

    for frequency_index in range(frequency.size):
        spin = (spin_output[frequency_index])[:, 2]
        frequency_dressing = frequency[frequency_index]
        shift[frequency_index], shift_predict[frequency_index] = fit_frequency(frequency_dressing, spin, time_coarse)

    if archive:
        archive_group_frequency_shift = archive.archive_file.require_group("diagnostics/frequency_shift")
        archive_group_frequency_shift["frequency"] = frequency
        archive_group_frequency_shift["frequency_shift"] = shift
        archive_group_frequency_shift["frequency_shift_predict"] = shift_predict

    print(f"{C.g}Done!{C.d}\a")

    if do_plot:
        plt.figure()
        plt.loglog(frequency, shift_predict, "k+")
        plt.loglog(frequency, shift, "rx")
        plt.xlabel("Dressing frequency (Hz)")
        plt.ylabel("Frequency shift (Hz)")
        plt.legend(("Expected", "Calculated"))
        if archive:
            plt.title(f"{archive.execution_time_string}\nFrequency shift diagnostics")
            plt.savefig(f"{archive.plot_path}diagnostics_frequency_shift.pdf")
            plt.savefig(f"{archive.plot_path}diagnostics_frequency_shift.png")
        plt.show()
    return

def fit_frequency_detuning(archive, signal, frequency, detuning, state_properties, do_plot = True, do_plot_individuals = False):
    """

    """
    spin_output = []
    error = []
    time_coarse = signal.time_properties.time_coarse

    simulation_manager = manager.SimulationManager(signal, detuning, archive, state_properties, spin_output = spin_output, measurement_method = manager.MeasurementMethod.HARD_PULSE_DETUNING_TEST)
    simulation_manager.evaluate(False)

    shift = 0*frequency
    shift_predict = 0*frequency

    print(f"{C.y}Starting shift analysis...{C.d}")  

    for frequency_index in range(frequency.size):
        spin = (spin_output[frequency_index])[:, 2]
        frequency_dressing = frequency[frequency_index]
        shift[frequency_index], shift_predict[frequency_index] = fit_frequency(frequency_dressing, spin, time_coarse, do_plot_individuals)

    if archive:
        archive_group_frequency_shift = archive.archive_file.require_group("diagnostics/frequency_shift_detuning")
        archive_group_frequency_shift["detuning"] = detuning
        archive_group_frequency_shift["frequency_shift"] = shift
        archive_group_frequency_shift["frequency_shift_predict"] = shift_predict

    print(f"{C.g}Done!{C.d}\a")

    if do_plot:
        plt.figure()
        plt.plot(detuning, shift_predict, "k+")
        plt.plot(detuning, shift, "rx")
        plt.xlabel("Detuning (Hz)")
        plt.ylabel("Frequency shift (Hz)")
        plt.legend(("Expected", "Calculated"))
        if archive:
            plt.title(f"{archive.execution_time_string}\nFrequency shift detuning diagnostics")
            plt.savefig(f"{archive.plot_path}diagnostics_frequency_shift_detuning.pdf")
            plt.savefig(f"{archive.plot_path}diagnostics_frequency_shift_detuning.png")
        plt.show()

    return

def fit_frequency(frequency_dressing, spin, time_coarse, do_plot = False):
    frequency_guess = math.sqrt(frequency_dressing**2 + (0.25*(frequency_dressing**2)/460e3)**2)
    frequency_guess_previous = frequency_guess
    frequency_step = 1e-1*1e3

    dc_guess = 0.0
    dc_guess_previous = dc_guess
    dc_step = 1e-3*1e3

    amplitude_guess = 1.0
    amplitude_guess_previous = amplitude_guess
    amplitude_step = 1e-3*1e3

    phase_guess = 0.0
    phase_guess_previous = phase_guess
    phase_step = 1e-2*1e3

    # # Correlation
    # print("Dressing:\t{:1.7f}".format(frequency_guess))
    # print("{:s}\t{:s}\t{:s}\t\t{:s}\t{:s}".format("index", "frequency", "left", "correlation", "right"))
    # correlation_max = 0
    # step_size = 1
    # step_index = 0
    # while step_size > 1e-5:
    #     correlation = 2*np.average(spin*np.cos(math.tau*frequency_guess*time_coarse))
    #     correlation_left = 2*np.average(spin*np.cos(math.tau*(frequency_guess - frequency_step)*time_coarse))
    #     correlation_right = 2*np.average(spin*np.cos(math.tau*(frequency_guess + frequency_step)*time_coarse))
    #     if step_index % 10 == 0:
    #         print("{:d}\t{:1.7f}\t{:1.7f}\t{:1.7f}\t{:1.7f}".format(step_index, frequency_guess, correlation_left, correlation, correlation_right))
    #     frequency_guess += step_size*(correlation_right - correlation_left)/frequency_step
    #     if correlation <= correlation_max:
    #         step_size /= 1.5
    #     else:
    #         correlation_max = correlation
    #     step_index += 1
    # print("{:d}\t{:1.7f}\t{:1.7f}\t{:1.7f}\t{:1.7f}".format(step_index, frequency_guess, correlation_left, correlation, correlation_right))

    step_size = 1e0
    step_index = 0
    # print("{:8s}\t{:8s}\t{:8s}\t{:8s}\t{:8s}\t{:8s}".format("Step Index", "Loss", "Frequency", "Phase", "DC", "Amplitude"))
    # print("{:8s}\t{:8s}\t{:8s}\t{:8s}\t{:8s}".format("Step Index", "Loss", "Frequency", "Dressing", "Shift"))
    difference_min = 100
    while step_size > 1e-9 and step_index < 200:
        frequency_derivative = (np.average(np.abs(spin - (amplitude_guess*np.cos(math.tau*(frequency_guess + frequency_step*step_size)*time_coarse + phase_guess) + dc_guess))) - np.average(np.abs(spin - (amplitude_guess*np.cos(math.tau*(frequency_guess - frequency_step*step_size)*time_coarse + phase_guess) + dc_guess))))
        # dc_derivative = (np.average(np.abs(spin - (amplitude_guess*np.cos(math.tau*frequency_guess*time_coarse + phase_guess) + (dc_guess + dc_step*step_size)))) - np.average(np.abs(spin - (amplitude_guess*np.cos(math.tau*frequency_guess*time_coarse + phase_guess) + (dc_guess - dc_step*step_size)))))
        # amplitude_derivative = (np.abs(np.average(spin - ((amplitude_guess + amplitude_step*step_size)*np.cos(math.tau*frequency_guess*time_coarse + phase_guess) + dc_guess))) - np.average(np.abs(spin - ((amplitude_guess - amplitude_step*step_size)*np.cos(math.tau*frequency_guess*time_coarse + phase_guess) + dc_guess))))
        # phase_derivative = np.average(np.abs(spin - (amplitude_guess*np.cos(math.tau*frequency_guess*time_coarse + (phase_guess + phase_step*step_size)) + dc_guess))) - np.average(np.abs(spin - (amplitude_guess*np.cos(math.tau*frequency_guess*time_coarse + (phase_guess - phase_step*step_size)) + dc_guess)))

        frequency_guess_previous = frequency_guess
        dc_guess_previous = dc_guess
        amplitude_guess_previous = amplitude_guess
        phase_guess_previous = phase_guess

        frequency_guess -= frequency_derivative
        # dc_guess -= dc_derivative*1e-1
        # amplitude_guess -= amplitude_derivative*1e-1
        # phase_guess -= phase_derivative*1e1

        # if dc_guess < 0.0:
        #     dc_guess = 0.0
        # if amplitude_guess > 1.0:
        #     amplitude_guess = 1.0

        difference = np.average(np.abs(spin - (amplitude_guess*np.cos(math.tau*frequency_guess*time_coarse + phase_guess) + dc_guess)))

        if difference > difference_min:# difference > (1 - 0.01*step_size)*difference_min:
            step_size /= 1.5
            # frequency_guess = frequency_guess_previous
            # dc_guess = dc_guess_previous
            # amplitude_guess = amplitude_guess_previous
            # phase_guess = phase_guess_previous
        else:
            difference_min = difference

        # print("{:8d}\t{:1.7f}\t{:1.7f}\t{:1.7f}\t{:1.7f}\t{:1.7f}".format(step_index, difference, frequency_guess, phase_guess, dc_guess, amplitude_guess), end = "\r")
        print("{:8d}\t{:1.7f}\t{:1.7f}".format(step_index, difference, frequency_guess), end = "\r")

        step_index += 1

    if do_plot:
        plt.figure()
        plt.plot(time_coarse, spin)
        plt.plot(time_coarse, amplitude_guess*np.cos(math.tau*frequency_guess*time_coarse + phase_guess) + dc_guess)
        plt.show()
    
    frequency_shift = frequency_guess - frequency_dressing
    # print("{:8d}\t{:1.7f}\t{:1.7f}\t{:1.7f}\t{:1.7f}\t{:1.7f}".format(step_index, difference, frequency_guess, phase_guess, dc_guess, amplitude_guess))
    print("{:8d}\t{:1.7f}\t{:1.7f}".format(step_index, difference, frequency_guess))
    # print("Frequency: {:1.7f}Hz\nGuess: {:1.7f}Hz\nShift: {:1.7f}Hz".format(frequency_dressing, frequency_guess, frequency_shift))
    
    shift = np.abs(frequency_shift)
    shift_predict = math.sqrt(frequency_dressing**2 + (0.25*(frequency_dressing**2)/460e3)**2) - frequency_dressing

    return shift, shift_predict

class Spectrogram():
    def __init__(self, time, signal, bin_size, frequency_range):
        self.time = np.ascontiguousarray(time)
        self.signal = np.ascontiguousarray(signal)
        self.frequency_range = frequency_range
        time_step = (time[time.size - 1] - time[0])/time.size
        # print((int((time[time.size - 1] - time[0])//bin_size[0]), int((frequency_range[1] - frequency_range[0])//bin_size[1])))
        self.stft = np.empty((int(time.size//bin_size[0]), int((frequency_range[1] - frequency_range[0])//bin_size[1]), 3), np.complex128)
        self.bin_size = np.asarray(bin_size, np.uint8)
        self.bin_size[0] = int(self.bin_size[0]/time_step)
        self.bin_start = [
            time[0],
            frequency_range[0]
        ]
        self.bin_start = np.asarray(self.bin_start, np.uint64)

        threads_per_block = (8, 8)
        blocks_per_grid = (
            (self.time.size + (threads_per_block[0] - 1)) // threads_per_block[0],
            (int((self.frequency_range[1] - self.frequency_range[0])//bin_size[1]) + (threads_per_block[1] - 1)) // threads_per_block[1],
        )

        get_stft[blocks_per_grid, threads_per_block](self.time, self.signal, self.bin_start, self.bin_size, self.stft)
        # print(self.stft.real)

    def plot(self):
        plt.figure()
        stft = np.swapaxes(self.stft.real, 0, 1)
        stft = stft/np.max(stft)
        print(stft.shape)
        plt.imshow(
            stft,
            aspect = "auto",
            extent = (self.time[0], self.time[self.time.size - 1], self.frequency_range[0], self.frequency_range[1]),
            origin = "lower"
        )
        plt.show()

    def write_to_file(self, archive):
        pass

@cuda.jit
def get_stft(time, signal, bin_start, bin_size, stft):
    time_index = cuda.threadIdx.x + cuda.blockIdx.x*cuda.blockDim.x
    frequency_index = cuda.threadIdx.y + cuda.blockIdx.y*cuda.blockDim.y
    if time_index < stft.shape[0] and frequency_index < stft.shape[1]:
        frequency = bin_start[1] + frequency_index*bin_size[1]
        time_sample_index = time_index*bin_size[0]
        stft[time_index, frequency_index, 0] = 0
        for time_fine_index in range(bin_size[0]):
            stft[time_index, frequency_index, 0] += signal[time_sample_index + time_fine_index]*(math.cos(math.tau*frequency*time[time_sample_index + time_fine_index]) - 1j*math.sin(math.tau*frequency*time[time_sample_index + time_fine_index]))*math.exp(-1e14*(time[time_sample_index + int(bin_size[0]/2)] - time[time_sample_index + time_fine_index])**2)
        stft[time_index, frequency_index, 0] = (stft[time_index, frequency_index, 0].real**2 + stft[time_index, frequency_index, 0].imag**2)/(frequency**2)

        stft[time_index, frequency_index, 0] = math.log(stft[time_index, frequency_index, 0].real + 1)
        stft[time_index, frequency_index, 1] = 0#stft[time_index, frequency_index, 0]
        stft[time_index, frequency_index, 2] = 0#stft[time_index, frequency_index, 0]