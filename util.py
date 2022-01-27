from matplotlib.colors import Colormap
import numpy as np
import numba as nb
import numba.cuda as cuda
import math
import matplotlib.pyplot as plt
import h5py

import test_signal
import archive as arch
# from sim import manager

class PrettyTritty:
  d = "\033[0m"
  r = "\033[31m"
  g = "\033[32m"
  y = "\033[33m"
  k = "\033[30m"

  level = 0

  @classmethod
  def starting(cls, message:str):
    tabs = cls.level*"  "
    message = message.replace('\n', f'\n{tabs}')
    print(f"{tabs}{cls.y}Starting {message}...{cls.d}")
    cls.level += 1

  @classmethod
  def finished(cls, message:str):
    cls.level = max(cls.level - 1, 0)
    tabs = cls.level*"  "
    message = message.replace('\n', f'\n{tabs}')
    print(f"{tabs}{cls.g}Finished {message}!{cls.d}")

  @classmethod
  def print(cls, message:str, end = "\n"):
    tabs = cls.level*"  "
    message = message.replace('\n', f'\n{tabs}')
    print(f"{tabs}{message}", end = end)

C = PrettyTritty
class Seeds:
  metroid = 19960806

class ScaledParameters():
  def __init__(self, scaled_frequency = 5000, scaled_density = 1/25, scaled_samples = 10, scaled_amplitude = 800, scaled_sweep = [2000, 7000], scaled_pulse_time_fraction = 0.2333333, scaled_time_step = None, scaled_time_end = None, scaled_pulse_time = None, scaled_frequency_step = None, scaled_stagger_constant = None, scaled_sample_frequencies = None):
    self.frequency = scaled_frequency
    self.density = scaled_density
    self.samples = scaled_samples
    self.amplitude = scaled_amplitude
    self.sweep = np.asarray(scaled_sweep)

    if scaled_time_step is not None:
      self.time_step = scaled_time_step
    else:
      self.time_step = 1/(self.frequency*self.samples)
    if scaled_time_end is not None:
      self.time_end = scaled_time_end
    else:
      self.time_end = 1/(self.frequency*self.density)
    if scaled_pulse_time is not None:
      self.pulse_time = np.array(scaled_pulse_time)
    else:
      self.pulse_time = np.array(scaled_pulse_time_fraction)*self.time_end
    if scaled_frequency_step is not None:
      self.frequency_step = scaled_frequency_step
    else:
      self.frequency_step = self.density*self.frequency/2

    if scaled_stagger_constant is not None:
      self.stagger_constant = scaled_stagger_constant
    else:
      self.stagger_constant = 0
    if scaled_sample_frequencies is not None:
      self.sample_frequencies = scaled_sample_frequencies
    else:
      self.sample_frequencies = np.arange(self.sweep[0], min(max(self.sweep[1], 0), self.samples*self.frequency/2), self.frequency_step)
      self.sample_frequencies += np.fmod((self.stagger_constant*self.sample_frequencies)**2 + self.frequency_step/2, self.frequency_step) - self.frequency_step/2

  def write_to_file(self, archive:arch.Archive):
    archive.archive_file["scaled_parameters"] = np.empty(0)
    archive.archive_file["scaled_parameters"].attrs["frequency"] = self.frequency
    archive.archive_file["scaled_parameters"].attrs["density"] = self.density
    archive.archive_file["scaled_parameters"].attrs["samples"] = self.samples
    archive.archive_file["scaled_parameters"].attrs["amplitude"] = self.amplitude
    archive.archive_file["scaled_parameters"].attrs["sweep"] = self.sweep

    archive.archive_file["scaled_parameters"].attrs["time_step"] = self.time_step
    archive.archive_file["scaled_parameters"].attrs["time_end"] = self.time_end
    archive.archive_file["scaled_parameters"].attrs["pulse_time"] = self.pulse_time
    archive.archive_file["scaled_parameters"].attrs["frequency_step"] = self.frequency_step
    archive.archive_file["scaled_parameters"].attrs["stagger_constant"] = self.stagger_constant

  def get_neural_pulses(self):
    neural_pulses = []
    for pulse_time_instance in self.pulse_time:
      neural_pulses.append(test_signal.NeuralPulse(pulse_time_instance, self.amplitude, self.frequency))
    return neural_pulses


  @staticmethod
  def new_from_archive_time(archive:arch.Archive, archive_time):
    archive_previous = arch.Archive(archive.archive_path[:-25], "")
    archive_previous.open_archive_file(archive_time)
    scaled = ScaledParameters(
      scaled_frequency = archive_previous.archive_file["scaled_parameters"].attrs["frequency"],
      scaled_density = archive_previous.archive_file["scaled_parameters"].attrs["density"],
      scaled_samples = archive_previous.archive_file["scaled_parameters"].attrs["samples"],
      scaled_amplitude = archive_previous.archive_file["scaled_parameters"].attrs["amplitude"],
      scaled_sweep = np.asarray(archive_previous.archive_file["scaled_parameters"].attrs["sweep"]),
      scaled_time_step = archive_previous.archive_file["scaled_parameters"].attrs["time_step"],
      scaled_time_end = archive_previous.archive_file["scaled_parameters"].attrs["time_end"],
      scaled_pulse_time = archive_previous.archive_file["scaled_parameters"].attrs["pulse_time"],
      scaled_frequency_step = archive_previous.archive_file["scaled_parameters"].attrs["frequency_step"],
      scaled_stagger_constant = archive_previous.archive_file["scaled_parameters"].attrs["stagger_constant"],
      scaled_sample_frequencies = np.asarray(archive_previous.archive_file["scaled_parameters"].attrs["sample_frequencies"])
    )
    return scaled

  @staticmethod
  def new_from_experiment_time(archive_time):
    if archive_time == "20210429T125734":
      scaled = ScaledParameters(
        scaled_frequency = 5013,
        scaled_density = 1/25,
        scaled_samples = 10,
        scaled_amplitude = 995.5,
        scaled_sweep = [5013/5, 14000],
        scaled_pulse_time_fraction = [0.2333333]
      )
    elif archive_time == "20210429T125734e":
      scaled = ScaledParameters(
        scaled_frequency = 5013,
        scaled_density = 1/24,
        scaled_samples = 10,
        scaled_amplitude = 995.5,
        scaled_sweep = [5013/5, 14000],
        scaled_pulse_time_fraction = [0.2333333*25/24]
      )
    elif archive_time == "20210429T125734i":
      scaled = ScaledParameters(
        scaled_frequency = 5013,
        scaled_density = 1/25,
        scaled_samples = 10/math.sqrt(2),
        scaled_amplitude = 995.5,
        scaled_sweep = [5013/5, 14000],
        scaled_pulse_time_fraction = [0.2333333]
      )
    elif archive_time == "20210429T125734s":
      scaled = ScaledParameters(
        scaled_frequency = 5013,
        scaled_density = 1/25,
        scaled_samples = 10,
        scaled_amplitude = 995.5,
        scaled_sweep = [5013/5, 14000],
        scaled_pulse_time_fraction = [0.2333333],
        scaled_stagger_constant = math.sqrt(5)
      )
    elif archive_time == "20210430T162501":
      scaled = ScaledParameters(
        scaled_frequency = 5013,
        scaled_density = 1/25,
        scaled_samples = 10,
        scaled_amplitude = 995.5/2,
        scaled_sweep = [5013/5, 14000],
        scaled_pulse_time_fraction = [0.2333333]
      )
    elif archive_time == "20210504T142057":
      scaled = ScaledParameters(
        scaled_frequency = 5000,
        scaled_density = 1/100,
        scaled_samples = 10,
        scaled_amplitude = 995.5/2,
        scaled_sweep = [2000, 7000],
        scaled_pulse_time_fraction = [0.2333333]
      )
    elif archive_time == "20210504T142057s":
      scaled = ScaledParameters(
        scaled_frequency = 5000,
        scaled_density = 1/100,
        scaled_samples = 10,
        scaled_amplitude = 995.5,
        scaled_sweep = [2000, 14000],
        scaled_pulse_time_fraction = [0.2333333],
        scaled_frequency_step = (1/100)*5000,
        scaled_stagger_constant = math.tau
      )
    elif archive_time in ["20211202T153620", "20211117T123323", "20211117T155508", "20211125T124842"]:
      scaled = ScaledParameters(
        scaled_frequency = 5000,
        scaled_density = 1/25,
        scaled_samples = 10,
        scaled_amplitude = 1000,
        scaled_sweep = [1000, 14000],
        scaled_pulse_time_fraction = [0.2333333]
      )
    # One signal, 150 shots
    elif archive_time in ["20220118T131910", "20211209T143732", "20211202T124902"]:
      scaled = ScaledParameters(
        scaled_frequency = 5000,
        scaled_density = 1/25,
        scaled_samples = 10,
        scaled_amplitude = 1000,
        scaled_sweep = [100, 25001],
        scaled_pulse_time_fraction = [0.2333333]
      )
    # No signal, 250 shots
    elif archive_time in ["20211216T113507"]:
      scaled = ScaledParameters(
        scaled_frequency = 5000,
        scaled_density = 1/25,
        scaled_samples = 10,
        scaled_amplitude = 0,
        scaled_sweep = [100, 25001],
        scaled_pulse_time_fraction = [0]
      )
    # Two signals, 250 shots
    elif archive_time in ["20220118T124831", "20211216T161624", "20220127T131147"]:
      scaled = ScaledParameters(
        scaled_frequency = 5000,
        scaled_density = 1/25,
        scaled_samples = 10,
        scaled_amplitude = 1000,
        scaled_sweep = [100, 25001],
        # scaled_pulse_time_fraction = [0.4333333, 0.64999995]
        scaled_pulse_time = [0.0021666, 0.0031749]
      )
    return scaled

  def print(self):
    print(f"{'freq_n':>10s} {'period_n':>10s} {'time_n':>10s} {'sig_dense':>10s} {'samp_num':>10s} {'freq_d_s':>10s} {'freq_d_e':>10s} {'dfreq_d':>10s} {'time_e':>10s}")
    print(f"{self.frequency:10.4e} {1/self.frequency:10.4e} {self.pulse_time[0]:10.4e} {self.density:10.4e} {self.samples:10.4e} {self.sweep[0]:10.4e} {self.sweep[1]:10.4e} {self.frequency_step:10.4e} {self.time_end:10.4e}")

def get_noise_evaluation(archive_time):
  if archive_time == "20211117T155508":
    return "20211125T170254"
  elif archive_time in ["20211216T161624"]:
    return "20211217T111849"
    # return "20211124T161452"
  elif archive_time in ["20211216T113507", "20211209T143732", "20211216T161624"]:
    return "20220113T104353"
  elif archive_time in ["20220118T124831", "20220118T131910", "20220127T131147"]:
    return "20220118T172329"

def fit_frequency_shift(archive, signal, frequency, state_properties, do_plot = True, do_plot_individuals = False):
  """

  """
  from sim import manager
  
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
    plt.draw()
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
    plt.draw()

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
  #   correlation = 2*np.average(spin*np.cos(math.tau*frequency_guess*time_coarse))
  #   correlation_left = 2*np.average(spin*np.cos(math.tau*(frequency_guess - frequency_step)*time_coarse))
  #   correlation_right = 2*np.average(spin*np.cos(math.tau*(frequency_guess + frequency_step)*time_coarse))
  #   if step_index % 10 == 0:
  #     print("{:d}\t{:1.7f}\t{:1.7f}\t{:1.7f}\t{:1.7f}".format(step_index, frequency_guess, correlation_left, correlation, correlation_right))
  #   frequency_guess += step_size*(correlation_right - correlation_left)/frequency_step
  #   if correlation <= correlation_max:
  #     step_size /= 1.5
  #   else:
  #     correlation_max = correlation
  #   step_index += 1
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
    #   dc_guess = 0.0
    # if amplitude_guess > 1.0:
    #   amplitude_guess = 1.0

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
    plt.draw()
  
  frequency_shift = frequency_guess - frequency_dressing
  # print("{:8d}\t{:1.7f}\t{:1.7f}\t{:1.7f}\t{:1.7f}\t{:1.7f}".format(step_index, difference, frequency_guess, phase_guess, dc_guess, amplitude_guess))
  print("{:8d}\t{:1.7f}\t{:1.7f}".format(step_index, difference, frequency_guess))
  # print("Frequency: {:1.7f}Hz\nGuess: {:1.7f}Hz\nShift: {:1.7f}Hz".format(frequency_dressing, frequency_guess, frequency_shift))
  
  shift = np.abs(frequency_shift)
  shift_predict = math.sqrt(frequency_dressing**2 + (0.25*(frequency_dressing**2)/460e3)**2) - frequency_dressing

  return shift, shift_predict

def fft_cuda(signal):
  power_of_2_shape = int(2**math.ceil(math.log2(signal.shape[1])))
  number_of_iterations = int(math.ceil(math.log2(signal.shape[1])))
  if signal.ndim == 2:
    signal_reshape = np.zeros((signal.shape[0], power_of_2_shape))
  else:
    signal_reshape = np.zeros((1, power_of_2_shape))
  signal_reshape[:, 0:signal.shape[1]] = signal
  output = np.array(signal_reshape, np.cdouble)
  # output = cuda.to_device(output)

  block_size = (int(max(128//signal_reshape.shape[0], 1)), int(signal_reshape.shape[0]))
  number_of_blocks = (int(signal_reshape.shape[1] // block_size[0]), int(signal_reshape.shape[0] // block_size[1]))
  print(block_size)
  print(number_of_blocks)
  for iteration_index in range(1, number_of_iterations):
    stride = int(2**iteration_index)
    # block_size = (int(max(128//signal_reshape.shape[0], 1)), int(signal_reshape.shape[0]))
    # number_of_blocks = (int(max((signal_reshape.shape[1]/stride) // block_size[0], 1)), int(signal_reshape.shape[0] // block_size[1]))
    # print(block_size)
    dit_fft_cuda[number_of_blocks, block_size](output, stride)
  
  return output

@cuda.jit("(complex128[:, :], int32)")
def dit_fft_cuda(input, stride):
  sample_index = cuda.blockIdx.x*cuda.blockDim.x + cuda.threadIdx.x
  fft_index = cuda.blockIdx.y*cuda.blockDim.y + cuda.threadIdx.x
  if fft_index < input.shape[0]:
    if sample_index < input.shape[1]//stride:
      for stride_index in range(input.shape[1]//stride//2):
        p = input[fft_index, sample_index*stride + stride_index]
        q = input[fft_index, sample_index*stride + stride_index + stride//2]*(math.cos(math.tau/(input.shape[1]//stride)*stride_index) - 1j*math.sin(math.tau/(input.shape[1]//stride)*stride_index))
        input[fft_index, sample_index*stride + stride_index] = p + q
        input[fft_index, sample_index*stride + stride_index + stride//2] = p - q

def fft_cuda_test():
  time = np.linspace(0, 1, 4096)
  # signal = np.vstack([np.cos(math.tau*300*time), np.cos(math.tau*400*time)])
  signal = np.vstack([np.sign(np.cos(math.tau*300*time))])
  plt.figure()
  plt.plot(signal[0, :])
  plt.show()
  print(signal.shape)
  fft = np.abs(fft_cuda(signal))
  plt.figure()
  plt.plot(fft[0, :])
  # plt.plot(fft[1, :])
  plt.show()

if __name__ == "__main__":
  fft_cuda_test()
# class Sepctrogram():
#   def __init__(self, signal):
#     self.signal = signal

#   @nb.njit
#   def compute_stft(self):
#     pass
  
#   @nb.njit
#   def compute_fft(self, signal):
#     signal_reshape = np.zeros(2**math.ceil(math.log2(signal.size)))
#     self.signal[0, signal.size - 1] = signal
#     output = self.compute_dit_fft(signal, 0, signal_reshape.size, 1)
#     return output
  
#   @nb.njit
#   def compute_dit_fft(self, signal, start, number_of_samples, stride):
#     output = np.empty(number_of_samples)
#     if number_of_samples == 1:
#       output[0] = np.array([signal[start]])
#     else:
#       for i in nb.prange(2):
#         if i == 0:
#           output_even = self.compute_dit_fft(signal, start, number_of_samples/2, stride*2)
#         else:
#           output_odd = self.compute_dit_fft(signal, start + stride, number_of_samples/2, stride*2)
#       np.concatenate(output_even, output_odd)
#       for sample_point in nb.prange(number_of_samples/2):
#         p = output[sample_point]
#         q = output[sample_point + number_of_samples/2]*math.exp(-1j*math.tau/number_of_samples*sample_point)
#         output[sample_point] = p + q
#         output[sample_point + number_of_samples/2] = p - q
#     return output


# class Spectrogram():
#   def __init__(self, time, signal, bin_size, window_size, frequency_range):
#     self.time = np.ascontiguousarray(time)
#     self.signal = np.ascontiguousarray(signal)
#     self.frequency_range = frequency_range
#     time_step = (time[time.size - 1] - time[0])/time.size
#     self.bin_size = [bin_size[0], bin_size[1]]
#     self.bin_size[0] /= time_step
#     self.window_size = int(0.5*window_size/time_step)
#     print(self.window_size)
#     # print((int((time[time.size - 1] - time[0])//bin_size[0]), int((frequency_range[1] - frequency_range[0])//bin_size[1])))
#     self.stft = np.empty((int(time.size//self.bin_size[0]), int((frequency_range[1] - frequency_range[0])//self.bin_size[1]), 3), np.float64)
#     self.bin_size[0] = int(time.size/self.stft.shape[0])
#     self.bin_size = np.asarray(self.bin_size, np.int64)
#     # self.bin_size[0] = int(self.bin_size[0]/time_step)
#     self.bin_start = [
#       time[0],
#       frequency_range[0]
#     ]
#     self.bin_start = np.asarray(self.bin_start, np.int64)

#     threads_per_block = (16, 8)
#     blocks_per_grid = (
#       (self.stft.shape[0] + (threads_per_block[0] - 1)) // threads_per_block[0],
#       (self.stft.shape[1] + (threads_per_block[1] - 1)) // threads_per_block[1],
#     )

#     get_stft[blocks_per_grid, threads_per_block](self.time, self.signal, self.bin_start, self.bin_size, self.window_size, self.stft)

#     # self.stft[:, :, 2] += np.min(self.stft[:, :, 2])
#     self.stft /= np.max(self.stft[:, :, 2])
#     self.stft[:, :, 0] = self.stft[:, :, 2]*0.5*(1 + self.stft[:, :, 0])
#     self.stft[:, :, 1] = self.stft[:, :, 2]*0.5*(1 + self.stft[:, :, 1])

#     # self.stft[:, :, 2] = 1 - self.stft[:, :, 2]
#     # print(self.stft.real)

#   def plot(self, do_technicolour = False):
#     plt.figure()
#     stft = np.swapaxes(self.stft, 0, 1)
#     if not do_technicolour:
#       stft = stft[:, :, 2]
#     plt.imshow(
#       stft,
#       aspect = "auto",
#       extent = (self.time[0], self.time[self.time.size - 1], self.frequency_range[0]/1e3, self.frequency_range[1]/1e3),
#       origin = "lower",
#       cmap = "inferno"
#     )
#     plt.xlabel("Time (s)")
#     plt.ylabel("Frequency (kHz)")
#     plt.draw()

#   def write_to_file(self, archive):
#     pass

# @cuda.jit
# def get_stft(time, signal, bin_start, bin_size, window_size, stft):
#   time_index = cuda.threadIdx.x + cuda.blockIdx.x*cuda.blockDim.x
#   frequency_index = cuda.threadIdx.y + cuda.blockIdx.y*cuda.blockDim.y
#   if time_index < stft.shape[0] and frequency_index < stft.shape[1]:
#     frequency = bin_start[1] + frequency_index*bin_size[1]
#     time_sample_index = time_index*bin_size[0]
#     stft_temp = 0
#     for time_fine_index in range(-window_size, window_size):
#       if time_sample_index + time_fine_index >= 0 and time_sample_index + time_fine_index < time.size:
#         stft_temp += signal[time_sample_index + time_fine_index]*(math.cos(math.tau*frequency*time[time_sample_index + time_fine_index]) - 1j*math.sin(math.tau*frequency*time[time_sample_index + time_fine_index]))*(1 + math.cos(0.5*math.tau*time_fine_index/window_size))
#         # *math.exp(-1e14*(time[time_sample_index + int(bin_size[0]/2)] - time[time_sample_index + time_fine_index])**2)
#     stft_abs = math.sqrt((stft_temp.real**2 + stft_temp.imag**2).real)#/(frequency**2)

#     # stft_log = math.log(stft_abs + 0.001)
#     stft[time_index, frequency_index, 0] = stft_temp.real
#     stft[time_index, frequency_index, 1] = stft_temp.imag
#     stft[time_index, frequency_index, 2] = stft_abs