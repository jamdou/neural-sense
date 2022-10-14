import numba as nb
import numpy as np
import math
import archive
import matplotlib.pyplot as plt

def fit_3rd_harmonic_wrap(time:np.ndarray, spin:np.ndarray, sample_time, parameter_guess:np.ndarray, parameter_gradient_step = 1e0, max_iteration = 10000):
    parameter = parameter_guess.copy()
    parameter_gradient = np.empty_like(parameter)
    spin_model = np.empty_like(spin)

    for iteration_index in range(max_iteration):
        spin_model = sample_time*(parameter[0]*np.cos(math.tau*50*time) + parameter[1]*np.sin(math.tau*50*time) + parameter[2]*np.cos(math.tau*150*time) + parameter[3]*np.sin(math.tau*150*time) + parameter[4])

        parameter_gradient[0] = 2*np.sum((np.sin(spin_model) - spin)*np.cos(spin_model)*np.cos(math.tau*50*time))
        parameter_gradient[1] = 2*np.sum((np.sin(spin_model) - spin)*np.cos(spin_model)*np.sin(math.tau*50*time))
        parameter_gradient[2] = 2*np.sum((np.sin(spin_model) - spin)*np.cos(spin_model)*np.cos(math.tau*150*time))
        parameter_gradient[3] = 2*np.sum((np.sin(spin_model) - spin)*np.cos(spin_model)*np.sin(math.tau*150*time))
        parameter_gradient[4] = 2*np.sum((np.sin(spin_model) - spin)*np.cos(spin_model))

        parameter -= parameter_gradient*parameter_gradient_step

        if iteration_index % 1000 == 0:
            print(parameter)
            plt.figure()
            plt.plot(time, spin, "kx-")
            plt.plot(time, np.sin(spin_model), "rx-")
            plt.xlabel("Time (s)")
            plt.ylabel("Spin projection (hbar)")
            plt.legend(["Input", "Fit"])
            plt.title(f"Gradient descent fitting,\niteration {iteration_index}")
            plt.show()

    return parameter

time = np.arange(0, 40e-3, 2e-4, np.double)
# sample_time = 37.5e-6
sample_time = 1000e-6
spin = sample_time*(2400*np.cos(math.tau*50*time) + 3200*np.sin(math.tau*50*time) + 1600*np.cos(math.tau*150*time) + 1200*np.sin(math.tau*150*time) - 2400)
spin = np.sin(spin + 0.4*(1 - 2*np.random.random(spin.shape)))
parameter_guess = 8*np.array([500, 500, 250, 250, -400], np.double)
parameter = fit_3rd_harmonic_wrap(time, spin, sample_time, parameter_guess)
print(parameter)
# plt.show()