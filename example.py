import colorama         # Colourful terminal
colorama.init()

import spinsim.spinsim as spinsim

import numpy as np
import matplotlib.pyplot as plt

"""
Note: please see documentation for a full explanation of the source code.
"""

"""
======================================
Example 1: Spin half Larmor precession
======================================
"""
# Define a numba.cuda compatible source sampling function
def get_source_larmor(time_sample, simulation_index, source_sample):
    source_sample[2] = 1000         # Split spin z eigenstates by 1kHz

    source_sample[0] = 0            # Zero source in x direction
    source_sample[1] = 0            # Zero source in y direction

# Return a solver which uses this function
get_time_evolution_larmor = spinsim.time_evolver_factory(get_source_larmor, spinsim.SpinQuantumNumber.HALF)

# The resultion of the output of the simulation is 100ns
time_step_coarse = 100e-9
# The resultion of the integration in the simulation is 10ns
time_step_fine = 10e-9
# Run between times of 0ms and 100ms.
time_end_points = np.asarray([0e-3, 100e-3], np.double)

# The number of samples in the output
time_index_max = int((time_end_points[1] - time_end_points[0])/time_step_coarse)
# Define an empty array to write the time to
time = np.empty(time_index_max, np.double)

# Define the initial state of the system (eigenstate of spin x)
state_init = np.asarray([1/np.sqrt(2), 1/np.sqrt(2)], np.cdouble)
# Define an empty array to write the state to
state = np.empty((time_index_max, 2), np.cdouble)
# Define an empty array to write the spin to
spin = np.empty((time_index_max, 3), np.double)

# Define an empty array to write the time evolution operator to
time_evolution = np.empty((time_index_max, 2, 2), np.cdouble)

# Set up the gpu to have threads of size 64
threads_per_block = 64
blocks_per_grid = (time_index_max + (threads_per_block - 1)) // threads_per_block

# Find the time evolution operator using our settings
get_time_evolution_larmor[blocks_per_grid, threads_per_block](0, time, time_end_points, time_step_fine, time_step_coarse, time_evolution)

# Chain the time evolution operators together to find the state at each point in time
spinsim.get_state(state_init, state, time_evolution)

# Calculate the spin at each point in time
spinsim.get_spin[blocks_per_grid, threads_per_block](state, spin)

# Plot result
plt.figure()
plt.plot(time, spin)
plt.legend(["x", "y", "z"])
plt.xlim(0e-3, 2e-3)
plt.xlabel("time (s)")
plt.ylabel("spin expectation (hbar)")
plt.title("Spin projection for Larmor precession")
plt.show()

"""
=================================
Example 2: Spin one Rabi flopping
=================================
"""

# The math, or cmath library must be used over numpy for the source function to be cuda compilable
import math

def get_source_rabi(time_sample, simulation_index, source_sample):
    # Dress atoms from the x direction, Rabi flopping at 1kHz, then doubles each simulation_index
    source_sample[0] = 2000*math.cos(math.tau*20e3*simulation_index*time_sample)
    source_sample[1] = 0                        # Zero source in y direction
    source_sample[2] = 20e3*simulation_index    # Split spin z eigenstates by 700kHz

    source_sample[3] = 0                        # Zero quadratic shift, found in spin one systems

# Return a solver which uses this function
get_time_evolution_rabi = spinsim.time_evolver_factory(get_source_rabi, spinsim.SpinQuantumNumber.ONE)

# Some optimisations can be done with memory, using numba.cuda functions
from numba import cuda

# The resultion of the output of the simulation is 100ns
time_step_coarse = 100e-9
# The resultion of the integration in the simulation is 10ns
time_step_fine = 10e-9
# Run between times of 0ms and 100ms.
time_end_points = np.asarray([0e-3, 100e-3], np.double)

# The number of samples in the output
time_index_max = int((time_end_points[1] - time_end_points[0])/time_step_coarse)

# Define the initial state of the system (eigenstate of spin z)
state_init = np.asarray([1, 0, 0], np.cdouble)
# Define an empty array to write the state to
state = np.empty((time_index_max, 3), np.cdouble)

# Define an empty array directly on the gpu to write the time evolution operator to
time_evolution = cuda.device_array((time_index_max, 3, 3), np.cdouble)
# Define an empty array directly on the gpu to write the time to
time = cuda.device_array(time_index_max, np.double)
# Define an empty array directly on the gpu to write the spin to
spin = cuda.device_array((time_index_max, 3), np.double)

# Set up the gpu to have threads of size 64
threads_per_block = 64
blocks_per_grid = (time_index_max + (threads_per_block - 1)) // threads_per_block

# Find the time evolution operator using our settings
get_time_evolution_rabi[blocks_per_grid, threads_per_block](1, time, cuda.to_device(time_end_points), time_step_fine, time_step_coarse, time_evolution)

# Get arrays off the gpu
time = time.copy_to_host()
time_evolution = time_evolution.copy_to_host()

# Chain the time evolution operators together to find the state at each point in time
spinsim.get_state(state_init, state, time_evolution)

# Calculate the spin at each point in time
spinsim.get_spin[blocks_per_grid, threads_per_block](cuda.to_device(state), spin)

# Get spin off the gpu
spin = spin.copy_to_host()

# Plot result
plt.figure()
plt.plot(time, spin)
plt.legend(["x", "y", "z"])
plt.xlim(0e-3, 2e-3)
plt.xlabel("time (s)")
plt.ylabel("spin expectation (hbar)")
plt.title("Spin projection for Rabi flopping")
plt.show()

# Define an empty array directly on the gpu to write the time evolution operator to
time_evolution = cuda.device_array((time_index_max, 3, 3), np.cdouble)
# Define an empty array directly on the gpu to write the time to
time = cuda.device_array(time_index_max, np.double)
# Define an empty array directly on the gpu to write the spin to
spin = cuda.device_array((time_index_max, 3), np.double)

# Find the time evolution operator using our settings
get_time_evolution_rabi[blocks_per_grid, threads_per_block](2, time, cuda.to_device(time_end_points), time_step_fine, time_step_coarse, time_evolution)

# Get arrays off the gpu
time = time.copy_to_host()
time_evolution = time_evolution.copy_to_host()

# Chain the time evolution operators together to find the state at each point in time
spinsim.get_state(state_init, state, time_evolution)

# Calculate the spin at each point in time
spinsim.get_spin[blocks_per_grid, threads_per_block](cuda.to_device(state), spin)

# Get spin off the gpu
spin = spin.copy_to_host()

# Plot result
plt.figure()
plt.plot(time, spin)
plt.legend(["x", "y", "z"])
plt.xlim(0e-3, 2e-3)
plt.xlabel("time (s)")
plt.ylabel("spin expectation (hbar)")
plt.title("Spin projection for Rabi flopping")
plt.show()