simulation module
=================

.. .. _overviewOfSimulationMethod:

.. *********************************
.. Overview of the simulation method
.. *********************************

.. The goal here is to evaluate the value of the spin, and thus the quantum state of a 3 level atom in a coarse grained time series with step :math:`\mathrm{D}t`. The time evolution between time indices is

.. .. math::
..    \begin{align*}
..    \psi(t + \mathrm{D}t) &= U(t \rightarrow t + \mathrm{D}t) \psi(t)\\
..    \psi(t + \mathrm{D}t) &= U(t) \psi(t)
..    \end{align*}

.. Each :math:`U(t)` is completely independent of :math:`\psi(t_0)` or :math:`U(t_0)` for any other time value :math:`t_0`. Therefore each :math:`U(t)` can be calculated independently of each other. This is done in parallel using a GPU kernel in the function :func:`getTimeEvolutionCommutatorFree4RotatingWave()` (the highest performing variant of this solver). Afterwards, the final result of

.. .. math::
..    \psi(t + \mathrm{D}t) = U(t) \psi(t)

.. is calculated sequentially for each :math:`t` in the function :func:`getState()`. Afterwards, the spin at each time step is calculated in parallel in the function :func:`getSpin()`.

.. All magnetic signals fed into the integrator in the form of sine waves, with varying amplitude, frequency, phase, and start and end times. This can be used to simulate anything from the bias and dressing fields, to the fake neural pulses, to AC line and DC detuning noise. These sinusoids are superposed and sampled at any time step needed to for the solver. The magnetic signals are written in high level as :class:`testSignal.TestSignal` objects, and are converted to a parametrisation readable to the integrator in the form of :class:`SourceProperties` objects.

.. **********
.. Components
.. **********

.. automodule:: simulation
   :members:
   :undoc-members:
   :show-inheritance: