"""
Some device functions for doing maths and interpolation with :mod:`numba.cuda`.
"""

import math
import numpy as np
import numba as nb
from numba import cuda

from . import scalar
from . import spinHalf
from . import spinOne

@cuda.jit(device = True, inline = True)
def interpolateSourceLinear(source, timeSample, timeStepSource, sourceSample):
    """
    Samples the source using linear interpolation.

    Parameters
    ----------
    source : :class:`numpy.ndarray` of :class:`numpy.double` (spatialIndex, timeSourceIndex)
        The input source of the environment of the spin system. In units of Hz.
    timeSample : `float`
        The time to be sampled.
    timeStepSource : `float`
        The time step between source samples.
    sourceSample : :class:`numpy.ndarray` of :class:`numpy.double` (spatialIndex)
        The interpolated value of source.
    """
    timeSample = timeSample/timeStepSource
    timeIndexSource = int(timeSample)
    timeSample -= timeIndexSource

    timeIndexSourceNext = timeIndexSource + 1
    if timeIndexSourceNext > source.shape[0] - 1:
        timeIndexSourceNext = source.shape[0] - 1

    for spacialIndex in range(source.shape[1]):
        sourceSample[spacialIndex] = (timeSample*(source[timeIndexSourceNext, spacialIndex] - source[timeIndexSource, spacialIndex]) + source[timeIndexSource, spacialIndex])

@cuda.jit(device = True, inline = True)
def interpolateSourceCubic(source, timeSample, timeStepSource, sourceSample):
    """
    Samples the source using cubic interpolation.

    Parameters
    ----------
    source : :class:`numpy.ndarray` of :class:`numpy.double` (spatialIndex, timeSourceIndex)
        The input source of the environment of the spin system. In units of Hz.
    timeSample : `float`
        The time to be sampled.
    timeStepSource : `float`
        The time step between source samples.
    sourceSample : :class:`numpy.ndarray` of :class:`numpy.double` (spatialIndex)
        The interpolated value of source.
    """
    timeSample = timeSample/timeStepSource
    timeIndexSource = int(timeSample)
    timeSample -= timeIndexSource

    timeIndexSourceNext = timeIndexSource + 1
    if timeIndexSourceNext > source.shape[0] - 1:
        timeIndexSourceNext = source.shape[0] - 1
    timeIndexSourceNextNext = timeIndexSourceNext + 1
    if timeIndexSourceNextNext > source.shape[0] - 1:
        timeIndexSourceNextNext = source.shape[0] - 1
    timeIndexSourcePrevious = timeIndexSource - 1
    if timeIndexSourcePrevious < 0:
        timeIndexSourcePrevious = 0
    for spacialIndex in range(source.shape[1]):
        gradient = (source[timeIndexSourceNext, spacialIndex] - source[timeIndexSourcePrevious, spacialIndex])/2
        gradientNext = (source[timeIndexSourceNextNext, spacialIndex] - source[timeIndexSource, spacialIndex])/2

        sourceSample[spacialIndex] = (\
            (2*(timeSample**3) - 3*(timeSample**2) + 1)*source[timeIndexSource, spacialIndex] + \
            ((timeSample**3) - 2*(timeSample**2) + timeSample)*gradient + \
            (-2*(timeSample**3) + 3*(timeSample**2))*source[timeIndexSourceNext, spacialIndex] + \
            (timeSample**3 - timeSample**2)*gradientNext)