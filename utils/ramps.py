# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Functions for ramping hyperparameters up or down

Each function takes the current training step or epoch, and the
ramp length in the same format, and returns a multiplier between
0 and 1.
"""


import numpy as np


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0: # If rampup_length is 0, the function immediately returns 1.0, indicating that the ramp-up phase is complete
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length) # it clips the current value (i.e., epoch) to ensure it is within the range [0, rampup_length]
        phase = 1.0 - current / rampup_length  # It then calculates the phase as 1.0 - current / rampup_length, which represents the remaining portion of the ramp-up period
        return float(np.exp(-5.0 * phase * phase)) # it returns the exponential of -5.0 * phase * phase, which produces a value between 0 and 1 that increases smoothly as current progresses from 0 to rampup_length.


def linear_rampup(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length


def cosine_rampdown(current, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    assert 0 <= current <= rampdown_length
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))
