#! /usr/bin/env python

import jax.numpy as jnp
from brax.training.acme import types
from flax import struct


@struct.dataclass
class Statistics:
    mean: types.Nest
    std: types.Nest


@struct.dataclass
class RunningStatistics(Statistics):
    count: jnp.ndarray
    summed_variance: types.Nest
