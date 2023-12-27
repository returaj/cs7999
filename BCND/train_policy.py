#!/usr/bin/env python3

from typing import Any
import functools

import jax
import jax.numpy as jnp
from jax.scipy.stats import multivariate_normal

import flax
from flax import linen as nn
from flax.training import train_state


DTYPE = Any


class MLP(nn.Module):
    out: int
    num_hidden_units: int = 100
    dtype: DTYPE = jnp.float32

    def setup(self):
        # these values are based on the BCND paper: https://openreview.net/forum?id=zrT3HcsWSAt
        self.mean = nn.Sequential(
            [
                nn.Dense(self.num_hidden_units),
                nn.tanh,
                nn.Dense(self.num_hidden_units),
                nn.tanh,
                nn.Dense(self.out),
                nn.tanh,
            ]
        )
        self.log_std = self.param(
            "log_std", nn.initializers.zeros, (self.out,), self.dtype
        )

    def __call__(self, x, _):
        return x, (self.mean(x), self.log_std)


class MeanPolicy:
    def __init__(self, k, usize):
        self.k = k
        self.usize = usize
        self.policies = nn.scan(
            MLP,
            variable_axes={"params": 0},
            variable_broadcast=False,
            split_rngs={"params": True},
            length=k,
        )(out=usize)

    def initialize_params(self, key, dummy_x):
        return self.policies.init(key, dummy_x, None)

    @staticmethod
    @jax.jit
    def log_value(x, means, log_stds):
        def log_norm(x, mu, log_std):
            cov = jnp.diag(jnp.exp(log_std * 2))
            return multivariate_normal.logpdf(x, mean=mu, cov=cov)

        values = jax.vmap(log_norm, in_axes=(None, 0, 0))(x, means, log_stds)
        return jnp.log(jnp.mean(jnp.exp(values)))

    def predict_means_and_logstds(self, x, params):
        _, (means, log_stds) = self.policies.apply(params, x)
        return means, log_stds

    def sample(self, x, params, key):
        _, (means, log_stds) = self.policies.apply(params, x, None)
        normal_samples = jax.random.normal(key, shape=log_stds.shape)
        values = means + normal_samples * jnp.exp(log_stds)
        return jnp.mean(values, axis=0)


class TrainState(train_state.TrainState):
    old_params: flax.core.FrozenDict[str, Any] = None


def train_epoch_bcnd(trainstate, perm, dataset):
    X, Y = dataset

    def train_batch(trainstate, p):
        batch_x, batch_y = X[p], Y[p]


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    k = 4
    xsize, usize = 5, 2

    mean_policy = MeanPolicy(k=k, usize=usize)
    params = mean_policy.initialize_params(key, jnp.zeros(xsize))
    normal_samples = jax.random.normal(key, shape=(k, usize))

    print(jax.tree_map(lambda x: x.shape, params))
    print(mean_policy.predict(jnp.ones(xsize), normal_samples, params))
