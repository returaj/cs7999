#!/usr/bin/env python3

from typing import Any

import argparse
import functools
from brax import envs
import os
import json
import jax
import jax.numpy as jnp
from jax.scipy.stats import multivariate_normal
import flax
from flax import linen as nn
from flax.training import train_state
import optax


DTYPE = Any


def get_argparser():
    parser = argparse.ArgumentParser(description="Parse arguments for BCND")
    parser.add_argument("-seed", type=int, default=0, help="seed")
    parser.add_argument("--batch", type=int, default=128, help="batch size")
    parser.add_argument(
        "--epochs", type=int, default=500, help="number of training epochs"
    )
    parser.add_argument("--env", type=str, default="ant", help="environment name")
    parser.add_argument("--noise_name", type=str, default="normal", help="noise name")
    parser.add_argument("--noise_level", type=str, default="0.1", help="noise level")
    parser.add_argument("--k", type=int, default=5, help="number of ensembles")
    parser.add_argument(
        "--num_evals", type=int, default=5, help="number of evaluations"
    )
    parser.add_argument("--algo", type=str, default="bc", help="algorithm type")
    return parser


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
    def __init__(self, k, xsize, usize):
        self.k = k
        self.xsize = xsize
        self.usize = usize
        self.policies = nn.scan(
            MLP,
            variable_axes={"params": 0},
            variable_broadcast=False,
            split_rngs={"params": True},
            length=k,
        )(out=usize)

    def initialize_params(self, key):
        dummy_x = jnp.zeros(self.xsize)
        return self.policies.init(key, dummy_x, None)

    @functools.partial(jax.jit, static_argnums=0)
    def log_value(self, x, u, params):
        def log_norm(u, mu, log_std):
            cov = jnp.diag(jnp.exp(log_std * 2))
            return multivariate_normal.logpdf(u, mean=mu, cov=cov)

        means, log_stds = self.predict_means_and_logstds(x, params)
        values = jax.vmap(log_norm, in_axes=(None, 0, 0))(u, means, log_stds)
        return jnp.log(jnp.mean(jnp.exp(values)))

    def predict_means_and_logstds(self, x, params):
        _, (means, log_stds) = self.policies.apply(params, x, None)
        return means, log_stds

    @functools.partial(jax.jit, static_argnums=0)
    def sample(self, x, params, key):
        _, (means, log_stds) = self.policies.apply(params, x, None)
        normal_samples = jax.random.normal(key, shape=log_stds.shape)
        values = means + normal_samples * jnp.exp(log_stds)
        return jnp.mean(values, axis=0)


class TrainState(train_state.TrainState):
    old_params: flax.core.FrozenDict[str, Any] = None


def get_trajectory_dataset(path):
    with open(path, "r") as f:
        data = json.load(f)
    X, Y = [], []
    for traj in data:
        X.extend(traj["states"])
        Y.extend(traj["actions"])
    return jnp.array(X), jnp.array(Y)


def create_trainstate(policy_model, key, learning_rate):
    def _get_batch_processing_fn(fn, in_axes):
        return jax.vmap(fn, in_axes=in_axes)

    params = policy_model.initialize_params(key)
    tx = optax.adam(learning_rate=learning_rate)
    return TrainState.create(
        apply_fn=_get_batch_processing_fn(policy_model.log_value, in_axes=(0, 0, None)),
        params=params,
        old_params=params,
        tx=tx,
    )


@jax.jit
def train_epoch_bcnd(trainstate, perm, dataset):
    X, Y = dataset

    @jax.jit
    def train_batch(trainstate, p):
        batch_x, batch_y = X[p], Y[p]
        batch_rwd = jnp.exp(
            trainstate.apply_fn(batch_x, batch_y, trainstate.old_params)
        )
        batch_rwd /= jnp.sum(batch_rwd)

        def loss_fn(params):
            log_values = trainstate.apply_fn(batch_x, batch_y, params)
            return -jnp.mean(batch_rwd * log_values)

        loss, grads = jax.value_and_grad(loss_fn)(trainstate.params)
        trainstate = trainstate.apply_gradients(grads=grads)
        return trainstate, loss

    trainstate, losses = jax.lax.scan(train_batch, trainstate, perm)
    trainstate = trainstate.replace(old_params=trainstate.params)
    return trainstate, jnp.mean(losses)


@jax.jit
def train_epoch_bc(trainstate, perm, dataset):
    X, Y = dataset

    @jax.jit
    def train_batch(trainstate, p):
        batch_x, batch_y = X[p], Y[p]

        def loss_fn(params):
            log_values = trainstate.apply_fn(batch_x, batch_y, params)
            return -jnp.mean(log_values)

        loss, grads = jax.value_and_grad(loss_fn)(trainstate.params)
        trainstate = trainstate.apply_gradients(grads=grads)
        return trainstate, loss

    trainstate, losses = jax.lax.scan(train_batch, trainstate, perm)
    return trainstate, jnp.mean(losses)


def train(trainstate, dataset, key, batch_size, num_epochs, train_epoch_fn):
    datasize = dataset[0].shape[0]
    steps_per_epoch = datasize // batch_size

    losses = []
    for ep in range(num_epochs):
        key, subkey = jax.random.split(key)
        perm = jax.random.choice(
            subkey, datasize, shape=(steps_per_epoch, batch_size), replace=False
        )
        trainstate, loss = train_epoch_fn(trainstate, perm, dataset)
        losses.append(loss)
        if ((ep + 1) % 10) == 0:
            print(f"epoch: {ep + 1}, loss: {loss}")
    return trainstate, losses


def evaluate(env, key, policy_model, params, num_evals):
    jit_env_reset = jax.jit(env.reset)
    jit_env_step = jax.jit(env.step)

    avg_rwd = 0
    for i in range(num_evals):
        key, state_key, sub_key = jax.random.split(key, 3)
        rwd = 0
        state = jit_env_reset(rng=state_key)
        for _ in range(1000):
            sub_key, act_key = jax.random.split(sub_key)
            act = policy_model.sample(state.obs, params, act_key)
            state = jit_env_step(state, act)
            rwd += state.reward
        avg_rwd += (rwd - avg_rwd) / (i + 1)
    return avg_rwd


if __name__ == "__main__":
    parser = get_argparser()
    args = parser.parse_args()

    print(args)

    current_file_path = os.path.dirname(__file__)
    dataset_path = f"{current_file_path}/noisy_data/{args.env}/expert-{args.noise_name}/{args.noise_level}/trajectories.json"
    X, Y = get_trajectory_dataset(dataset_path)
    key = jax.random.PRNGKey(seed=args.seed)
    policy_model = MeanPolicy(k=args.k, xsize=X.shape[-1], usize=Y.shape[-1])
    learning_rate = args.k * 1e-4
    trainstate = create_trainstate(policy_model, key, learning_rate)
    train_epoch_fn = train_epoch_bc if args.algo == "bc" else train_epoch_bcnd
    trainstate, losses = train(
        trainstate=trainstate,
        dataset=(X, Y),
        key=key,
        batch_size=args.batch,
        num_epochs=args.epochs,
        train_epoch_fn=train_epoch_fn,
    )
    env = envs.create(env_name=args.env, backend="positional")
    avg_rwd = evaluate(env, key, policy_model, trainstate.params, args.num_evals)
    print(
        f"Avg reward for {args.env} under {args.noise_level} epsilon {args.noise_name} policy: {avg_rwd}"
    )
