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
from flax import linen as nn
import optax


DTYPE = Any


def get_argparser():
    parser = argparse.ArgumentParser(description="Parse arguments for BCND")
    parser.add_argument("--seed", type=int, default=0, help="seed")
    parser.add_argument("--batch", type=int, default=128, help="batch size")
    parser.add_argument(
        "--epochs", type=int, default=500, help="number of training epochs"
    )
    parser.add_argument("--env", type=str, default="ant", help="environment name")
    parser.add_argument("--noise_name", type=str, default="normal", help="noise name")
    parser.add_argument("--noise_level", type=str, default="0.1", help="noise level")
    parser.add_argument("--k", type=int, default=5, help="number of ensembles")
    parser.add_argument("--algo", type=str, default="bc", help="algorithm type")
    parser.add_argument(
        "--rwd_update",
        type=int,
        default=100,
        help="calculate reward after every rwd_update epoch",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="if we want to normalize our state space",
    )
    return parser


@jax.jit
def logmeanexp(x):
    # x is a 1d array
    xmax = jnp.max(x)
    # log(exp(x1) + exp(x2) + ..) = xmax + log(exp(x1-xmax) + exp(x2-xmax) + ..)
    logsumexp = xmax + jnp.log(jnp.sum(jnp.exp(x - xmax)))
    return logsumexp - jnp.log(x.shape[-1])


def get_trajectory_dataset(path):
    with open(path, "r") as f:
        data = json.load(f)
    X, Y = [], []
    for traj in data:
        X.extend(traj["states"])
        Y.extend(traj["actions"])
    return jnp.array(X), jnp.array(Y)


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

    def __call__(self, x):
        return (self.mean(x), self.log_std)


class PolicyModel:
    def __init__(self, xsize, usize):
        self.xsize = xsize
        self.usize = usize
        self.model = MLP(out=usize)

    def initialize_params(self, key):
        dummy_x = jnp.zeros(self.xsize)
        return self.model.init(key, dummy_x)

    @functools.partial(jax.jit, static_argnums=0)
    def mean_and_logstd(self, x, params):
        return self.model.apply(params, x)

    @functools.partial(jax.jit, static_argnums=0)
    def log_value(self, u, mean, log_std):
        sigma_2 = jnp.maximum(jnp.exp(log_std * 2), 1e-6)  # avoid exp floating error
        cov = jnp.diag(sigma_2)
        return multivariate_normal.logpdf(u, mean=mean, cov=cov)

    @functools.partial(jax.jit, static_argnums=0)
    def sample(self, x, params, key):
        mean, log_std = self.mean_and_logstd(x, params)
        normal_sample = jax.random.normal(key, shape=log_std.shape)
        return mean + normal_sample * jnp.exp(log_std)


class MeanPolicy:
    def __init__(self, k, polciy_model: PolicyModel):
        self.k = k
        self.policy_model = polciy_model

    def initialize_params(self, key):
        rngs = jax.random.split(key, self.k)
        params = []
        for rng in rngs:
            params.append(self.policy_model.initialize_params(rng))
        return params

    def means_and_logstds(self, x, params):
        means, logstds = [], []
        for subparams in params:
            mu, logstd = self.policy_model.mean_and_logstd(x, subparams)
            means.append(mu)
            logstds.append(logstd)
        return jnp.array(means), jnp.array(logstds)

    @functools.partial(jax.jit, static_argnums=0)
    def log_value(self, u, means, log_stds):
        log_values = jax.vmap(self.policy_model.log_value, in_axes=(None, 0, 0))(
            u, means, log_stds
        )
        return logmeanexp(log_values)

    def sample(self, x, params, key):
        means, log_stds = self.means_and_logstds(x, params)
        normal_samples = jax.random.normal(key, shape=log_stds.shape)
        values = means + normal_samples * jnp.exp(log_stds)
        return jnp.mean(values, axis=0)


def evaluate(env_tuples, normalize_fn, mean_policy, num_evals, params, key):
    jit_env_reset, jit_env_step = env_tuples

    rewards = []
    for _ in range(num_evals):
        key, state_key, sub_key = jax.random.split(key, 3)
        rwd = 0
        state = jit_env_reset(rng=state_key)
        for _ in range(1000):
            sub_key, act_key = jax.random.split(sub_key)
            act = mean_policy.sample(normalize_fn(state.obs), params, act_key)
            state = jit_env_step(state, act)
            rwd += state.reward
        rewards.append(rwd)
    rewards = jnp.array(rewards)
    return jnp.mean(rewards), jnp.std(rewards), jnp.min(rewards), jnp.max(rewards)


@functools.partial(jax.jit, static_argnums=0)
def create_log_rewards(mean_policy, dataset, params):
    X, Y = dataset
    policy_model = mean_policy.policy_model

    def fn(x, y, params):
        mean, logstd = policy_model.mean_and_logstd(x, params)
        return policy_model.log_value(y, mean, logstd)

    # fix 0th model for reward calculation K_old = 1
    # See: BCND paper: https://openreview.net/forum?id=zrT3HcsWSAt
    subparams = params[0]
    return jax.vmap(fn, in_axes=(0, 0, None))(X, Y, subparams)


def get_pred_fn(mean_policy):
    policy_model = mean_policy.policy_model

    def pred_fn(model_params, X, Y):
        def fn(params, x, y):
            mean, log_std = policy_model.mean_and_logstd(x, params)
            return policy_model.log_value(y, mean, log_std)

        return jax.vmap(fn, in_axes=(None, 0, 0))(model_params, X, Y)

    return jax.jit(pred_fn)


@functools.partial(jax.jit, static_argnums=0)
def train_each_model(predfn_and_opt, opt_state, params, perm, dataset, log_rewards):
    (pred_fn, opt) = predfn_and_opt
    X, Y = dataset

    @jax.jit
    def train_batch(carry, p):
        opt_state, params = carry
        batch_x, batch_y, batch_logrwd = X[p], Y[p], log_rewards[p]
        batch_rwd = jnp.exp(
            batch_logrwd - jnp.max(batch_logrwd)
        )  # divide by the max value, the equation remains the same

        def loss_fn(params):
            log_value = pred_fn(params, batch_x, batch_y)
            return -jnp.mean(batch_rwd * log_value)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return (opt_state, params), loss

    (opt_state, params), losses = jax.lax.scan(train_batch, (opt_state, params), perm)
    return opt_state, params, jnp.mean(losses)


def train_epoch(pred_fn, opt_tuple, params, perm, dataset, log_rewards):
    opt, opt_states = opt_tuple

    updated_opt_states, updated_params, losses = [], [], []
    for opt_state, subparams, p in zip(opt_states, params, perm):
        opt_state, subparams, loss = train_each_model(
            predfn_and_opt=(pred_fn, opt),
            opt_state=opt_state,
            params=subparams,
            perm=p,
            dataset=dataset,
            log_rewards=log_rewards,
        )
        updated_opt_states.append(opt_state)
        updated_params.append(subparams)
        losses.append(loss)

    epoch_loss = jnp.mean(jnp.array(losses))
    return updated_opt_states, updated_params, epoch_loss


def train(
    evaluate_fn,
    mean_policy,
    opt_tuple,
    params,
    dataset,
    key,
    batch_size,
    num_epochs,
    algo,
    rwd_update,
):
    k = mean_policy.k
    datasize = dataset[0].shape[0]
    steps_per_model = datasize // (k * batch_size)
    opt, opt_states = opt_tuple
    pred_fn = get_pred_fn(mean_policy)
    log_rewards = jnp.zeros(datasize)

    losses, eval_rewards = [], []
    for ep in range(num_epochs):
        key, subkey = jax.random.split(key)
        perm = jax.random.choice(
            subkey, datasize, shape=(k, steps_per_model, batch_size), replace=False
        )
        if (algo == "bcnd") and (((ep + 1) % rwd_update) == 0):
            log_rewards = create_log_rewards(mean_policy, dataset, params)
        opt_states, params, loss = train_epoch(
            pred_fn=pred_fn,
            opt_tuple=(opt, opt_states),
            params=params,
            perm=perm,
            dataset=dataset,
            log_rewards=log_rewards,
        )
        losses.append(loss)
        if ((ep + 1) % 20) == 0:
            key, subkey = jax.random.split(key)
            eval_rwd, eval_std, eval_min, eval_max = evaluate_fn(params, subkey)
            eval_rewards.append((eval_rwd, eval_std, eval_min, eval_max))
            print(
                f"epoch: {ep + 1}, loss: {loss:.5f}, eval_reward: {eval_rwd:.3f}, "
                + f"eval_std: {eval_std:.3f}, eval_min: {eval_min:.3f}, eval_max: {eval_max:.3f}"
            )
    return params, losses, eval_rewards


def create_opts_params(mean_policy, key, learning_rate):
    params = mean_policy.initialize_params(key)
    opt = optax.adamw(learning_rate=learning_rate)
    opt_states = [opt.init(p) for p in params]
    return opt, opt_states, params


def get_normalize_fn(data, normalize):
    if not normalize:
        return lambda x: x

    mean_data = jnp.mean(data, axis=0)
    std_data = jnp.mean(data, axis=0)

    def normalize_fn(X):
        return (X - mean_data) / std_data

    return jax.jit(normalize_fn)


def main(
    seed,
    env,
    noise_name,
    noise_level,
    k,
    batch,
    epochs,
    algo,
    rwd_update=100,
    normalize=False,
):
    current_file_path = os.path.dirname(__file__)
    dataset_path = f"{current_file_path}/noisy_data/{env}/expert-{noise_name}/{noise_level}/trajectories.json"
    X, Y = get_trajectory_dataset(dataset_path)
    normalize_fn = get_normalize_fn(X, normalize)
    norm_X, norm_Y = normalize_fn(X), Y  # is assumed to be in between (-1, 1)
    key = jax.random.PRNGKey(seed=seed)
    policy_model = PolicyModel(xsize=X.shape[-1], usize=Y.shape[-1])
    mean_policy = MeanPolicy(k=k, polciy_model=policy_model)
    learning_rate = k * 1e-4
    opt, opt_states, params = create_opts_params(mean_policy, key, learning_rate)
    env = envs.create(env_name=env, backend="positional")
    env_tuples = jax.jit(env.reset), jax.jit(env.step)
    num_evals = 3
    evaluate_fn = functools.partial(
        evaluate, env_tuples, normalize_fn, mean_policy, num_evals
    )
    params, losses, eval_rewards = train(
        evaluate_fn=evaluate_fn,
        mean_policy=mean_policy,
        opt_tuple=(opt, opt_states),
        params=params,
        dataset=(norm_X, norm_Y),
        key=key,
        batch_size=batch,
        num_epochs=epochs,
        algo=algo,
        rwd_update=rwd_update,
    )
    return params, losses, eval_rewards


if __name__ == "__main__":
    parser = get_argparser()
    args = parser.parse_args()

    print(args)

    _, _, eval_rewards = main(
        seed=args.seed,
        env=args.env,
        noise_name=args.noise_name,
        noise_level=args.noise_level,
        k=args.k,
        batch=args.batch,
        epochs=args.epochs,
        algo=args.algo,
        rwd_update=args.rwd_update,
        normalize=args.normalize,
    )

    print(
        f"Avg reward for {args.env} under {args.noise_level} epsilon {args.noise_name} policy: {eval_rewards[-1][0]:.3f} +- {eval_rewards[-1][1]:.3f}"
    )
