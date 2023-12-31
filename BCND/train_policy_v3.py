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
    parser.add_argument(
        "--iterations", type=int, default=5, help="number of repeat iteration"
    )
    parser.add_argument("--env", type=str, default="ant", help="environment name")
    parser.add_argument("--noise_name", type=str, default="normal", help="noise name")
    parser.add_argument("--noise_level", type=str, default="0.1", help="noise level")
    parser.add_argument("--k", type=int, default=5, help="number of ensembles")
    parser.add_argument(
        "--k_rwd",
        type=int,
        default=5,
        help="number of models required for calculating reward",
    )
    parser.add_argument("--algo", type=str, default="bc", help="algorithm type")
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


def get_pred_fn(policy_model):
    def pred_fn(model_params, X, Y):
        def fn(params, x, y):
            mean, log_std = policy_model.mean_and_logstd(x, params)
            return policy_model.log_value(y, mean, log_std)

        return jax.vmap(fn, in_axes=(None, 0, 0))(model_params, X, Y)

    return jax.jit(pred_fn)


def create_opt_params(policy_model, key, learning_rate):
    params = policy_model.initialize_params(key)
    opt = optax.adamw(learning_rate=learning_rate)
    opt_state = opt.init(params)
    return opt, opt_state, params


def create_log_rewards_fn(mean_policy, dataset, k=-1):
    policy_model = mean_policy.policy_model
    X, Y = dataset
    k = mean_policy.k if k < 0 else k

    @jax.jit
    def calculate_log_reward_per_model(X, Y, model_params):
        def fn(x, y, params):
            mean, logstd = policy_model.mean_and_logstd(x, params)
            return policy_model.log_value(y, mean, logstd)

        return jax.vmap(fn, in_axes=(0, 0, None))(X, Y, model_params)

    def calculate_mean_log_rewards(params):
        val = []
        for i in range(k):
            val.append(calculate_log_reward_per_model(X, Y, params[i]))
        return jnp.mean(jnp.array(val), axis=0)

    return calculate_mean_log_rewards


@functools.partial(jax.jit, static_argnums=0)
def train_each_model_per_epoch(
    predfn_and_opt, opt_state, params, perm, dataset, log_rewards
):
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


def train_models(
    policy_model,
    key,
    learning_rate,
    batch_size,
    num_epochs,
    dataset,
    log_rewards,
    k_perm,
    iteration_cnt,
):
    X, Y = dataset
    model_dataset_size = k_perm.shape[-1]
    steps_per_epoch = model_dataset_size // batch_size
    pred_fn = get_pred_fn(policy_model)
    all_params = []
    for k, model_perm in enumerate(k_perm):
        model_dataset = (X[model_perm], Y[model_perm])
        model_logrewards = log_rewards[model_perm]
        key, subkey = jax.random.split(key)
        opt, opt_state, params = create_opt_params(policy_model, subkey, learning_rate)

        print("-" * 30)
        for ep in range(num_epochs):
            key, subkey = jax.random.split(key)
            perm = jax.random.choice(
                subkey,
                model_dataset_size,
                shape=(steps_per_epoch, batch_size),
                replace=False,
            )
            opt_state, params, loss = train_each_model_per_epoch(
                predfn_and_opt=(pred_fn, opt),
                opt_state=opt_state,
                params=params,
                perm=perm,
                dataset=model_dataset,
                log_rewards=model_logrewards,
            )
            if ((ep + 1) % 100) == 0:
                print(
                    f"Itr: {iteration_cnt}, Update for model: {k}, epoch: {ep + 1}, loss: {loss:.5f}"
                )
        all_params.append(params)
    return all_params


def train(
    evaluate_fn,
    mean_policy,
    key,
    batch_size,
    num_epochs,
    num_iterations,
    k_rwd,
    dataset,
):
    key, subkey = jax.random.split(key)
    num_models = mean_policy.k
    datasize = dataset[0].shape[0]
    model_datasize = datasize // num_models
    k_perm = jax.random.choice(
        subkey, datasize, shape=(num_models, model_datasize), replace=False
    )
    log_reward_fn = create_log_rewards_fn(mean_policy, dataset, k=k_rwd)
    log_rewards = jnp.zeros(datasize)
    eta = 1

    eval_rewards = []
    for itr in range(num_iterations):
        key, trainkey, evalkey = jax.random.split(key, 3)
        all_params = train_models(
            policy_model=mean_policy.policy_model,
            key=trainkey,
            learning_rate=eta * 1e-4,
            batch_size=batch_size,
            num_epochs=num_epochs,
            dataset=dataset,
            log_rewards=log_rewards,
            k_perm=k_perm,
            iteration_cnt=itr,
        )
        log_rewards = log_reward_fn(all_params)
        # eta = jnp.max(jnp.exp(-logmeanexp(log_rewards)), 1e4)
        eval_rwd, eval_std, eval_min, eval_max = evaluate_fn(all_params, evalkey)
        eval_rewards.append((eval_rwd, eval_std, eval_min, eval_max))
        print(
            f"iteration: {itr + 1}, eval_reward: {eval_rwd:.3f}, eval_std: {eval_std:.3f}, "
            + f"eval_min: {eval_min:.3f}, eval_max: {eval_max:.3f}"
        )

    return eval_rewards


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
    num_iterations=5,
    k_rwd=5,
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
    env = envs.create(env_name=env, backend="positional")
    env_tuples = jax.jit(env.reset), jax.jit(env.step)
    num_evals = 5
    evaluate_fn = functools.partial(
        evaluate, env_tuples, normalize_fn, mean_policy, num_evals
    )
    num_iterations = 1 if algo == "bc" else num_iterations
    eval_rewards = train(
        evaluate_fn=evaluate_fn,
        mean_policy=mean_policy,
        key=key,
        batch_size=batch,
        num_epochs=epochs,
        num_iterations=num_iterations,
        k_rwd=k_rwd,
        dataset=(norm_X, norm_Y),
    )
    return eval_rewards


if __name__ == "__main__":
    parser = get_argparser()
    args = parser.parse_args()

    print(args)

    eval_rewards = main(
        seed=args.seed,
        env=args.env,
        noise_name=args.noise_name,
        noise_level=args.noise_level,
        k=args.k,
        batch=args.batch,
        epochs=args.epochs,
        algo=args.algo,
        num_iterations=args.iterations,
        k_rwd=args.k_rwd,
        normalize=args.normalize,
    )

    print(
        f"Avg reward for {args.env} under {args.noise_level} epsilon {args.noise_name} policy: {eval_rewards[-1][0]:.3f} +- {eval_rewards[-1][1]:.3f}"
    )
