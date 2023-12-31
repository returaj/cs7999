#! /usr/bin/env python3

import os
import json
import jax


def collect_data(key, expert_data, noise_data, ep):
    num_episode = len(expert_data)
    uniform_samples = jax.random.uniform(key, shape=(num_episode,))
    mix_data = []
    for i, z in enumerate(uniform_samples):
        if z < ep:
            mix_data.append(noise_data[i])
        else:
            mix_data.append(expert_data[i])
    return mix_data


def load_data(path):
    with open(path, "r") as f:
        d = json.load(f)
    return d


def save_data(path, data):
    if not os.path.exists(path):
        os.makedirs(path)
    path = f"{path}/trajectories.json"
    with open(path, "w") as f:
        json.dump(data, f)


def main(expert_data_path, noise_data_path, save_path, ep, seed):
    expert_data = load_data(expert_data_path)
    noise_data = load_data(noise_data_path)
    mix_data = collect_data(
        key=jax.random.PRNGKey(seed),
        expert_data=expert_data,
        noise_data=noise_data,
        ep=ep,
    )

    save_path = f"{save_path}/{ep}"
    save_data(save_path, mix_data)


if __name__ == "__main__":
    env_name = "ant"
    eps = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    name_and_seed = [("random", 1), ("normal", 1), ("zero", 1)]

    current_file_path = os.path.dirname(__file__)
    data_path = f"{current_file_path}/../trajectories/data"
    expert_data_path = f"{data_path}/{env_name}/expert/trajectories.json"

    for ep in eps:
        for noise_name, seed in name_and_seed:
            noise_data_path = f"{data_path}/{env_name}/{noise_name}/trajectories.json"
            save_data_path = (
                f"{current_file_path}/noisy_data/{env_name}/expert-{noise_name}"
            )
            main(
                expert_data_path=expert_data_path,
                noise_data_path=noise_data_path,
                save_path=save_data_path,
                ep=ep,
                seed=seed,
            )
