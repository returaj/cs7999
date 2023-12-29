#! /usr/bin/env python3

import os
import json

import BCND.train_policy_v1 as train_policy_v1


def make_all_val_float(list_data):
    return [float(x) for x in list_data]


def save_data(path, data):
    if not os.path.exists(path):
        os.makedirs(path)
    path = f"{path}/meta_data.json"
    with open(path, "w") as f:
        json.dump(data, f)


def main(env, noise_name, k, batch, epochs, algo):
    seeds = [0, 1, 2, 3, 4]
    noise_levels = ["0.0", "0.1", "0.2", "0.3", "0.4", "0.5"]

    meta_data = []
    for noise_level in noise_levels:
        noise_level_meta_data = []
        for seed in seeds:
            print("-" * 20)
            print(
                f"seed: {seed}, env: {env}, noise_name: {noise_name}, noise_level: {noise_level}, k: {k}, algo: {algo}\n"
            )
            _, losses, eval_rewards = train_policy_v1.main(
                seed=seed,
                env=env,
                noise_name=noise_name,
                noise_level=noise_level,
                k=k,
                batch=batch,
                epochs=epochs,
                algo=algo,
            )
            losses = make_all_val_float(losses)
            eval_rewards = make_all_val_float(eval_rewards)
            noise_level_meta_data.append(
                {
                    "seed": seed,
                    "losses": losses,
                    "eval_rewards": eval_rewards,
                }
            )
        meta_data.append(
            {"noise_level": noise_level, "meta_data": noise_level_meta_data}
        )

    store_data = {
        "env": env,
        "noise_name": noise_name,
        "k": k,
        "algo": algo,
        "batch": batch,
        "epochs": epochs,
        "meta_data": meta_data,
    }

    current_file_path = os.path.dirname(__file__)
    path = f"{current_file_path}/noisy_data/{env}/expert-{noise_name}/{algo}-{k}"
    save_data(path, store_data)


if __name__ == "__main__":
    parser = train_policy_v1.get_argparser()
    # in args define only env, noise_name, k, algo
    args = parser.parse_args()
    main(
        env=args.env,
        noise_name=args.noise_name,
        k=args.k,
        algo=args.algo,
        batch=args.batch,
        epochs=args.epochs,
    )
