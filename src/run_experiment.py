import jax

import yaml
import os
import ipdb

import simulators

jax.config.update('jax_platform_name', 'cpu')


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    system = config["system"]["type"]

    if system == 'beacon':
        simulator = simulators.BeaconSimulator(dt=config["system"]["dt"])
    elif system == 'x-plane':
        raise NotImplementedError("X-Plane simulator not implemented yet.")
    else:
        raise ValueError(f"Unknown system type: {system}")

    results_dict = simulator.run(
        num_steps=config["experiment_setup"]["num_steps"],
        filtering=config["hjnnv"]["filtering"],
        bounding_method=config["hjnnv"]["bounding_method"],
        experiment_number=config["experiment_setup"]["expt_number"]
    )

    simulator.plot(
        results_dict,
        experiment_number=config["experiment_setup"]["expt_number"],
        show=config["experiment_setup"]["show_plot"],
        save=config["experiment_setup"]["save"],
    )


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
