import argparse
from pathlib import Path

from src.experiment_runner import run_experiment, run_many
from src.utils import read_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--params", type=str, default="params.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params = read_yaml(args.params) if Path(args.params).exists() else {}
    if args.config:
        run_experiment(args.config, params)
        return
    config_paths = params.get("run_all", [])
    if config_paths:
        run_many(config_paths, params)


if __name__ == "__main__":
    main()
