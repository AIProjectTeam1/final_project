import wandb
import yaml
import argparse
from multi_prompt_train import train

project_name = "resume_eval"

def run_sweep(sweep_config_file):
    with open(sweep_config_file, 'r') as f:
        sweep_config = yaml.safe_load(f)

    sweep_id = wandb.sweep(sweep=sweep_config, project=project_name)
    print(f"Sweep ID: {sweep_id}")

    wandb.agent(sweep_id, function=train_with_sweep)

def train_with_sweep():
    wandb.init(project=project_name)
    config = wandb.config
    train(config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="model/sweep_config.yaml")
    parser.add_argument("--project", type=str, default="resume_eval")
    args = parser.parse_args()
    project_name = args.project

    run_sweep(args.config)
