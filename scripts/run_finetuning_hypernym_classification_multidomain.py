import argparse
import sys
import os
import sys
import time
import json

from typing import Dict, Any
sys.path.append("./")

parser = argparse.ArgumentParser()
parser.add_argument("--experiment-name", type=str)

# Leaving these duplicate arguments for now, for quick validation and manual resuming from checkpoints.
parser.add_argument("--validate-only", type=str, default="False")
parser.add_argument("--reload-from-checkpoint", type=str, default="False")
parser.add_argument("--reload-from-epoch-num", type=str, default="0")
args = parser.parse_args()

with open(
    os.path.join("../ctp/experiment_configs", args.experiment_name + ".json")
) as experiment_json_file:
    experiment_parameters = json.load(experiment_json_file)


def command_from_params(experiment_parameters: Dict[str, Any]) -> str:
    """
    Take in the experiment parameters dictionary and create the python command that
    runs the experiment. Use the parameteres from the dictionary if provided, otherwise,
    use the default parameters.
    """

    # Required parameters
    train_filename = experiment_parameters["train_filename"]
    test_filenames = experiment_parameters["test_filenames"]
    model = experiment_parameters["model"]

    base_command = f"python -u ctp/finetuning_hypernym_classification.py --train-filename {train_filename} --test-filenames {test_filenames} --experiment-name {args.experiment_name} --validate-only {args.validate_only} --reload-from-epoch-num {args.reload_from_epoch_num} --reload-from-checkpoint {args.reload_from_checkpoint} "

    # Add parameters that override the defaults
    for parameter_name, parameter_value in experiment_parameters.items():
        # This should probably be deleted from the experiment configs
        if parameter_name == "predicted_labels":
            continue
        base_command += f' --{parameter_name.replace("_", "-")} {parameter_value} '

    log_command = f'>> ./outputs/logs/output_{args.experiment_name}_validate_{args.validate_only}_{time.strftime("%y%m%d")}.log'

    return base_command #+ log_command


cmd = command_from_params(experiment_parameters)
print(cmd)
os.chdir("../")
os.system(cmd)
