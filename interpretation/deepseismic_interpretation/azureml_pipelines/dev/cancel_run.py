"""
Cancel pipeline run
"""
from azureml.core.run import Run
from azureml.core import Workspace, Experiment
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--run_id', type=str, help='run id value', required=True)
parser.add_argument('--step_id', type=str, help='step id value', required=True)

args = parser.parse_args()

ws = Workspace.from_config()

experiment = Experiment(workspace=ws, name="DEV-train-pipeline", _id=args.run_id)
fetched_run = Run(experiment=experiment, run_id=args.step_id)
fetched_run.cancel()
