# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Create pipeline and kickoff run
"""
from deepseismic_interpretation.azureml_pipelines.train_pipeline import TrainPipeline
import fire


def kickoff_pipeline(
    experiment="DEV-train-pipeline",
    orchestrator_config="interpretation/deepseismic_interpretation/azureml_pipelines/pipeline_config.json",
):
    """Kicks off pipeline run

    Args:
        experiment (str): name of experiment
        orchestrator_config (str): path to pipeline configuration
    """
    orchestrator = TrainPipeline(orchestrator_config)
    orchestrator.construct_pipeline()
    run = orchestrator.run_pipeline(experiment_name=experiment)


if __name__ == "__main__":
    """Example:
    python interpretation/deepseismic_interpretation/azureml_pipelines/dev/kickoff_train_pipeline.py --experiment=DEV-train-pipeline-name --orchestrator_config=orchestrator_config="interpretation/deepseismic_interpretation/azureml_pipelines/pipeline_config.json"
    or
    python interpretation/deepseismic_interpretation/azureml_pipelines/dev/kickoff_train_pipeline.py 

    """
    fire.Fire(kickoff_pipeline)
