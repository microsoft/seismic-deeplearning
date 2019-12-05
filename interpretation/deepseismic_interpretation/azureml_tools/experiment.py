# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import logging
import logging.config
import os

import azureml.core
from azure.common.credentials import get_cli_profile
from azureml.core import Datastore, Environment
from azureml.core.compute import AmlCompute, ComputeTarget
from azureml.core.compute_target import ComputeTargetException
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.runconfig import EnvironmentDefinition
from azureml.train.dnn import Gloo, Nccl, PyTorch
from toolz import curry

from deepseismic_interpretation.azureml_tools import workspace_for_user
from deepseismic_interpretation.azureml_tools.config import experiment_config
from deepseismic_interpretation.azureml_tools.resource_group import create_resource_group
from deepseismic_interpretation.azureml_tools.storage import create_premium_storage
from deepseismic_interpretation.azureml_tools.subscription import select_subscription

_GPU_IMAGE = "mcr.microsoft.com/azureml/base-gpu:openmpi3.1.2-cuda10.0-cudnn7-ubuntu16.04"


def _create_cluster(workspace, cluster_name, vm_size, min_nodes, max_nodes):
    """Creates AzureML cluster

    Args:
        cluster_name (string): The name you wish to assign the cluster.
        vm_size (string): The type of sku to use for your vm.
        min_nodes (int): Minimum number of nodes in cluster.
                                    Use 0 if you don't want to incur costs when it isn't being used.
        max_nodes (int): Maximum number of nodes in cluster.

    """
    logger = logging.getLogger(__name__)
    try:
        compute_target = ComputeTarget(workspace=workspace, name=cluster_name)
        logger.info("Found existing compute target.")
    except ComputeTargetException:
        logger.info("Creating a new compute target...")
        compute_config = AmlCompute.provisioning_configuration(
            vm_size=vm_size, min_nodes=min_nodes, max_nodes=max_nodes
        )

        # create the cluster
        compute_target = ComputeTarget.create(workspace, cluster_name, compute_config)
        compute_target.wait_for_completion(show_output=True)

    # use get_status() to get a detailed status for the current AmlCompute.
    logger.debug(compute_target.get_status().serialize())

    return compute_target


@curry
def _create_estimator(
    estimator_class, project_folder, entry_script, compute_target, script_params, node_count, env_def, distributed,
):
    logger = logging.getLogger(__name__)

    estimator = estimator_class(
        project_folder,
        entry_script=entry_script,
        compute_target=compute_target,
        script_params=script_params,
        node_count=node_count,
        environment_definition=env_def,
        distributed_training=distributed,
    )

    logger.debug(estimator.conda_dependencies.__dict__)
    return estimator


def _create_datastore(
    aml_workspace, datastore_name, container_name, account_name, account_key, create_if_not_exists=True,
):
    """Creates datastore

    Args:
        datastore_name (string): Name you wish to assign to your datastore.
        container_name (string): Name of your container.
        account_name (string): Storage account name.
        account_key (string): The storage account key.

    Returns:
        azureml.core.Datastore
    """
    logger = logging.getLogger(__name__)
    ds = Datastore.register_azure_blob_container(
        workspace=aml_workspace,
        datastore_name=datastore_name,
        container_name=container_name,
        account_name=account_name,
        account_key=account_key,
        create_if_not_exists=create_if_not_exists,
    )
    logger.info(f"Registered existing blob storage: {ds.name}.")
    return ds


def _check_subscription_id(config):
    if config.SUBSCRIPTION_ID is None:
        profile = select_subscription()
        config.SUBSCRIPTION_ID = profile.get_subscription_id()
    return True, f"Selected subscription id is {config.SUBSCRIPTION_ID}"


_CHECK_FUNCTIONS = (_check_subscription_id,)


class ConfigError(Exception):
    pass


def _check_config(config):
    logger = logging.getLogger(__name__)
    check_gen = (f(config) for f in _CHECK_FUNCTIONS)
    check_results = list(filter(lambda state_msg: state_msg[0] == False, check_gen))
    if len(check_results) > 0:
        error_msgs = "\n".join([msg for state, msg in check_results])
        msg = f"Config failed \n {error_msgs}"
        logger.info(msg)
        raise ConfigError(msg)


class BaseExperiment(object):
    def __init__(self, experiment_name, config=experiment_config):

        self._logger = logging.getLogger(__name__)
        self._logger.info("SDK version:" + str(azureml.core.VERSION))
        _check_config(config)

        profile = select_subscription(sub_name_or_id=config.SUBSCRIPTION_ID)
        profile_credentials, subscription_id, _ = profile.get_login_credentials()
        rg = create_resource_group(profile_credentials, subscription_id, config.REGION, config.RESOURCE_GROUP)
        prem_str, storage_keys = create_premium_storage(
            profile_credentials, subscription_id, config.REGION, config.RESOURCE_GROUP, config.ACCOUNT_NAME,
        )

        self._ws = workspace_for_user(
            workspace_name=config.WORKSPACE,
            resource_group=config.RESOURCE_GROUP,
            subscription_id=config.SUBSCRIPTION_ID,
            workspace_region=config.REGION,
        )
        self._experiment = azureml.core.Experiment(self._ws, name=experiment_name)
        self._cluster = _create_cluster(
            self._ws,
            cluster_name=config.CLUSTER_NAME,
            vm_size=config.CLUSTER_VM_SIZE,
            min_nodes=config.CLUSTER_MIN_NODES,
            max_nodes=config.CLUSTER_MAX_NODES,
        )

        self._datastore = _create_datastore(
            self._ws,
            datastore_name=config.DATASTORE_NAME,
            container_name=config.CONTAINER_NAME,
            account_name=prem_str.name,
            account_key=storage_keys["key1"],
        )

    @property
    def cluster(self):
        return self._cluster

    @property
    def datastore(self):
        return self._datastore


_DISTRIBUTED_DICT = {"nccl": Nccl(), "gloo": Gloo()}


def _get_distributed(distributed_string):
    if distributed_string is not None:
        return _DISTRIBUTED_DICT.get(distributed_string.lower())
    else:
        return None


def create_environment_from_local(name="amlenv", conda_env_name=None):
    """Creates environment from environment

    If no value is passed in to the conda_env_name it will simply select the 
    currently running environment
    
    Args:
        name (str, optional): name of environment. Defaults to "amlenv".
        conda_env_name (str, optional): name of the environment to use. Defaults to None.
    
    Returns:
        azureml.core.Environment
    """
    conda_env_name = os.getenv("CONDA_DEFAULT_ENV") if conda_env_name is None else conda_env_name
    return Environment.from_existing_conda_environment(name, conda_env_name)


def create_environment_from_conda_file(conda_path, name="amlenv"):
    """Creates environment from supplied conda file
    
    Args:
        conda_path (str): path to conda environment file
        name (str, optional): name of environment. Defaults to "amlenv".
    
    Returns:
        azureml.core.Environment
    """
    return Environment.from_existing_conda_specification(name, conda_path)


class PyTorchExperiment(BaseExperiment):
    """Creates Experiment object that can be used to create clusters and submit experiments
    
    Returns:
        PyTorchExperiment: PyTorchExperiment object
    """

    def _complete_datastore(self, script_params):
        def _replace(value):
            if isinstance(value, str) and "{datastore}" in value:
                data_path = value.replace("{datastore}/", "")
                return self.datastore.path(data_path).as_mount()
            else:
                return value

        return {key: _replace(value) for key, value in script_params.items()}

    def submit(
        self,
        project_folder,
        entry_script,
        script_params,
        node_count=1,
        workers_per_node=1,
        distributed=None,
        environment=None,
    ):
        """Submit experiment for remote execution on AzureML clusters.

        Args:
            project_folder (string): Path of you source files for the experiment
            entry_script (string): The filename of your script to run. Must be found in your project_folder
            script_params (dict): Dictionary of script parameters
            dependencies_file (string, optional): The location of your environment.yml to use to
                                                  create the environment your training script requires.
            node_count (int, optional): [description].
            wait_for_completion (bool, optional): Whether to block until experiment is done. Defaults to True.
            docker_args (tuple, optional): Docker arguments to pass. Defaults to ().
        
        Returns:
            azureml.core.Run: AzureML Run object
        """
        self._logger.debug(script_params)

        transformed_params = self._complete_datastore(script_params)
        self._logger.debug("Transformed script params")
        self._logger.debug(transformed_params)

        if environment is None:
            environment = create_environment_from_local()

        environment.docker.shm_size = "8g"
        environment.docker.base_image = _GPU_IMAGE

        estimator = _create_estimator(
            PyTorch,
            project_folder,
            entry_script,
            self.cluster,
            transformed_params,
            node_count,
            environment,
            _get_distributed(distributed),
        )

        self._logger.debug(estimator.conda_dependencies.__dict__)
        return self._experiment.submit(estimator)

