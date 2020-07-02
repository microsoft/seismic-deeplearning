# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
base class for constructing and running an azureml pipeline and some of the
accompanying resources.
"""
from azureml.core import Datastore, Workspace, RunConfiguration
from azureml.core.model import Model
from azureml.core.compute import AmlCompute, ComputeTarget
from azureml.core.dataset import Dataset
from azureml.core.experiment import Experiment
from azureml.pipeline.steps import PythonScriptStep, MpiStep
from azureml.pipeline.core import Pipeline, PipelineData, StepSequence
from azureml.contrib.pipeline.steps import ParallelRunStep, ParallelRunConfig
from azureml.core.runconfig import DEFAULT_GPU_IMAGE
from azureml.core.conda_dependencies import CondaDependencies
from msrest.exceptions import HttpOperationError
from azureml.data.data_reference import DataReference
from azureml.core import Environment
from dotenv import load_dotenv
import os
import re
from abc import ABC, abstractmethod
import json


class DeepSeismicAzMLPipeline(ABC):
    """
    Abstract base class for pipelines in AzureML
    """

    def __init__(self, pipeline_config, ws_config=None):
        """
        constructor for DeepSeismicAzMLPipeline class

        :param str pipeline_config: [required] path to the pipeline config file
        :param str ws_config: [optional] if not specified, will look for
                              .azureml/config.json. If you have multiple config files, you
                              can specify which workspace you want to use by passing the
                              relative path to the config file in this constructor.
        """
        self.ws = Workspace.from_config(path=ws_config)
        self._load_environment()
        self._load_config(pipeline_config)
        self.steps = []
        self.pipeline_tags = None
        self.last_output_data = None

    def _load_config(self, config_path):
        """
        helper function for loading in pipeline config file.

        :param str config_path: path to the pipeline config file
        """
        try:
            with open(config_path, "r") as f:
                self.config = json.load(f)
        except Exception as e:
            raise Exception("Was unable to load pipeline config file. {}".format(e))

    @abstractmethod
    def construct_pipeline(self):
        """
        abstract method for constructing a pipeline. Must be implemented by classes
        that inherit from this base class.
        """
        raise NotImplementedError("construct_pipeline is not implemented")

    @abstractmethod
    def _setup_steps(self):
        """
        abstract method for setting up pipeline steps. Must be implemented by classes
        that inherit from this base class.
        """
        raise NotImplementedError("setup_steps is not implemented")

    def _load_environment(self):
        """
        loads environment variables needed for the pipeline.
        """
        load_dotenv()
        self.account_name = os.getenv("BLOB_ACCOUNT_NAME")
        self.container_name = os.getenv("BLOB_CONTAINER_NAME")
        self.account_key = os.getenv("BLOB_ACCOUNT_KEY")
        self.blob_sub_id = os.getenv("BLOB_SUB_ID")

        self.comp_name = os.environ.get("AML_COMPUTE_CLUSTER_NAME")
        self.comp_min_nodes = os.environ.get("AML_COMPUTE_CLUSTER_MIN_NODES")
        self.comp_max_nodes = os.environ.get("AML_COMPUTE_CLUSTER_MAX_NODES")
        self.comp_vm_size = os.environ.get("AML_COMPUTE_CLUSTER_SKU")

    def _setup_model(self, model_name, model_path=None):
        """
        sets up the model in azureml. Either retrieves an already registered model
        or registers a local model.

        :param str model_name: [required] name of the model that you want to retrieve
                               from the workspace or the name you want to give the local
                               model when you register it.
        :param str model_path: [optional] If you do not have a model registered, pass
                               the relative path to the model locally and it will be
                               registered.
        """
        models = Model.list(self.ws, name=model_name)
        for model in models:
            if model.name == model_name:
                self.model = model
                print("Found model: " + self.model.name)
                break

        if model_path is not None:
            self.model = Model.register(model_path=model_path, model_name=model_name, workspace=self.ws)

        if self.model is None:
            raise Exception(
                """no model was found or registered. Ensure that you
                             have a model registered in this workspace or that
                             you passed the path of a local model"""
            )

    def _setup_datastore(self, blob_dataset_name, output_path=None):
        """
        sets up the datastore in azureml. Either retrieves a pre-existing datastore
        or registers a new one in the workspace.

        :param str blob_dataset_name: [required] name of the datastore registered with the
                                 workspace. If the datastore does not yet exist, the
                                 name it will be registered under.
        :param str output_path: [optional] if registering a datastore for inferencing,
                                the output path for writing back predictions.
        """
        try:
            self.blob_ds = Datastore.get(self.ws, blob_dataset_name)
            print("Found Blob Datastore with name: %s" % blob_dataset_name)
        except HttpOperationError:
            self.blob_ds = Datastore.register_azure_blob_container(
                workspace=self.ws,
                datastore_name=blob_dataset_name,
                account_name=self.account_name,
                container_name=self.container_name,
                account_key=self.account_key,
                subscription_id=self.blob_sub_id,
            )

            print("Registered blob datastore with name: %s" % blob_dataset_name)
        if output_path is not None:
            self.output_dir = PipelineData(
                name="output", datastore=self.ws.get_default_datastore(), output_path_on_compute=output_path
            )

    def _setup_dataset(self, ds_name, data_paths):
        """
        registers datasets with azureml workspace

        :param str ds_name: [required] name to give the dataset in azureml.
        :param str data_paths: [required] list of paths to your data on the datastore.
        """
        self.named_ds = []
        count = 1
        for data_path in data_paths:
            curr_name = ds_name + str(count)
            path_on_datastore = self.blob_ds.path(data_path)
            input_ds = Dataset.File.from_files(path=path_on_datastore, validate=False)
            try:
                registered_ds = input_ds.register(workspace=self.ws, name=curr_name, create_new_version=True)
            except Exception as e:
                n, v = self._parse_exception(e)
                registered_ds = Dataset.get_by_name(self.ws, name=n, version=v)
            self.named_ds.append(registered_ds.as_named_input(curr_name))
            count = count + 1

    def _setup_datareference(self, name, path):
        """
        helper function to setup a datareference object in AzureML.

        :param str name: [required] name of the data reference\
        :param str path: [required] path on the datastore where the data lives.
        :returns: input_data
        :rtype: DataReference
        """
        input_data = DataReference(datastore=self.blob_ds, data_reference_name=name, path_on_datastore=path)
        return input_data

    def _setup_pipelinedata(self, name, output_path=None):
        """
        helper function to setup a PipelineData object in AzureML

        :param str name: [required] name of the data object in AzureML
        :param str output_path: path on output datastore to write data to
        :returns: output_data
        :rtype: PipelineData
        """
        if output_path is not None:
            output_data = PipelineData(
                name=name,
                datastore=self.blob_ds,
                output_name=name,
                output_mode="mount",
                output_path_on_compute=output_path,
                is_directory=True,
            )
        else:
            output_data = PipelineData(name=name, datastore=self.ws.get_default_datastore(), output_name=name)
        return output_data

    def _setup_compute(self):
        """
        sets up the compute in the azureml workspace. Either retrieves a
        pre-existing compute target or creates one (uses environment variables).

        :returns: compute_target
        :rtype: ComputeTarget
        """
        if self.comp_name in self.ws.compute_targets:
            self.compute_target = self.ws.compute_targets[self.comp_name]
            if self.compute_target and type(self.compute_target) is AmlCompute:
                print("Found compute target: " + self.comp_name)
        else:
            print("creating a new compute target...")
            p_cfg = AmlCompute.provisioning_configuration(
                vm_size=self.comp_vm_size, min_nodes=self.comp_min_nodes, max_nodes=self.comp_max_nodes
            )

            self.compute_target = ComputeTarget.create(self.ws, self.comp_name, p_cfg)
            self.compute_target.wait_for_completion(show_output=True, min_node_count=None, timeout_in_minutes=20)

            print(self.compute_target.get_status().serialize())
        return self.compute_target

    def _get_conda_deps(self, step):
        """
        converts requirements.txt from user into conda dependencies for AzML

        :param dict step: step defined by user that we are currently building

        :returns: conda_dependencies
        :rtype: CondaDependencies
        """
        with open(step["requirements"], "r") as f:
            packages = [line.strip() for line in f]

        return CondaDependencies.create(pip_packages=packages)

    def _setup_env(self, step):
        """
        sets up AzML env given requirements defined by the user

        :param dict step: step defined by user that we are currently building

        :returns: env
        :rtype: Environment
        """
        conda_deps = self._get_conda_deps(step)

        env = Environment(name=step["name"] + "_environment")
        env.docker.enabled = True

        env.docker.base_image = DEFAULT_GPU_IMAGE
        env.spark.precache_packages = False
        env.python.conda_dependencies = conda_deps
        env.python.conda_dependencies.add_conda_package("pip==20.0.2")
        return env

    def _generate_run_config(self, step):
        """
        generates an AzML run config if the user gives specifics about requirements

        :param dict step: step defined by user that we are currently building

        :returns: run_config
        :rtype: RunConfiguration
        """
        try:
            conda_deps = self._get_conda_deps(step)
            conda_deps.add_conda_package("pip==20.0.2")
            return RunConfiguration(script=step["script"], conda_dependencies=conda_deps)
        except KeyError:
            return None

    def _generate_parallel_run_config(self, step):
        """
        generates an AzML parralell run config if the user gives specifics about requirements

        :param dict step: step defined by user that we are currently building

        :returns: parallel_run_config
        :rtype: ParallelRunConfig
        """
        return ParallelRunConfig(
            source_directory=step["source_directory"],
            entry_script=step["script"],
            mini_batch_size=str(step["mini_batch_size"]),
            error_threshold=10,
            output_action="summary_only",
            environment=self._setup_env(step),
            compute_target=self.compute_target,
            node_count=step.get("node_count", 1),
            process_count_per_node=step.get("processes_per_node", 1),
            run_invocation_timeout=60,
        )

    def _create_pipeline_step(self, step, arguments, input_data, output=None, run_config=None):
        """
        function to create an AzureML pipeline step and apend it to the list of
        steps that will make up the pipeline.

        :param dict step: [required] dictionary containing the config parameters for this step.
        :param list arguments: [required] list of arguments to be passed to the step.
        :param DataReference input_data: [required] the input_data in AzureML for this step.
        :param DataReference output: [required] output location in AzureML
        :param ParallelRunConfig run_config: [optional] the run configuration for a MpiStep
        """

        if step["type"] == "PythonScriptStep":
            run_config = self._generate_run_config(step)
            pipeline_step = PythonScriptStep(
                script_name=step["script"],
                arguments=arguments,
                inputs=[input_data],
                outputs=output,
                name=step["name"],
                compute_target=self.compute_target,
                source_directory=step["source_directory"],
                allow_reuse=True,
                runconfig=run_config,
            )

        elif step["type"] == "MpiStep":
            pipeline_step = MpiStep(
                name=step["name"],
                source_directory=step["source_directory"],
                arguments=arguments,
                inputs=[input_data],
                node_count=step.get("node_count", 1),
                process_count_per_node=step.get("processes_per_node", 1),
                compute_target=self.compute_target,
                script_name=step["script"],
                environment_definition=self._setup_env(step),
            )

        elif step["type"] == "ParallelRunStep":
            run_config = self._generate_parallel_run_config(step)

            pipeline_step = ParallelRunStep(
                name=step["name"],
                models=[self.model],
                parallel_run_config=run_config,
                inputs=input_data,
                output=output,
                arguments=arguments,
                allow_reuse=False,
            )
        else:
            raise Exception("Pipeline step type {} not supported".format(step["type"]))

        self.steps.append(pipeline_step)

    def run_pipeline(self, experiment_name, tags=None):
        """
        submits batch inference pipeline as an experiment run

        :param str experiment_name: [required] name of the experiment in azureml
        :param dict tags: [optional] dictionary of tags
        :returns: run
        :rtype: Run
        """
        if tags is None:
            tags = self.pipeline_tags
        step_sequence = StepSequence(steps=self.steps)
        pipeline = Pipeline(workspace=self.ws, steps=step_sequence)
        run = Experiment(self.ws, experiment_name).submit(pipeline, tags=tags, continue_on_step_failure=False)
        return run

    def _parse_exception(self, e):
        """
        helper function to parse exception thrown by azureml

        :param Exception e: [required] the exception to be parsed
        :returns: name, version
        :rtype: str, str
        """
        s = str(e)
        result = re.search('name="(.*)"', s)
        name = result.group(1)
        version = s[s.find("version=") + 8 : s.find(")")]

        return name, version
