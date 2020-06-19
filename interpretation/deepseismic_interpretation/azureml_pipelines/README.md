# Integrating with AzureML

## AzureML Pipeline Background 

Azure Machine Learning is a cloud-based environment you can use to train, deploy, automate, manage, and track ML models.

An Azure Machine Learning pipeline is an independently executable workflow of a complete machine learning task. Subtasks are encapsulated as a series of steps within the pipeline. An Azure Machine Learning pipeline can be as simple as one that calls a Python script, so may do just about anything. Pipelines should focus on machine learning tasks such as:

- Data preparation including importing, validating and cleaning, munging and transformation, normalization, and staging
- Training configuration including parameterizing arguments, filepaths, and logging / reporting configurations
- Training and validating efficiently and repeatedly. Efficiency might come from specifying specific data subsets, different hardware compute resources, distributed processing, and progress monitoring
- Deployment, including versioning, scaling, provisioning, and access control

An Azure ML pipeline performs a complete logical workflow with an ordered sequence of steps. Each step is a discrete processing action. Pipelines run in the context of an Azure Machine Learning Experiment.
In the early stages of an ML project, it's fine to have a single Jupyter notebook or Python script that does all the work of Azure workspace and resource configuration, data preparation, run configuration, training, and validation. But just as functions and classes quickly become preferable to a single imperative block of code, ML workflows quickly become preferable to a monolithic notebook or script.
By modularizing ML tasks, pipelines support the Computer Science imperative that a component should "do (only) one thing well." Modularity is clearly vital to project success when programming in teams, but even when working alone, even a small ML project involves separate tasks, each with a good amount of complexity. Tasks include: workspace configuration and data access, data preparation, model definition and configuration, and deployment. While the outputs of one or more tasks form the inputs to another, the exact implementation details of any one task are, at best, irrelevant distractions in the next. At worst, the computational state of one task can cause a bug in another.

There are many ways to leverage AzureML. Currently DeepSeismic has integrated with AzureML to train a pipeline, which will include creating an experiment titled "DEV-train-pipeline" which will contain all training runs, associated logs, and the ability to navigate seemlessly through this information. AzureML will take data from a blob storage account and the associated models will be saved to this account upon completion of the run.

Please refer to microsoft docs for additional information on AzureML pipelines and related capabilities ['What are Azure Machine Learning pipelines?'](https://docs.microsoft.com/en-us/azure/machine-learning/concept-ml-pipelines) 

## Files needed for this AzureML run

You will need the following files to complete an run in AzureML

- [.azureml/config.json](../../../.azureml.example/config.json) This is used to import your subscription, resource group, and AzureML workspace
- [.env](../../../.env.example) This is used to import your environment variables including blob storage information and AzureML compute cluster specs
- [kickoff_train_pipeline.py](dev/kickoff_train_pipeline.py) This script shows how to run an AzureML train pipeline 
- [cancel_run.py](dev/cancel_run.py) This script is used to cancel an AzureML train pipeline run
- [base_pipeline.py](base_pipeline.py) This script is used as a base class and train_pipeline.py inherits from it. This is intended to be a helpful abstraction that an an future addition of an inference pipeline can leverage
- [train_pipeline.py](train_pipeline.py) This script inherts from base_pipeline.py and is used to construct the pipeline and its steps. The script kickoff_train_pipeline.py will call the function defined here and the pipeline_config
- [pipeline_config.json](pipeline_config.json) This pipeline configuration specifies the steps of the pipeline, location of data, and any specific arguments. This is consumed once the kickoff_train_script.py is run
- [train.py](../../../experiments/interpretation/dutchf3_patch/train.py) This is the training script that is used to train the model
- [unet.yaml](../../../experiments/interpretation/dutchf3_patch/configs/unet.yaml) This config specifices the model configuration to be used in train.py and is referenced in the pipeline_config.json
- [azureml_requirements.txt](../../../experiments/interpretation/dutchf3_patch/azureml_requirements.txt) This file holds all dependencies for train.py so they can be installed on the compute in Azure ML
- [logging.config](../../../experiments/interpretation/dutchf3_patch/logging.config) This logging config is used to set up logging
- local environment with cv_lib and interpretation set up using guidance [here](../../../README.md)

## Running a Pipeline in AzureML

Go into the [Azure Portal](https://portal.azure.com) and create a blob storage. Once you have created a [blob storage](https://azure.microsoft.com/en-us/services/storage/blobs/) you may use [Azure Storage Explorer](https://docs.microsoft.com/en-us/azure/vs-azure-tools-storage-manage-with-storage-explorer?tabs=windows) to manage your blob instance. You can either manually upload data through Azure Storage Explorer, or you can use [AzCopy](https://docs.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-v10) to migrate the data to your blob storage. Once you blob storage set up and the data migrated, you may being to fill in the environemnt variables below. There is a an example [.env file](../../../.env.example) that you may leverage. More information on how to activate these environment variables are below.

With your run you will need to specifiy the below compute. Once you populate these variables, AzureML will spin up a run based creation compute, this means that the compute will be created by AzureML at run time specifically for your run. The compute is deleted automatically once the run completes. With AzureML you also have the option of creating and attaching your own compute. For more information on run-based compute creation and persistent compute please refer to the [Azure Machine Learning Compute](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-set-up-training-targets) section in Microsoft docs.

`AML_COMPUTE_CLUSTER_SKU` refers to VM family of the nodes created by Azure Machine Learning Compute. If not specified, defaults to Standard_NC6. For compute options see [HardwareProfile object values](https://docs.microsoft.com/en-us/azure/templates/Microsoft.Compute/2019-07-01/virtualMachines?toc=%2Fen-us%2Fazure%2Fazure-resource-manager%2Ftoc.json&bc=%2Fen-us%2Fazure%2Fbread%2Ftoc.json#hardwareprofile-object
)
`AML_COMPUTE_CLUSTER_MAX_NODES` refers to the max number of nodes to autoscale up to when you run a job on Azure Machine Learning Compute. This is not the max number of nodes for multi-node training, instead this is for the amount of nodes available to process single-node jobs.

If you would like additional information with regards AzureML compute provisioning class please refer to Microsoft docs on [AzureML compute provisioning class](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.compute.amlcompute.amlcomputeprovisioningconfiguration?view=azure-ml-py)

Set the following environment variables:
```
BLOB_ACCOUNT_NAME
BLOB_CONTAINER_NAME
BLOB_ACCOUNT_KEY
BLOB_SUB_ID
AML_COMPUTE_CLUSTER_NAME
AML_COMPUTE_CLUSTER_MIN_NODES
AML_COMPUTE_CLUSTER_MAX_NODES
AML_COMPUTE_CLUSTER_SKU
```

On Linux:
`export VARIABLE=value`
Our code can pick the environment variables from the .env file; alternatively you can `source .env` to activate these variables in your environment. An example .env file is found at the ROOT of this repo [here](../../../.env.example). You can rename this to .env. Feel free to use this as your .env file but be sure to add this to your .gitignore to ensure you do not commit any secrets. 

You will be able to download a config.json that will already have your subscription id, resource group, and workspace name directly in the [Azure Portal](https://portal.azure.com). You will want to navigate to your AzureML workspace and then you can click the `Download config.json` option towards the top left of the browser. Once you do this you can rename the .azureml.example folder to .azureml and replace the config.json with your downloaded config.json. If you would prefer to migrate the information manually refer to the guidance below.

Create a .azureml/config.json file in the project's root directory that looks like so:
```json
{
"subscription_id": "<subscription id>",
"resource_group": "<resource group>",
"workspace_name": "<workspace name>"
}

```
At the ROOT of this repo you will find an example [here](../../../.azureml.example/config.json). This is an example please rename the file to .azureml/config.json, input your account information and add this to your .gitignore. 


## Training Pipeline
Here's an example of a possible pipeline configuration file:
```json
{
    "step1":
    {
        "type": "MpiStep",
        "name": "train step",
        "script": "train.py",
        "input_datareference_path": "normalized_data/",
        "input_datareference_name": "normalized_data_conditioned",
        "input_dataset_name": "normalizeddataconditioned",
        "source_directory": "train/",
        "arguments": ["--splits", "splits",
        "--train_data_paths", "normalized_data/file.npy",
        "--label_paths", "label.npy"],
        "requirements": "train/requirements.txt",
        "node_count": 1,
        "processes_per_node": 1,
        "base_image": "pytorch/pytorch"
    }
}
```
  
If you want to create a train pipeline:
1) All of your steps are isolated
    - Your scripts will need to conform to the interface you define in the pipeline configuration file
        - I.e., if step1 is expected to output X and step 2 is expecting X as an input, your scripts need to reflect that
    - If one of your steps has pip package dependencies, make sure it's specified in a requirements.txt file
    - If your script has local dependencies (i.e., is importing from another script) make sure that all dependencies fall underneath the source_directory
2) You have configured your pipeline configuration file to specify the steps needed (see the section below "Configuring a Pipeline" for guidance)

Note: the following arguments are automatically added to any script steps by AzureML:
```--input_data``` and ```--output``` (if output is specified in the pipeline_config.json)
Make sure to add these arguments in your scripts like so:
```python
parser.add_argument('--input_data', type=str, help='path to preprocessed data')
parser.add_argument('--output', type=str, help='output from training')
```
```input_data``` is the absolute path to the input_datareference_path on the blob you specified.
  
# Configuring a Pipeline
  
## Train Pipeline
Define parameters for the run in a pipeline configuration file. See an example in this repo [here](pipeline_config.json). For additional guidance on [pipeline steps](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-create-your-first-pipeline#steps) please refer to Microsoft docs.
```json
{
    "step1":
    {
        "type": "<type of step. Supported types include PythonScriptStep and MpiStep>",
        "name": "<name in AzureML for this step>",
        "script": "<path to script for this step>",
        "output": "<name of the output in AzureML for this step - optional>",
        "input_datareference_path": "<path on the data reference for the input data - optional>",
        "input_datareference_name": "<name of the data reference in AzureML where the input data lives - optional>",
        "input_dataset_name": "<name of the datastore in AzureML - optional>",
        "source_directory": "<source directory containing the files for this step>",
        "arguments": "<arguments to pass to the script - optional>",
        "requirements": "<path to the requirements.txt file for the step - optional>",
        "node_count": "<number of nodes to run the script on - optional>",
        "processes_per_node": "<number of processes to run on each node - optional>",
        "base_image": "<name of an image registered on dockerhub that you want to use as your base image"
    },
  
    "step2":
    {
        .
        .
        .
    }
}
```
  
## Kicking off a Pipeline
In order to kick off a pipeline, you will need to use the AzureCLI to login to the subscription where your workspace resides. Once you successfully log in, there will be a print out of all of the subscriptions you have access to. You can either get your subscription id this way or you could go directly to the azure portal, navigate to your subscriptions, and then locate the right subscription id to pass into az account set -s:
```bash
az login
az account set -s <subscription id>
```
Kick off the training pipeline defined in your config via your python environment of choice. First activate your local environment that has cv_lib and interpretation set up using guidance [here](../../../README.md). You will run the kick off for the training pipeline from the ROOT directory. The code will look like this:
```python
from src.azml.train_pipeline.train_pipeline import TrainPipeline

orchestrator = TrainPipeline("<path to your pipeline configuration file>")
orchestrator.construct_pipeline()
run = orchestrator.run_pipeline(experiment_name="DEV-train-pipeline")
```
See an example in [dev/kickoff_train_pipeline.py](dev/kickoff_train_pipeline.py)

If you run into a subscription access error you might a work around in [Troubleshooting](##troubleshooting) section.
  
## Cancelling a Pipeline Run
If you kicked off a pipeline and want to cancel it, run the [cancel_run.py](dev/cancel_run.py) script with the corresponding run_id and step_id. The corresponding run_id and step_id will be printed once you have run the script. You can also find this information when viewing your run in the portal https://portal.azure.com/. If you would prefer to cancel your run in the portal you may also do this as well.

## Troubleshooting

If you run into issues gaining access to the Azure ML subscription, you may be able to connect by using a workaround:
Go to [base_pipeline.py](../base_pipeline.py) and add the following import:
```python
from azureml.core.authentication import AzureCliAuthentication
```
Then find the code where we connect to the workspace which looks like this:
```python
self.ws = Workspace.from_config(path=ws_config)
```
and replace it with  this:
```python
cli_auth = AzureCliAuthentication()
self.ws = Workspace(subscription_id=<subscription id>, resource_group=<resource group>, workspace_name=<workspace name>, auth=cli_auth)
```
to get this to run, you will also need to `pip install azure-cli-core`
Then you can go back and follow the instructions above, including az login and setting the subscription, and kick off the pipeline.