# DeepSeismic

## Imaging

This tutorial shows how to run [devito](https://www.devitoproject.org/) tutorial [notebooks](https://github.com/opesci/devito/tree/master/examples/seismic/tutorials) in Azure Machine Learning ([Azure ML](https://docs.microsoft.com/en-us/azure/machine-learning/)) using [Azure Machine Learning Python SDK](https://docs.microsoft.com/en-us/azure/machine-learning/service/tutorial-1st-experiment-sdk-setup).   
  
For best experience use a Linux (Ubuntu) Azure [DSVM](https://docs.microsoft.com/en-us/azure/machine-learning/data-science-virtual-machine/dsvm-ubuntu-intro) and Jupyter Notebook with AzureML Python SDK and [Azure CLI](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli?view=azure-cli-latest) to run the notebooks (see __Setting up Environment__ section below).  

Devito is a domain-specific Language (DSL) and code generation framework for the design of highly optimized finite difference kernels via symbolic computation for use in inversion methods. Here we show how ```devito``` can be openly used in the cloud by leveraging AzureML experimentation framework as a transparent and scalable platform for generic computation workloads. We focus on Full waveform inversion (__FWI__) problems where non-linear data-fitting procedures are applied for computing  estimates of subsurface properties from seismic data.   

 
### Setting up Environment

The [conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/environments.html) that encapsulates all the dependencies needed to run the notebooks described above can be created using the fwi_dev_conda_environment.yml file. See [here](https://github.com/Azure/MachineLearningNotebooks/blob/master/NBSETUP.md) generic instructions on how to install and run AzureML Python SDK in Jupyter Notebooks.

To create the conda environment, run:
```
conda env create -f fwi_dev_conda_environment.yml

```

then, one can see the created environment within the list of available environments and activate it:
```
conda env list
conda activate fwi_dev_conda_environment
```

Finally, start Jupyter notebook from within the activated environment:
```
jupyter notebook
```

[Azure CLI](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli?view=azure-cli-latest) is also used to create an ACR in notebook 000_Setup_GeophysicsTutorial_FWI_Azure_devito, and then push and pull docker images. One can also create the ACR via Azure [portal](https://azure.microsoft.com/).

### Run devito in Azure
The devito fwi examples are run in AzuremL using 4 notebooks:
 - ```000_Setup_GeophysicsTutorial_FWI_Azure_devito.ipynb```: sets up Azure resources (like resource groups, AzureML [workspace](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-manage-workspace)). 
 - ```010_CreateExperimentationDockerImage_GeophysicsTutorial_FWI_Azure_devito.ipynb```: Creates a custom docker file and the associated image that contains ```devito``` [github repository](https://github.com/opesci/devito.git) (including devito fwi tutorial [notebooks](https://github.com/opesci/devito/tree/master/examples/seismic/tutorials)) and runs the official devito install [tests](https://github.com/opesci/devito/tree/master/tests). 
 - ```020_UseAzureMLEstimatorForExperimentation_GeophysicsTutorial_FWI_Azure_devito.ipynb```: shows how the devito fwi tutorial [notebooks](https://github.com/opesci/devito/tree/master/examples/seismic/tutorials) can be run in AzureML using Azure Machine Learning [generic](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.estimator?view=azure-ml-py) [estimators](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-train-ml-models) with custom docker images. FWI computation takes place on a managed AzureML [remote compute cluster](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-set-up-training-targets).  
 
   ```Devito``` fwi computation artifacts (images and notebooks with data processing output results) are tracked under the AzureML workspace, and can be later downloaded and visualized.   
  
   Two ways of running devito code are shown:  
    (1) using __custom code__ (slightly modified graphing functions that save images to files). The AzureML experimentation job is defined by the devito code packaged as a py file. The experimentation job (defined by [azureml.core.experiment.Experiment](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.experiment.experiment?view=azure-ml-py) class can be used to track metrics or other artifacts (images) that are available in Azure portal.   
    
    (2) using [__papermill__](https://github.com/nteract/papermill) invoked via its Python API to run unedited devito demo notebooks (including the  [dask](https://dask.org/) local cluster [example](https://github.com/opesci/devito/blob/master/examples/seismic/tutorials/04_dask.ipynb) on the remote compute target and the results as saved notebooks that are available in Azure portal.  
    
 - ```030_ScaleJobsUsingAzuremL_GeophysicsTutorial_FWI_Azure_devito.ipynb```: shows how the devito fwi tutorial notebooks can be run in parallel on the elastically allocated AzureML [remote compute cluster](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-set-up-training-targets) created before. By submitting multiple jobs via azureml.core.Experiment submit(azureml.train.estimator.Estimator) one can use the [portal](https://portal.azure.com) to visualize the elastic allocation of AzureML [remote compute cluster](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-set-up-training-targets) nodes. 
 

