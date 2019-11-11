# DeepSeismic

## Imaging

This tutorial shows how to run devito tutorial notebooks in Azure. For best experience use a linux Azure DSVM and Jupyter notebook to run the notebooks.
 
### Setting up Environment

The conda .yml file should be used to create the conda environment, before starting Jupyter:

```
channels:
  - anaconda
dependencies:
  - python=3.6  
  - numpy
  - cython
  - notebook 
  - nb_conda
  - scikit-learn
  - pip
  - pip:
    - python-dotenv
    - papermill[azure]
    - azureml-sdk[notebooks,automl,explain]==1.0.69
```

You may also need az cli to create and ACR in notebook 000_Setup_GeophysicsTutorial_FWI_Azure_devito. Or you can create the ACR via portal and comment the cli commands for ACR creation.