# DeepSeismic

## Imaging

This tutorial shows how to run devito tutorial notebooks in Azure. For best experience use a linux Azure DSVM and Jupyter notebook to run the notebooks.
 
### Setting up Environment

This conda .yml file should be used to create the conda environment, before starting Jupyter:

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
    - azureml-sdk[notebooks,automl,explain] 
```

You may also need az cli to create an ACR in notebook 000_Setup_GeophysicsTutorial_FWI_Azure_devito, and then push and pull images. One can also create the ACR via Azure [portal](https://azure.microsoft.com/).