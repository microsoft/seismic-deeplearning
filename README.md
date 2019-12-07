# DeepSeismic
![DeepSeismic](./assets/DeepSeismicLogo.jpg )

This repository shows you how to perform seismic imaging and interpretation on Azure. It empowers geophysicists and data scientists to run seismic experiments using state-of-art DSL-based PDE solvers and segmentation algorithms on Azure.  

The repository provides sample notebooks, data loaders for seismic data, utilities, and out-of-the box ML pipelines, organized as follows:
- **sample notebooks**: these can be found in the `examples` folder - they are standard Jupyter notebooks which highlight how to use the codebase by walking the user through a set of pre-made examples
- **experiments**: the goal is to provide runnable Python scripts which train and test (score) our machine learning models in `experiments` folder. The models themselves are swappable, meaning a single train script can be used to run a different model on the same dataset by simply swapping out the configuration file which defines the model. Experiments are organized by model types and datasets - for example, "2D segmentation on Dutch F3 dataset", "2D segmentation on Penobscot dataset" and "3D segmentation on Penobscot dataset" are all different experiments. As another example, if one is swapping 2D segmentation models on Dutch F3 dataset, one would just point the train and test scripts to a different configuration file within the same experiment.
- **pip installable utilities**: we provide `cv_lib` and `deepseismic_interpretation` utilities (more info below) which are used by both sample notebooks and experiments mentioned above

DeepSeismic currently focuses on Seismic Interpretation (3D segmentation aka facies classification) with experimental code provided around Seismic Imaging.

### Quick Start

There are two ways to get started with the DeepSeismic codebase, which currently focuses on Interpretation:
- if you'd like to get an idea of how our interpretation (segmentation) models are used, simply review the [HRNet demo notebook](https://github.com/microsoft/DeepSeismic/blob/staging/examples/interpretation/notebooks/HRNet_demo_notebook.ipynb)
- to actually run the code, you'll need to set up a compute environment (which includes setting up a GPU-enabled Linux VM and downloading the appropriate Anaconda Python packages) and download the datasets which you'd like to work with - detailed steps for doing this are provided in the next `Interpretation` section below.

If you run into any problems, chances are your problem has already been solved in the [Troubleshooting](#troubleshooting) section.

## Interpretation
For seismic interpretation, the repository consists of extensible machine learning pipelines, that shows how you can leverage state-of-the-art segmentation algorithms (UNet, SEResNET, HRNet) for seismic interpretation, and also benchmarking results from running these algorithms using various seismic datasets (Dutch F3, and Penobscot).

To run examples available on the repo, please follow instructions below to:
1) [Set up the environment](#setting-up-environment)
2) [Download the data sets](#dataset-download-and-preparation)
3) [Run example notebooks and scripts](#run-examples)

### Setting up Environment

Follow the instruction bellow to read about compute requirements and install required libraries.

<details>
  <summary><b>Compute environment</b></summary>

We recommend using a virtual machine to run the example notebooks and scripts. Specifically, you will need a GPU powered Linux machine, as this repository is developed and tested on __Linux only__. The easiest way to get started is to use the [Azure Data Science Virtual Machine (DSVM) for Linux (Ubuntu)](https://docs.microsoft.com/en-us/azure/machine-learning/data-science-virtual-machine/dsvm-ubuntu-intro). This VM will come installed with all the system requirements that are needed to create the conda environment described below and then run the notebooks in this repository. 

For this repo, we recommend selecting a multi-GPU Ubuntu VM of type [Standard_NC12](https://docs.microsoft.com/en-us/azure/virtual-machines/windows/sizes-gpu#ncv3-series). The machine is powered by NVIDIA Tesla K80 (or V100 GPU for NCv2 series) which can be found in most Azure regions.

> NOTE: For users new to Azure, your subscription may not come with a quota for GPUs. You may need to go into the Azure portal to increase your quota for GPU VMs. Learn more about how to do this here: https://docs.microsoft.com/en-us/azure/azure-subscription-service-limits.

</details>

<details>
  <summary><b>Package Installation</b></summary>

To install packages contained in this repository, navigate to the directory where you pulled the DeepSeismic repo to run:
```bash
conda env create -f environment/anaconda/local/environment.yml
```
This will create the appropriate conda environment to run experiments.

Next you will need to install the common package for interpretation:
```bash
conda activate seismic-interpretation
pip install -e interpretation
```

Then you will also need to install `cv_lib` which contains computer vision related utilities:
```bash
pip install -e cv_lib
```

Both repos are installed in developer mode with the `-e` flag. This means that to update simply go to the folder and pull the appropriate commit or branch. 

During development, in case you need to update the environment due to a conda env file change, you can run
```
conda env update --file environment/anaconda/local/environment.yml
```
from the root of DeepSeismic repo.

</details>

### Dataset download and preparation

This repository provides examples on how to run seismic interpretation on two publicly available annotated seismic datasets: [Penobscot](https://zenodo.org/record/1341774) and [F3 Netherlands](https://github.com/olivesgatech/facies_classification_benchmark).

#### Penobscot
To download the Penobscot dataset run the [download_penobscot.sh](scripts/download_penobscot.sh) script, e.g.

```
data_dir="$HOME/data/penobscot"
mkdir -p "$data_dir"
./scripts/download_penobscot.sh "$data_dir"
```

Note that the specified download location should be configured with appropriate `write` permissions. On some Linux virtual machines, you may want to place the data into `/mnt` or `/data` folder so you have to make sure you have write access.

To make things easier, we suggested you use your home directory where you might run out of space. If this happens on an [Azure Data Science Virtual Machine](https://azure.microsoft.com/en-us/services/virtual-machines/data-science-virtual-machines/) you can resize the disk quite easily from [Azure Portal](https://portal.azure.com) - please see the [Troubleshooting](#troubleshooting) section at the end of this README regarding [how to do this](#how-to-resize-data-science-virtual-machine-disk).

To prepare the data for the experiments (e.g. split into train/val/test), please run the following script (modifying arguments as desired):

```
python scripts/prepare_penobscot.py split_inline --data-dir=/data/penobscot --val-ratio=.1 --test-ratio=.2
```

#### F3 Netherlands
To download the F3 Netherlands dataset for 2D experiments, please follow the data download instructions at
[this github repository](https://github.com/yalaudah/facies_classification_benchmark) (section Dataset).

Once you've downloaded the data set, make sure to create an empty `splits` directory, under the downloaded `data` directory; you can re-use the same data directory as the one for Penobscot dataset created earlier. This is where your training/test/validation splits will be saved.

```
cd data
mkdir splits
```

At this point, your `data` directory tree should look like this:

```
data
├── splits
├── test_once
│   ├── test1_labels.npy
│   ├── test1_seismic.npy
│   ├── test2_labels.npy
│   └── test2_seismic.npy
└── train
    ├── train_labels.npy
    └── train_seismic.npy
```

To prepare the data for the experiments (e.g. split into train/val/test), please run the following script:

```
# For section-based experiments
python scripts/prepare_dutchf3.py split_train_val section --data-dir=/mnt/dutchf3


# For patch-based experiments
python scripts/prepare_dutchf3.py split_train_val patch --data-dir=/mnt/dutchf3 --stride=50 --patch=100

```

Refer to the script itself for more argument options.

### Run Examples

#### Notebooks
We provide example notebooks under `examples/interpretation/notebooks/` to demonstrate how to train seismic interpretation models and evaluate them on Penobscot and F3 datasets. 

Make sure to run the notebooks in the conda environment we previously set up (`seismic-interpretation`). To register the conda environment in Jupyter, please run:

```
python -m ipykernel install --user --name seismic-interpretation
```

#### Experiments

We also provide scripts for a number of experiments we conducted using different segmentation approaches. These experiments are available under `experiments/interpretation`, and can be used as examples. Within each experiment start from the `train.sh` and `test.sh` scripts under the `local/` (single GPU) and `distributed/` (multiple GPUs) directories, which invoke the corresponding python scripts, `train.py` and `test.py`. Take a look at the experiment configurations (see Experiment Configuration Files section below) for experiment options and modify if necessary.

Please refer to individual experiment README files for more information. 

#### Configuration Files
We use [YACS](https://github.com/rbgirshick/yacs) configuration library to manage configuration options for the experiments. There are three ways to pass arguments to the experiment scripts (e.g. train.py or test.py):

- __default.py__ - A project config file `default.py` is a one-stop reference point for all configurable options, and provides sensible defaults for all arguments. If no arguments are passed to `train.py` or `test.py` script (e.g. `python train.py`), the arguments are by default loaded from `default.py`. Please take a look at `default.py` to familiarize yourself with the experiment arguments the script you run uses.

- __yml config files__ - YAML configuration files under `configs/` are typically created one for each experiment. These are meant to be used for repeatable experiment runs and reproducible settings. Each configuration file only overrides the options that are changing in that experiment (e.g. options loaded from `defaults.py` during an experiment run will be overridden by arguments loaded from the yaml file). As an example, to use yml configuration file with the training script, run:

    ```
    python train.py --cfg "configs/hrnet.yaml"
    ```

- __command line__ - Finally, options can be passed in through `options` argument, and those will override arguments loaded from the configuration file. We created CLIs for all our scripts (using Python Fire library), so you can pass these options via command-line arguments, like so:

    ```
    python train.py DATASET.ROOT "/mnt/dutchf3" TRAIN.END_EPOCH 10
    ```


### Pretrained Models

#### HRNet

To achieve the same results as the benchmarks above you will need to download the HRNet model [pretrained](https://github.com/HRNet/HRNet-Image-Classification) on ImageNet. We are specifically using the [HRNet-W48-C](https://1drv.ms/u/s!Aus8VCZ_C_33dKvqI6pBZlifgJk) pre-trained model - download this model to your local drive and make sure you add the path to the experiment (or notebook) configuration file under `TEST.MODEL_PATH` setting. Other  HRNet variants are also available [here](https://github.com/HRNet/HRNet-Image-Classification) - you can navigate to those from the [main HRNet landing page](https://github.com/HRNet/HRNet-Object-Detection) for object detection.

To facilitate easier download on a Linux machine of your choice (or [Azure Data Science Virtual Machine](https://azure.microsoft.com/en-us/services/virtual-machines/data-science-virtual-machines/) which we recommend), we created an automated download scipt for you, just run
```bash
./scripts/download_hrnet.sh 'your_folder_to_store_the_model' 'model_file'
```

### Viewers (optional)

For seismic interpretation (segmentation), if you want to visualize cross-sections of a 3D volume (both the input velocity model and the segmented output) you can use
[segyviewer](https://github.com/equinor/segyviewer). To install and use segyviewer, please follow the instructions below.

#### segyviewer

To install [segyviewer](https://github.com/equinor/segyviewer) run:
```bash
conda env create -n segyviewer python=2.7
conda activate segyviewer
conda install -c anaconda pyqt=4.11.4
pip install segyviewer
```

To visualize cross-sections of a 3D volume, you can run
[segyviewer](https://github.com/equinor/segyviewer) like so:
```bash
segyviewer /mnt/dutchf3/data.segy
```

### Benchmarks

#### Dense Labels

This section contains benchmarks of different algorithms for seismic interpretation on 3D seismic datasets with densely-annotated data.

Below are the results from the models contained in this repo. To run them check the instructions in <benchmarks> folder. Alternatively take a look in <examples> for how to run them on your own dataset

#### Netherlands F3

|    Source        |    Experiment                     |    PA       |    FW IoU    |    MCA     |
|------------------|-----------------------------------|-------------|--------------|------------|
|    Alaudah et al.|    Section-based                  |    0.905    |    0.817     |    .832    |
|                  |    Patch-based                    |    0.852    |    0.743     |    .689    |
|    DeepSeismic   |    Patch-based+fixed              |    .869     |    .761      |    .775    |
|                  |    SEResNet UNet+section depth    |    .917     |    .849      |    .834    |
|                  |    HRNet(patch)+patch_depth       |    .908     |    .843      |    .837    |
|                  |    HRNet(patch)+section_depth     |    .928     |    .871      |    .871    |

#### Penobscot

Trained and tested on full dataset. Inlines with artefacts were left in for training, validation and testing.
The dataset was split 70% training, 10% validation and 20% test. The results below are from the test set

|    Source        |    Experiment                       |    PA       |    IoU       |    MCA     |
|------------------|-------------------------------------|-------------|--------------|------------|
|    DeepSeismic   |    SEResNet UNet + section depth    |    1.0      |    .98        |    .99    |
|                  |    HRNet(patch) + section depth     |    1.0      |    .97        |    .98    |

![Best Penobscot SEResNet](assets/penobscot_seresnet_best.png "Best performing inlines, Mask and Predictions from SEResNet")
![Worst Penobscot SEResNet](assets/penobscot_seresnet_worst.png "Worst performing inlines  Mask and Predictions from SEResNet")


#### Scripts
- [parallel_training.sh](scripts/parallel_training.sh): Script to launch multiple jobs in parallel. Used mainly for local hyperparameter tuning. Look at the script for further instructions

- [kill_windows.sh](scripts/kill_windows.sh): Script to kill multiple tmux windows. Used to kill jobs that parallel_training.sh might have started.


## Contributing

This project welcomes contributions and suggestions. Most contributions require you to agree to a Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

### Submitting a Pull Request

We try to keep the repo in a clean state, which means that we only enable read access to the repo - read access still enables one to submit a PR or an issue. To do so, fork the repo, and submit a PR from a branch in your forked repo into our staging branch.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Build Status
| Build | Branch | Status |
| --- | --- | --- |
| **Legal Compliance** | staging | [![Build Status](https://dev.azure.com/best-practices/deepseismic/_apis/build/status/microsoft.ComponentGovernance?branchName=staging)](https://dev.azure.com/best-practices/deepseismic/_build/latest?definitionId=110&branchName=staging) |
| **Legal Compliance** | master | [![Build Status](https://dev.azure.com/best-practices/deepseismic/_apis/build/status/microsoft.ComponentGovernance?branchName=master)](https://dev.azure.com/best-practices/deepseismic/_build/latest?definitionId=110&branchName=master) |
| **Tests** | staging | [![Build Status](https://dev.azure.com/best-practices/deepseismic/_apis/build/status/microsoft.Tests?branchName=staging)](https://dev.azure.com/best-practices/deepseismic/_build/latest?definitionId=111&branchName=staging) |
| **Tests** | master | [![Build Status](https://dev.azure.com/best-practices/deepseismic/_apis/build/status/microsoft.Tests?branchName=master)](https://dev.azure.com/best-practices/deepseismic/_build/latest?definitionId=111&branchName=master) |
| **Notebook Tests** | staging | [![Build Status](https://dev.azure.com/best-practices/deepseismic/_apis/build/status/microsoft.Notebooks?branchName=staging)](https://dev.azure.com/best-practices/deepseismic/_build/latest?definitionId=120&branchName=staging) |
| **Notebook Tests** | master | [![Build Status](https://dev.azure.com/best-practices/deepseismic/_apis/build/status/microsoft.Notebooks?branchName=master)](https://dev.azure.com/best-practices/deepseismic/_build/latest?definitionId=120&branchName=master) |

# Related projects

[Microsoft AI Labs Github](https://aka.ms/ai-labs) Find other Best Practice projects, and Azure AI design patterns in our central repository. 

# Troubleshooting

For Data Science Virtual Machine conda package installation issues, make sure you locate the anaconda location on the DSVM, for example by running:
```bash
which python
```
A typical output will be:
```bash
someusername@somevm:/projects/DeepSeismic$ which python
/anaconda/envs/py35/bin/python
```
which will indicate that anaconda folder is __/anaconda__. We'll refer to this location in instructions below, but you should update the commands according to your local anaconda folder.

<details>
  <summary><b>Data Science Virtual Machine conda package installation errors</b></summary>

  It could happen that you don't have sufficient permissions to run conda commands / install packages in an Anaconda packages directory. To remedy the situation, please run the following commands
  ```bash
  rm -rf /anaconda/pkgs/*
  sudo chown -R $(whoami) /anaconda
  ```

  After these commands complete, try installing the packages again.

</details>

<details>
  <summary><b>Data Science Virtual Machine conda package installation warnings</b></summary>

  It could happen that while creating the conda environment defined by environment/anaconda/local/environment.yml on an Ubuntu DSVM, one can get multiple warnings like this:  
  WARNING conda.gateways.disk.delete:unlink_or_rename_to_trash(140): Could not remove or rename /anaconda/pkgs/ipywidgets-7.5.1-py_0/site-packages/ipywidgets-7.5.1.dist-info/LICENSE.  Please remove this file manually (you may need to reboot to free file handles)  
    
  If that happens, similar to instructions above, stop the conda environment creation (type ```Ctrl+C```) and then change recursively the ownership /anaconda directory from root to current user, by running this command: 

  ```bash
  sudo chown -R $USER /anaconda
  ```

  After these command completes, try creating the conda environment in __environment/anaconda/local/environment.yml__ again.

</details>

<details>
  <summary><b>Model training or scoring is not using GPU</b></summary>

  To see if GPU is being using while your model is being trained or used for inference, run
  ```bash
  nvidia-smi
  ```
  and confirm that you see you Python process using the GPU.

  If not, you may want to try reverting to an older version of CUDA for use with pyTorch. After the environment has been setup, run the following command (by default we use CUDA 10) after running `conda activate seismic-interpretation` to activate the conda environment:
  ```bash
  conda install pytorch torchvision cudatoolkit=9.2 -c pytorch
  ```

  To test whether this setup worked, right after you can open `ipython` and execute the following code
  ```python
  import torch                                                                                  
  torch.cuda.is_available() 
  ```

  The output should say "True".

  If the output is still "False", you may want to try setting your environment variable to specify the device manually - to test this, start a new `ipython` session and type:
  ```python
  import os
  os.environ['CUDA_VISIBLE_DEVICES']='0'
  import torch                                                                                  
  torch.cuda.is_available() 
  ```

  Output should say "True" this time. If it does, you can make the change permanent by adding
  ```bash
  export CUDA_VISIBLE_DEVICES=0
  ```
  to your `$HOME/.bashrc` file.

</details>

<details>
  <summary><b>GPU out of memory errors</b></summary>

  You should be able to see how much GPU memory your process is using by running
  ```bash
  nvidia-smi
  ```
  and seeing if this amount is close to the physical memory limit specified by the GPU manufacturer.

  If we're getting close to the memory limit, you may want to lower the batch size in the model configuration file. Specifically, `TRAIN.BATCH_SIZE_PER_GPU` and `VALIDATION.BATCH_SIZE_PER_GPU` settings.

</details>

<details>
  <summary><b>How to resize Data Science Virtual Machine disk</b></summary>

  1. Go to the [Azure Portal](https://portal.azure.com) and find your virtual machine by typing its name in the search bar at the very top of the page.

  2. In the Overview panel on the left hand side, click Stop button to stop the virtual machine.

  3. Next, select Disks in the same panel on the left hand side.

  4. Click the Name of the OS Disk - you'll be navigated to the Disk view. From this view, select Configuration on the left hand side and then increase Size in GB and hit the Save button.

  5. Navigate back to the Virtual Machine view in Step 2 and click the Start button to start the virtual machine.

</details>





