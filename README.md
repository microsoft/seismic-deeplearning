# DeepSeismic
![DeepSeismic](./assets/DeepSeismicLogo.jpg )

This repository shows you how to perform seismic imaging and interpretation on Azure. It empowers geophysicists and data scientists to run seismic experiments using state-of-art DSL-based PDE solvers and segmentation algorithms on Azure.  

The repository provides sample notebooks, data loaders for seismic data, utility codes, and out-of-the box ML pipelines.


## Interpretation
For seismic interpretation, the repository consists of extensible machine learning pipelines, that shows how you can leverage state-of-art segmentation algorithms (UNet, SEResNET, HRNet) for seismic interpretation, and also benchmarking results from running these algorithms using various seismic datasets (Dutch F3, and Penobscot).

### Setting up Environment
Navigate to the folder where you pulled the DeepSeismic repo to

Run
```bash
conda env create -f environment/anaconda/local/environment.yml
```
This will create the appropriate environment to run experiments

Then you will need to install the common packages for interpretation
```bash
conda activate seismic-interpretation
pip install -e interpretation
```

Then you will also need to install cv_lib
```bash
pip install -e cv_lib
```

Both repos are installed in developer mode with the -e flag. This means that to update simply go to the folder and pull the appropriate commit or branch. 

During development, in case you need to update the environment due to a conda env file change, you can run
```
conda env update --file environment/anaconda/local/environment.yml
```
from the root of DeepSeismic repo.

### Viewers

#### segyviewer

For seismic interpretation (segmentation), if you want to visualize cross-sections of a 3D volume (both the input velocity model and the segmented output) you can use
[segyviewer](https://github.com/equinor/segyviewer), for example like so:
```bash
segyviewer /mnt/dutchf3/data.segy
```

To install [segyviewer](https://github.com/equinor/segyviewer) run
```bash
conda env -n segyviewer python=2.7
conda activate segyviewer
conda install -c anaconda pyqt=4.11.4
pip install segyviewer
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



#### Data
##### Netherlands F3
To download the F3 Netherlands dataset for 2D experiments, please follow the data download instructions at
[this github repository](https://github.com/olivesgatech/facies_classification_benchmark).

To prepare the data for the experiments (e.g. split into train/val/test), please run the following script:

```
# For section-based experiments
python scripts/prepare_dutchf3.py split_train_val section --data-dir=/mnt/dutchf3


# For patch-based experiments
python scripts/prepare_dutchf3.py split_train_val patch --data-dir=/mnt/dutchf3 --stride=50 --patch=100

```

Refer to the script itself for more argument options.

##### Penobscot
To download the Penobscot dataset run the [download_penobscot.sh](scripts/download_penobscot.sh) script, e.g.

```
data_dir='/data/penobscot'
mkdir "$data_dir"
./scripts/download_penobscot.sh "$data_dir"
```

Note that the specified download location (e.g `/data/penobscot`) should be created beforehand, and configured appropriate `write` pemissions.

To prepare the data for the experiments (e.g. split into train/val/test), please run the following script (modifying arguments as desired):

```
python scripts/prepare_penobscot.py split_inline --data-dir=/data/penobscot --val-ratio=.1 --test-ratio=.2
```

### Pretrained Models
#### HRNet
To achieve the same results as the benchmarks above you will need to download the HRNet model pretrained on Imagenet. This can be found [here](https://1drv.ms/u/s!Aus8VCZ_C_33dKvqI6pBZlifgJk). Download this to your local drive and make sure you add the path to the Yacs configuration script.

#### Scripts
- [parallel_training.sh](scripts/parallel_training.sh): Script to launch multiple jobs in parallel. Used mainly for local hyperparameter tuning. Look at the script for further instructions

- [kill_windows.sh](scripts/kill_windows.sh): Script to kill multiple tmux windows. Used to kill jobs that parallel_training.sh might have started.


## Seismic Imaging
For seismic imaging, the repository shows how you can leverage open-source PDE solvers (e.g. Devito), and perform Full-Waveform Inversion (FWI) at scale on Azure, using Azure Machine Learning (Azure ML), and Azure Batch. The repository provides a collection of sample notebooks that shows 

* How you can create customized Docker containers with Devito and use this on Azure
* How you can create Azure ML estimators for performing FWI using Devito. 
This enable the Devito code to easily run on a single machine, as well as multiple machines using Azure ML managed computes.


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

## Related projects

[Microsoft AI Github](https://github.com/microsoft/ai) Find other Best Practice projects, and Azure AI Designed patterns in our central repository. 

