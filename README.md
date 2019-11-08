# DeepSeismic

## Interpretation

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

## Benchmarks

### Dense Labels

This section contains benchmarks of different algorithms for seismic interpretation on 3D seismic datasets with densely-annotated data.

Below are the results from the models contained in this repo. To run them check the instructions in <benchmarks> folder. Alternatively take a look in <examples> for how to run them on your own dataset

### Netherlands F3

|    Authorship    |    Experiment                     |    PA       |    FW IoU    |    MCA     |
|------------------|-----------------------------------|-------------|--------------|------------|
|    Alaudah       |    Section-based                  |    0.905    |    0.817     |    .832    |
|                  |    Patch-based                    |    0.852    |    0.743     |    .689    |
|    Ours          |    Patch-based+fixed              |    .869     |    .761      |    .775    |
|                  |    SEResNet UNet+section depth    |    .917     |    .849      |    .834    |
|                  |    HRNet(patch)+patch_depth       |    .908     |    .843      |    .837    |
|                  |    HRNet(patch)+section_depth     |    .928     |    .871      |    .871    |

### Penobscot

Trained and tested on full dataset. Inlines with artefacts were left in for training, validation and testing.
The dataset was split 70% training, 10% validation and 20% test. The results below are from the test set

|    Authorship    |    Experiment                       |    PA       |    IoU       |    MCA     |
|------------------|-------------------------------------|-------------|--------------|------------|
|    Ours          |    SEResNet UNet + section depth    |    1.0      |    .98        |    .99    |
|                  |    HRNet(patch) + section depth     |    1.0      |    .97        |    .98    |

![Best Penobscot SEResNet](images/penobscot_seresnet_best.png "Best performing inlines, Mask and Predictions from SEResNet")
![Worst Penobscot SEResNet](images/penobscot_seresnet_worst.png "Worst performing inlines  Mask and Predictions from SEResNet")

### Sparse Labels

This section contains benchmarks of different algorithms for seismic interpretation on 3D seismic datasets with sparsely-annotated data and is organized by the levels of sparsity.

| Model \ Dataset | Dutch F3 (Alaudah) | Penobscot |
| :---:           |    :---:           |     :---: |
Alaudah base slice | Pixel Acc,  IoU <br> train time (s), score time (s)| |
Alaudah base patch | Pixel Acc,  IoU <br> train time (s), score time (s)| |
HRNet slice | | |
DeepLab V3 slice | | |
| TODO: add all models | | |


#### Scribble-Level Labels

We present results of algorithms which are based on scribble-level annotations, where the annotator labels a large collection of consecutive pixels with no gaps, e.g. brushstroke label.

#### Pixel-Level Labels

We present results of algorithms which are based on pixel-level annotations, where the annotator labels individual pixels and gaps are allowed between pixels; the annotator can also label a small neighborhood of pixels, e.g. large dot of ~100 pixels.

### Data
#### Netherlands F3

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

#### Penobscot
To download the Penobscot dataset run the [download_penobscot.sh](scripts/download_penobscot.sh) script, e.g.

```
./download_penobscot.sh /data/penobscot
```

Note that the specified download location (e.g `/data/penobscot`) should be created beforehand, and configured appropriate `write` pemissions.

To prepare the data for the experiments (e.g. split into train/val/test), please run the following script (modifying arguments as desired):

```
python scripts/prepare_penobscot.py split_inline --data-dir=/mnt/penobscot --val-ratio=.1 --test-ratio=.2
```


### Scripts
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

