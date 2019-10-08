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


#### Penobscot
To download the Penobscot dataset run the [download_penobscot.sh](scripts/download_penobscot.sh) script



### Scripts
- [parallel_training.sh](scripts/parallel_training.sh): Script to launch multiple jobs in parallel. Used mainly for local hyperparameter tuning. Look at the script for further instructions

- [kill_windows.sh](scripts/kill_windows.sh): Script to kill multiple tmux windows. Used to kill jobs that parallel_training.sh might have started.


## Contributing

This project welcomes contributions and suggestions. Most contributions require you to agree to a Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
