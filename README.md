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

Both repos are installed in developer mode with the -e flag. This means that to update simply go to the folder and pull the appropriate commit or branch

## Benchmarks

### Dense Labels

This section contains benchmarks of different algorithms for seismic interpretation on 3D seismic datasets with densely-annotated data.

Below are the results from the models contained in this repo. To run them check the instructions in <benchmarks> folder. Alternatively take a look in <examples> for how to run them on your own dataset

|    Authorship    |    Experiment                     |    PA       |    FW IoU    |    MCA     |
|------------------|-----------------------------------|-------------|--------------|------------|
|    Alaudah       |    Section-based                  |    0.905    |    0.817     |    .832    |
|                  |    Patch-based                    |    0.852    |    0.743     |    .689    |
|    Ours          |    Patch-based+fixed              |    .869     |    .761      |    .775    |
|                  |    SEResNet UNet+section depth    |    .917     |    .849      |    .834    |
|                  |    HRNet(patch)+patch_depth       |    .908     |    .843      |    .837    |
|                  |    HRNet(patch)+section_depth     |    .928     |    .871      |    .871    |


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
The scripts expect the data to be contained in /mnt/dutchf3

## Contributing

This project welcomes contributions and suggestions. Most contributions require you to agree to a Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
