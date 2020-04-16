## F3 Netherlands Section Experiments
In this folder are training and testing scripts that work on the F3 Netherlands dataset. 
You can run one model on this dataset:
* [SectionDeconvNet-Skip](local/configs/section_deconvnet_skip.yaml)

This model takes 2D sections as input from the dataset whether these be inlines or crosslines and provides predictions for whole section.

To understand the configuration files and the dafault parameters refer to this [section in the top level README](../../../README.md#configuration-files)

### Setup

Please set up a conda environment following the instructions in the top-level [README.md](../../../README.md#setting-up-environment) file.
Also follow instructions for [downloading and preparing](../../../README.md#f3-Netherlands) the data.

### Running experiments

Now you're all set to run training and testing experiments on the F3 Netherlands dataset. Please start from the `train.sh` and `test.sh` scripts under the `local/` directory, which invoke the corresponding python scripts. Take a look at the project configurations in (e.g in `default.py`) for experiment options and modify if necessary. 

### Monitoring progress with TensorBoard
- from the this directory, run `tensorboard --logdir='output'` (all runtime logging information is
written to the `output` folder  
- open a web-browser and go to  either vmpublicip:6006 if running remotely or localhost:6006 if running locally  
> **NOTE**:If running remotely remember that the port must be open and accessible 
 
More information on Tensorboard can be found [here](https://www.tensorflow.org/get_started/summaries_and_tensorboard#launching_tensorboard).
