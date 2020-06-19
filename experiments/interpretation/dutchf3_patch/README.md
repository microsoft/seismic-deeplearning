## Dutch F3 Patch Experiments
In this folder are training and testing scripts that work on the F3 Netherlands dataset. 
You can run five different models on this dataset:

* [HRNet](configs/hrnet.yaml)
* [SEResNet](configs/seresnet_unet.yaml)
* [UNet](configs/unet.yaml)
* [PatchDeconvNet](configs/patch_deconvnet.yaml)
* [PatchDeconvNet-Skip](configs/patch_deconvnet_skip.yaml)

All these models take 2D patches of the dataset as input and provide predictions for those patches. The patches need to be stitched together to form a whole inline or crossline.

To understand the configuration files and the default parameters refer to this [section in the top level README](../../../README.md#configuration-files)

### Setup

Please set up a conda environment following the instructions in the top-level [README.md](../../../README.md#setting-up-environment) file. Also follow instructions for [downloading and preparing](../../../README.md#f3-Netherlands) the data.

### Running experiments

Now you're all set to run training and testing experiments on the Dutch F3 dataset. Please start from the `train.sh` and `test.sh` scripts, which invoke the corresponding python scripts. If you have a multi-GPU machine, you can also train the model in a distributed fashion by running `train_distributed.sh`. Take a look at the project configurations in (e.g in `default.py`) for experiment options and modify if necessary. 

Please note that we use [NVIDIA's NCCL](https://docs.nvidia.com/deeplearning/nccl/install-guide/index.html) library to enable distributed training. Please follow the installation instructions [here](https://docs.nvidia.com/deeplearning/nccl/install-guide/index.html#down) to install NCCL on your system.   

### Monitoring progress with TensorBoard
- from the this directory, run `tensorboard --logdir='output'` (all runtime logging information is written to the `output` folder  
- open a web-browser and go to either `<vm_public_ip>:6006` if running remotely or `localhost:6006` if running locally  
> **NOTE**:If running remotely remember that the port must be open and accessible 
 
More information on Tensorboard can be found [here](https://www.tensorflow.org/get_started/summaries_and_tensorboard#launching_tensorboard).
