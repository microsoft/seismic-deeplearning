# Seismic Interpretation on Penobscot dataset
In this folder are training and testing scripts that work on the Penobscot dataset. 
You can run two different models on this dataset:
* [HRNet](local/configs/hrnet.yaml)
* [SEResNet](local/configs/seresnet_unet.yaml)

All these models take 2D patches of the dataset as input and provide predictions for those patches. The patches need to be stitched together to form a whole inline or crossline.

To understand the configuration files and the dafault parameters refer to this [section in the top level README](../../../README.md#configuration-files)

### Setup

Please set up a conda environment following the instructions in the top-level [README.md](../../../README.md#setting-up-environment) file.
Also follow instructions for [downloading and preparing](../../../README.md#penobscot) the data.
    
### Usage
- [`train.sh`](local/train.sh) - Will train the Segmentation model. The default configuration will execute for 300 epochs which will complete in around 3 days on a V100 GPU. During these 300 epochs succesive snapshots will be taken. By default a cyclic learning rate is applied.
- [`test.sh`](local/test.sh) - Will test your model against the test portion of the dataset. You will be able to view the performance of the trained model in Tensorboard.  

### Monitoring progress with TensorBoard
- from the this directory, run `tensorboard --logdir='output'` (all runtime logging information is
written to the `output` folder  
- open a web-browser and go to  either vmpublicip:6006 if running remotely or localhost:6006 if running locally  
> **NOTE**:If running remotely remember that the port must be open and accessible 
 
More information on Tensorboard can be found [here](https://www.tensorflow.org/get_started/summaries_and_tensorboard#launching_tensorboard).

