# Seismic Interpretation on Penobscot dataset
In this folder are training and testing scripts that work on the Penobscot dataset. 
You can run two different models on this dataset:
* [HRNet](local/configs/hrnet.yaml)
* [SEResNet](local/configs/seresnet_unet.yaml)


### Conda env setup

Please set up a conda environment following the instructions in the top-level [README.md](../../../../README.md) file.

### Data Download
To download the Penobscot dataset run the [download_penobscot.sh](../../../scripts/download_penobscot.sh) script, e.g.

```
data_dir="$HOME/data/penobscot"
mkdir -p "$data_dir"
./scripts/download_penobscot.sh "$data_dir"
```

Note that the specified download location should be configured with appropriate `write` permissions. On some Linux virtual machines, you may want to place the data into `/mnt` or `/data` folder so you have to make sure you have write access.

To make things easier, we suggested you use your home directory where you might run out of space. If this happens on an [Azure Data Science Virtual Machine](https://azure.microsoft.com/en-us/services/virtual-machines/data-science-virtual-machines/) you can resize the disk quite easily from [Azure Portal](https://portal.azure.com) - please see the [Troubleshooting](../../../README.md#troubleshooting) section in the [README](../../../README.md) regarding [how to do this](#how-to-resize-data-science-virtual-machine-disk).

### Data preparation

To prepare the data for the experiments (e.g. split into train/val/test), please run the following script (modifying arguments as desired):

```
python scripts/prepare_penobscot.py split_inline --data-dir=/data/penobscot --val-ratio=.1 --test-ratio=.2
```

### Configuration files
We use [YACS](https://github.com/rbgirshick/yacs) configuration library to manage configuration options for our experiments. 

We use the following three ways to pass options to the experiment scripts (e.g. train.py or test.py):

- __default.py__ - A project config file `default.py` is a one-stop reference point for all configurable options, and provides sensible defaults for all options.

- __yml config files__ - YAML configuration files under `configs/` are typically created one for each experiment. These are meant to be used for repeatable experiment runs and reproducible settings. Each configuration file only overrides the options that are changing in that experiment (e.g. options loaded from `defaults.py` during an experiment run will be overridden by options loaded from the yaml file)

- __command line__ - Finally, options can be passed in through `options` argument, and those will override options loaded from the configuration file. We created CLIs for all our scripts (using Python Fire library), so you can pass these options via command-line arguments.
    
    
### Usage
- [`train.sh`](local/train.sh) - Will train the Segmentation model. The default configuration will execute for 300 epochs which will complete in around 3 days on a V100 GPU. During these 300 epochs succesive snapshots will be taken. By default a cyclic learning rate is applied.
- [`test.sh`](local/test.sh) - Will test your model against the test portion of the dataset. You will be able to view the performance of the trained model in Tensorboard.  

### Monitoring progress with TensorBoard
- from the this directory, run `tensorboard --logdir='output'` (all runtime logging information is
written to the `output` folder  
- open a web-browser and go to  either vmpublicip:6006 if running remotely or localhost:6006 if running locally  
> **NOTE**:If running remotely remember that the port must be open and accessible 
 
More information on Tensorboard can be found [here](https://www.tensorflow.org/get_started/summaries_and_tensorboard#launching_tensorboard).

