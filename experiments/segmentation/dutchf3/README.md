## F3 Netherlands Experiments Setup

### Conda env setup

Please set up a conda environment following the instructions in the top-level [README.md](../../../../README.md) file.

### Data download

To download the F3 Netherlands data set, please follow the data download instructions at:
https://github.com/olivesgatech/facies_classification_benchmark


Once you've downloaded the F3 data set, you'll find your data files in the following directory tree:

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

### Data preparation

To split the dataset into training and validation, please run the [prepare_data.py](prepare_data.py) script.

Example run:
```
// To split data into sections
python prepare_data.py split_train_val --data-dir=/mnt/dutchf3 --loader-type="section" --log-config=../logging.conf

// To split data into patches
python prepare_data.py split_train_val --data-dir=/mnt/dutchf3 --loader-type="patch" --stride=50 --log-config=../logging.conf
```

Please see `prepare_data.py` script for more arguments.


### Configuration files
We use [YACS](https://github.com/rbgirshick/yacs) configuration library to manage configuration options for our experiments. 

We use the following three ways to pass options to the experiment scripts (e.g. train.py or test.py):

- __default.py__ - A project config file `default.py` is a one-stop reference point for all configurable options, and provides sensible defaults for all options.

- __yml config files__ - YAML configuration files under `configs/` are typically created one for each experiment. These are meant to be used for repeatable experiment runs and reproducible settings. Each configuration file only overrides the options that are changing in that experiment (e.g. options loaded from `defaults.py` during an experiment run will be overridden by options loaded from the yaml file)

- __command line__ - Finally, options can be passed in through `options` argument, and those will override options loaded from the configuration file. We created CLIs for all our scripts (using Python Fire library), so you can pass these options via command-line arguments.
    

### Running experiments

Now you're all set to run training and testing experiments on the F3 Netherlands dataset. Please start from the `train.sh` and `test.sh` scripts under the `local/` and `distributed/` directories, which invoke the corresponding python scripts. Take a look at the project configurations in (e.g in `default.py`) for experiment options and modify if necessary. 
