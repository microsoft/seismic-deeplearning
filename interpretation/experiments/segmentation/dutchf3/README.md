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

### Data processing

To split the dataset into training and validation, please run the [prepare_data.py](prepare_data.py) script.

Example run:
```
python prepare_data.py split_non_overlapping --data-dir=/mnt/dutchf3 --stride=50 --log-config=../logging.conf
```

### Training and testing

Now you're all set to run training and testing experiments on the F3 Netherlands dataset. Please start from the `train.sh` and `test.sh` scripts under the `local/` and `distributed/` directories, which invoke the corresponding python scripts.


