from os import path
import numpy as np
import itertools
from sklearn.model_selection import train_test_split
import logging

DATA_ROOT = path.join("/mnt", "alaudah")
SPLITS = path.join(DATA_ROOT, "splits")
LABELS = path.join(DATA_ROOT, "train", "train_labels.npy")



def split_train_val(stride, per_val=0.2, loader_type="patch", labels_path=LABELS):
    # create inline and crossline pacthes for training and validation:
    print("hey")
    logger = logging.getLogger(__name__)

    labels = np.load(labels_path)
    iline, xline, depth = labels.shape
    logger.debug(f"Data shape [iline|xline|depth] {labels.shape}")
    # INLINE PATCHES: ------------------------------------------------
    i_list = []
    horz_locations = range(0, xline - stride, stride)
    vert_locations = range(0, depth - stride, stride)
    logger.debug("Generating Inline patches")
    logger.debug(horz_locations)
    logger.debug(vert_locations)
    for i in range(iline):
        # for every inline:
        # images are references by top-left corner:
        locations = [[j, k] for j in horz_locations for k in vert_locations]
        patches_list = [
            "i_" + str(i) + "_" + str(j) + "_" + str(k) for j, k in locations
        ]
        i_list.append(patches_list)

    # flatten the list
    i_list = list(itertools.chain(*i_list))

    # XLINE PATCHES: ------------------------------------------------
    x_list = []
    horz_locations = range(0, iline - stride, stride)
    vert_locations = range(0, depth - stride, stride)
    for j in range(xline):
        # for every xline:
        # images are references by top-left corner:
        locations = [[i, k] for i in horz_locations for k in vert_locations]
        patches_list = [
            "x_" + str(i) + "_" + str(j) + "_" + str(k) for i, k in locations
        ]
        x_list.append(patches_list)

    # flatten the list
    x_list = list(itertools.chain(*x_list))

    list_train_val = i_list + x_list

    # create train and test splits:
    list_train, list_val = train_test_split(
        list_train_val, test_size=per_val, shuffle=True
    )

    # write to files to disk:
    file_object = open(
        path.join(SPLITS, loader_type + "_train_val.txt"), "w"
    )
    file_object.write("\n".join(list_train_val))
    file_object.close()
    file_object = open(
        path.join(SPLITS, loader_type + "_train.txt"), "w"
    )
    file_object.write("\n".join(list_train))
    file_object.close()
    file_object = open(path.join(SPLITS, loader_type + "_val.txt"), "w")
    file_object.write("\n".join(list_val))
    file_object.close()
