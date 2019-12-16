# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license.

# code modified from https://github.com/waldeland/CNN-for-ASI

from __future__ import print_function

import torch
import numpy as np
from torch.autograd import Variable
from scipy.interpolate import interpn
import sys
import time

# global parameters
ST = 0
LAST_UPDATE = 0


def interpret(
    network, data, data_info, slice, slice_no, im_size, subsampl, return_full_size=True, use_gpu=True,
):
    """
    Down-samples a slice from the classified image and upsamples to full resolution if needed. Basically
    given a full 3D-classified voxel at a particular resolution (say we classify every n-th pixel as given by the
    subsampl variable below) we take a particular slice from the voxel and optoinally blow it up to full resolution
    as if we classified every single pixel.

    Args:
        network: pytorch model definition
        data: input voxel
        data_info: input voxel information
        slice: slice type which we want to interpret
        slice_no: slice number
        im_size: size of the voxel
        subsampl: at what resolution do we want to subsample, e.g. we move across every subsampl pixels
        return_full_size: boolean flag, enable if you want to return full size without downsampling
        use_gpu: boolean flag to use the GPU

    Returns:
        upsampled slice

    """

    # Wrap np.linspace in compact function call
    ls = lambda N: np.linspace(0, N - 1, N, dtype="int")

    # Size of cube
    N0, N1, N2 = data.shape

    # Coords for full cube
    x0_range = ls(N0)
    x1_range = ls(N1)
    x2_range = ls(N2)

    # Coords for subsampled cube
    pred_points = (x0_range[::subsampl], x1_range[::subsampl], x2_range[::subsampl])

    # Select slice
    if slice == "full":
        class_cube = data[::subsampl, ::subsampl, ::subsampl] * 0

    elif slice == "inline":
        slice_no = slice_no - data_info["inline_start"]
        class_cube = data[::subsampl, 0:1, ::subsampl] * 0
        x1_range = np.array([slice_no])
        pred_points = (pred_points[0], pred_points[2])

    elif slice == "crossline":
        slice_no = slice_no - data_info["crossline_start"]
        class_cube = data[::subsampl, ::subsampl, 0:1,] * 0
        x2_range = np.array([slice_no])
        pred_points = (pred_points[0], pred_points[1])

    elif slice == "timeslice":
        slice_no = slice_no - data_info["timeslice_start"]
        class_cube = data[0:1, ::subsampl, ::subsampl] * 0
        x0_range = np.array([slice_no])
        pred_points = (pred_points[1], pred_points[2])

    # Grid for small class slice/cube
    n0, n1, n2 = class_cube.shape
    x0_grid, x1_grid, x2_grid = np.meshgrid(ls(n0,), ls(n1), ls(n2), indexing="ij")

    # Grid for full slice/cube
    X0_grid, X1_grid, X2_grid = np.meshgrid(x0_range, x1_range, x2_range, indexing="ij")

    # Indexes for large cube at small cube pixels
    X0_grid_sub = X0_grid[::subsampl, ::subsampl, ::subsampl]
    X1_grid_sub = X1_grid[::subsampl, ::subsampl, ::subsampl]
    X2_grid_sub = X2_grid[::subsampl, ::subsampl, ::subsampl]

    # Get half window size
    w = im_size // 2

    # Loop through center pixels in output cube
    for i in range(X0_grid_sub.size):

        # Get coordinates in small and large cube
        x0 = x0_grid.ravel()[i]
        x1 = x1_grid.ravel()[i]
        x2 = x2_grid.ravel()[i]

        X0 = X0_grid_sub.ravel()[i]
        X1 = X1_grid_sub.ravel()[i]
        X2 = X2_grid_sub.ravel()[i]

        # Only compute when a full 65x65x65 cube can be extracted around center pixel
        if X0 > w and X1 > w and X2 > w and X0 < N0 - w + 1 and X1 < N1 - w + 1 and X2 < N2 - w + 1:

            # Get mini-cube around center pixel
            mini_cube = data[X0 - w : X0 + w + 1, X1 - w : X1 + w + 1, X2 - w : X2 + w + 1]

            # Get predicted "probabilities"
            mini_cube = Variable(torch.FloatTensor(mini_cube[np.newaxis, np.newaxis, :, :, :]))
            if use_gpu:
                mini_cube = mini_cube.cuda()
            out = network(mini_cube)
            out = out.data.cpu().numpy()

            out = out[:, :, out.shape[2] // 2, out.shape[3] // 2, out.shape[4] // 2]
            out = np.squeeze(out)

            # Make one output pr output channel
            if not isinstance(class_cube, list):
                class_cube = np.split(np.repeat(class_cube[:, :, :, np.newaxis], out.size, 3), out.size, axis=3,)

            # Insert into output
            if out.size == 1:
                class_cube[0][x0, x1, x2] = out
            else:
                for i in range(out.size):
                    class_cube[i][x0, x1, x2] = out[i]

        # Keep user informed about progress
        if slice == "full":
            printProgressBar(i, x0_grid.size)

    # Resize to input size
    if return_full_size:
        if slice == "full":
            print("Interpolating down sampled results to fit input cube")

        N = X0_grid.size

        # Output grid
        if slice == "full":
            grid_output_cube = np.concatenate(
                [X0_grid.reshape([N, 1]), X1_grid.reshape([N, 1]), X2_grid.reshape([N, 1]),], 1,
            )
        elif slice == "inline":
            grid_output_cube = np.concatenate([X0_grid.reshape([N, 1]), X2_grid.reshape([N, 1])], 1)
        elif slice == "crossline":
            grid_output_cube = np.concatenate([X0_grid.reshape([N, 1]), X1_grid.reshape([N, 1])], 1)
        elif slice == "timeslice":
            grid_output_cube = np.concatenate([X1_grid.reshape([N, 1]), X2_grid.reshape([N, 1])], 1)

        # Interpolation
        for i in range(len(class_cube)):
            is_int = (
                np.sum(
                    np.unique(class_cube[i]).astype("float") - np.unique(class_cube[i]).astype("int32").astype("float")
                )
                == 0
            )
            class_cube[i] = interpn(
                pred_points,
                class_cube[i].astype("float").squeeze(),
                grid_output_cube,
                method="linear",
                fill_value=0,
                bounds_error=False,
            )
            class_cube[i] = class_cube[i].reshape([x0_range.size, x1_range.size, x2_range.size])

            # If ouput is class labels we convert the interpolated array to ints
            if is_int:
                class_cube[i] = class_cube[i].astype("int32")

        if slice == "full":
            print("Finished interpolating")

    # Squeeze outputs
    for i in range(len(class_cube)):
        class_cube[i] = class_cube[i].squeeze()

    return class_cube


# TODO: this should probably be replaced with TQDM
def print_progress_bar(iteration, total, prefix="", suffix="", decimals=1, length=100, fill="="):
    """
    Privides a progress bar implementation.

    Adapted from https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console/14879561#14879561

    Args:
        iteration: iteration number
        total: total number of iterations
        prefix: comment prefix in display
        suffix: comment suffix in display
        decimals: how many decimals to display
        length: character length of progress bar
        fill: character to display as progress bar

    """
    global ST, LAST_UPDATE

    # Expect itteration to go from 0 to N-1
    iteration = iteration + 1

    # Only update every 5 second
    if time.time() - LAST_UPDATE < 5:
        if iteration == total:
            time.sleep(1)
        else:
            return

    if iteration <= 1:
        st = time.time()
        exp_h = ""
        exp_m = ""
        exp_s = ""
    elif iteration == total:
        exp_time = time.time() - ST
        exp_h = int(exp_time / 3600)
        exp_m = int(exp_time / 60 - exp_h * 60.0)
        exp_s = int(exp_time - exp_m * 60.0 - exp_h * 3600.0)
    else:
        exp_time = (time.time() - ST) / (iteration - 1) * total - (time.time() - ST)
        exp_h = int(exp_time / 3600)
        exp_m = int(exp_time / 60 - exp_h * 60.0)
        exp_s = int(exp_time - exp_m * 60.0 - exp_h * 3600.0)

    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + "-" * (length - filled_length)
    if iteration != total:
        print("\r%s |%s| %s%% %s - %sh %smin %ss left" % (prefix, bar, percent, suffix, exp_h, exp_m, exp_s))
    else:
        print("\r%s |%s| %s%% %s - %sh %smin %ss " % (prefix, bar, percent, suffix, exp_h, exp_m, exp_s))
    sys.stdout.write("\033[F")
    # Print New Line on Complete
    if iteration == total:
        print("")
    # last_update = time.time()


# TODO: rewrite this whole function to get rid of excepts
# TODO: also not sure what this function is for - it's almost as if it's not needed - try to remove it.
def gpu_no_of_var(var):
    """
    Function that returns the GPU number or whether the tensor is on GPU or not

    Args:
        var: torch tensor

    Returns:
        The CUDA device that the torch tensor is on, or whether the tensor is on GPU

    """

    try:
        is_cuda = next(var.parameters()).is_cuda
    except:
        is_cuda = var.is_cuda

    if is_cuda:
        try:
            return next(var.parameters()).get_device()
        except:
            return var.get_device()
    else:
        return False


# TODO: remove all the try except statements
def var_to_np(var):
    """
    Take a pyTorch tensor and convert it to numpy array of the same shape, as the name suggests.

    Args:
        var: input variable

    Returns:
        numpy array of the tensor

    """
    if type(var) in [np.array, np.ndarray]:
        return var

    # If input is list we do this for all elements
    if type(var) == type([]):
        out = []
        for v in var:
            out.append(var_to_np(v))
        return out

    try:
        var = var.cpu()
    except:
        None
    try:
        var = var.data
    except:
        None
    try:
        var = var.numpy()
    except:
        None

    if type(var) == tuple:
        var = var[0]
    return var


def compute_accuracy(predicted_class, labels):
    """
    Accuracy performance metric which needs to be computed

    Args:
        predicted_class: pyTorch tensor with predictions
        labels: pyTorch tensor with ground truth labels

    Returns:
        Accuracy calculation as a dictionary per class and average class accuracy across classes

    """
    labels = var_to_np(labels)
    predicted_class = var_to_np(predicted_class)

    accuracies = {}
    for cls in np.unique(labels):
        if cls >= 0:
            accuracies["accuracy_class_" + str(cls)] = int(np.mean(predicted_class[labels == cls] == cls) * 100)
    accuracies["average_class_accuracy"] = np.mean([acc for acc in accuracies.values()])
    return accuracies
