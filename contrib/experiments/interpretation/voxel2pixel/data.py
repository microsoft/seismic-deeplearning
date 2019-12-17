# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license.

# code modified from https://github.com/waldeland/CNN-for-ASI

from __future__ import print_function
from os.path import isfile, join

import segyio
from os import listdir
import numpy as np
import scipy.misc


def read_segy(filename):
    """
    Read in a SEGY-format file given a filename

    Args:
        filename: input filename

    Returns:
        numpy data array and its info as a dictionary (tuple)

    """
    print("Loading data cube from", filename, "with:")

    # Read full data cube
    data = segyio.tools.cube(filename)

    # Put temporal axis first
    data = np.moveaxis(data, -1, 0)

    # Make data cube fast to access
    data = np.ascontiguousarray(data, "float32")

    # Read meta data
    segyfile = segyio.open(filename, "r")
    print("  Crosslines: ", segyfile.xlines[0], ":", segyfile.xlines[-1])
    print("  Inlines:    ", segyfile.ilines[0], ":", segyfile.ilines[-1])
    print("  Timeslices: ", "1", ":", data.shape[0])

    # Make dict with cube-info
    # TODO: read this from segy
    # Read dt and other params needed to do create a new
    data_info = {
        "crossline_start": segyfile.xlines[0],
        "inline_start": segyfile.ilines[0],
        "timeslice_start": 1,
        "shape": data.shape,
    }

    return data, data_info


def write_segy(out_filename, in_filename, out_cube):
    """
    Writes out_cube to a segy-file (out_filename) with same header/size as in_filename

    Args:
        out_filename:
        in_filename:
        out_cube:

    Returns:

    """
    # Select last channel
    if type(out_cube) is list:
        out_cube = out_cube[-1]

    print("Writing interpretation to " + out_filename)
    # Copy segy file
    from shutil import copyfile

    copyfile(in_filename, out_filename)

    # Moving temporal axis back again
    out_cube = np.moveaxis(out_cube, 0, -1)

    # Open out-file
    with segyio.open(out_filename, "r+") as src:
        iline_start = src.ilines[0]
        dtype = src.iline[iline_start].dtype
        # loop through inlines and insert output
        for i in src.ilines:
            iline = out_cube[i - iline_start, :, :]
            src.iline[i] = np.ascontiguousarray(iline.astype(dtype))

    # TODO: rewrite this whole function
    # Moving temporal axis first again - just in case the user want to keep working on it
    out_cube = np.moveaxis(out_cube, -1, 0)

    print("Writing interpretation - Finished")
    return


# Alternative writings for slice-type
inline_alias = ["inline", "in-line", "iline", "y"]
crossline_alias = ["crossline", "cross-line", "xline", "x"]
timeslice_alias = ["timeslice", "time-slice", "t", "z", "depthslice", "depth"]


def read_labels(fname, data_info):
    """
    Read labels from an image.

    Args:
        fname: filename of labelling mask (image)
        data_info: dictionary describing the data

    Returns:
        list of labels and list of coordinates
    """

    label_imgs = []
    label_coordinates = {}

    # Find image files in folder

    tmp = fname.split("/")[-1].split("_")
    slice_type = tmp[0].lower()
    tmp = tmp[1].split(".")
    slice_no = int(tmp[0])

    if slice_type not in inline_alias + crossline_alias + timeslice_alias:
        print(
            "File:", fname, "could not be loaded.", "Unknown slice type",
        )
        return None

    if slice_type in inline_alias:
        slice_type = "inline"
    if slice_type in crossline_alias:
        slice_type = "crossline"
    if slice_type in timeslice_alias:
        slice_type = "timeslice"

    # Read file
    print("Loading labels for", slice_type, slice_no, "with")
    img = scipy.misc.imread(fname)
    img = interpolate_to_fit_data(img, slice_type, slice_no, data_info)
    label_img = parse_labels_in_image(img)

    # Get coordinates for slice
    coords = get_coordinates_for_slice(slice_type, slice_no, data_info)

    # Loop through labels in label_img and append to label_coordinates
    for cls in np.unique(label_img):
        if cls > -1:
            if str(cls) not in label_coordinates.keys():
                label_coordinates[str(cls)] = np.array(np.zeros([3, 0]))
            inds_with_cls = label_img == cls
            cords_with_cls = coords[:, inds_with_cls.ravel()]
            label_coordinates[str(cls)] = np.concatenate((label_coordinates[str(cls)], cords_with_cls), 1)
            print(
                " ", str(np.sum(inds_with_cls)), "labels for class", str(cls),
            )
    if len(np.unique(label_img)) == 1:
        print(" ", 0, "labels", str(cls))

    # Add label_img to output
    label_imgs.append([label_img, slice_type, slice_no])

    return label_imgs, label_coordinates


# Add colors to this table to make it possible to have more classes
class_color_coding = [
    [0, 0, 255],  # blue
    [0, 255, 0],  # green
    [0, 255, 255],  # cyan
    [255, 0, 0],  # red
    [255, 0, 255],  # blue
    [255, 255, 0],  # yellow
]


def parse_labels_in_image(img):
    """
    Convert RGB image to class img.

    Args:
        img: 3-channel image array

    Returns:
        monotonically increasing class labels
    """
    label_img = np.int16(img[:, :, 0]) * 0 - 1  # -1 = no class

    # decompose color channels (#Alpha is ignored)
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]

    # Alpha channel
    if img.shape[2] == 4:
        a = 1 - img.shape[2] // 255
        r = r * a
        g = g * a
        b = b * a

    tolerance = 1
    # Go through classes and find pixels with this class
    cls = 0
    for color in class_color_coding:
        # Find pixels with these labels
        inds = (
            (np.abs(r - color[0]) < tolerance) & (np.abs(g - color[1]) < tolerance) & (np.abs(b - color[2]) < tolerance)
        )
        label_img[inds] = cls
        cls += 1

    return label_img


def interpolate_to_fit_data(img, slice_type, slice_no, data_info):
    """
    Function to resize image if needed

    Args:
        img: image array
        slice_type: inline, crossline or timeslice slice type
        slice_no: slice number
        data_info: data info dictionary distracted from SEGY file

    Returns:
        resized image array

    """

    # Get wanted output size
    if slice_type == "inline":
        n0 = data_info["shape"][0]
        n1 = data_info["shape"][2]
    elif slice_type == "crossline":
        n0 = data_info["shape"][0]
        n1 = data_info["shape"][1]
    elif slice_type == "timeslice":
        n0 = data_info["shape"][1]
        n1 = data_info["shape"][2]
    return scipy.misc.imresize(img, (n0, n1), interp="nearest")


def get_coordinates_for_slice(slice_type, slice_no, data_info):
    """

    Get coordinates for slice in the full cube

    Args:
        slice_type: type of slice, e.g. inline, crossline, etc
        slice_no: slice number
        data_info: data dictionary array

    Returns:
        index coordinates of the voxel

    """
    ds = data_info["shape"]

    # Coordinates for cube
    x0, x1, x2 = np.meshgrid(
        np.linspace(0, ds[0] - 1, ds[0]),
        np.linspace(0, ds[1] - 1, ds[1]),
        np.linspace(0, ds[2] - 1, ds[2]),
        indexing="ij",
    )
    if slice_type == "inline":
        start = data_info["inline_start"]
        slice_no = slice_no - start

        x0 = x0[:, slice_no, :]
        x1 = x1[:, slice_no, :]
        x2 = x2[:, slice_no, :]
    elif slice_type == "crossline":
        start = data_info["crossline_start"]
        slice_no = slice_no - start
        x0 = x0[:, :, slice_no]
        x1 = x1[:, :, slice_no]
        x2 = x2[:, :, slice_no]

    elif slice_type == "timeslice":
        start = data_info["timeslice_start"]
        slice_no = slice_no - start
        x0 = x0[slice_no, :, :]
        x1 = x1[slice_no, :, :]
        x2 = x2[slice_no, :, :]

    # Collect indexes
    x0 = np.expand_dims(x0.ravel(), 0)
    x1 = np.expand_dims(x1.ravel(), 0)
    x2 = np.expand_dims(x2.ravel(), 0)
    coords = np.concatenate((x0, x1, x2), axis=0)

    return coords


def get_slice(data, data_info, slice_type, slice_no, window=0):
    """
    Return data-slice

    Args:
        data: input 3D voxel numpy array
        data_info: data info dictionary
        slice_type: type of slice, like inline, crossline, etc
        slice_no: slice number
        window: window size around center pixel

    Returns:
        2D slice of the voxel as a numpy array

    """

    if slice_type == "inline":
        start = data_info["inline_start"]

    elif slice_type == "crossline":
        start = data_info["crossline_start"]

    elif slice_type == "timeslice":
        start = data_info["timeslice_start"]

    slice_no = slice_no - start
    slice = data[:, slice_no - window : slice_no + window + 1, :]

    return np.squeeze(slice)
