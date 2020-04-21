# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
import scipy


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


def parse_labels_in_image(img):
    """
    Convert RGB image to class img.

    Args:
        img: 3-channel image array

    Returns:
        monotonically increasing class labels
    """

    # Add colors to this table to make it possible to have more classes
    class_color_coding = [
        [0, 0, 255],  # blue
        [0, 255, 0],  # green
        [0, 255, 255],  # cyan
        [255, 0, 0],  # red
        [255, 0, 255],  # blue
        [255, 255, 0],  # yellow
    ]

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


def get_grid(im_size):
    """
    getGrid returns z,x,y coordinates centered around (0,0,0)

    Args:
        im_size: size of window

    Returns
        numpy int array with size: 3 x im_size**3
    """
    win0 = np.linspace(-im_size[0] // 2, im_size[0] // 2, im_size[0])
    win1 = np.linspace(-im_size[1] // 2, im_size[1] // 2, im_size[1])
    win2 = np.linspace(-im_size[2] // 2, im_size[2] // 2, im_size[2])

    x0, x1, x2 = np.meshgrid(win0, win1, win2, indexing="ij")

    ex0 = np.expand_dims(x0.ravel(), 0)
    ex1 = np.expand_dims(x1.ravel(), 0)
    ex2 = np.expand_dims(x2.ravel(), 0)

    grid = np.concatenate((ex0, ex1, ex2), axis=0)

    return grid


def augment_flip(grid):
    """
    Random flip of non-depth axes.

    Args:
        grid: 3D coordinates of the voxel

    Returns:
        flipped grid coordinates
    """

    # Flip x axis
    if rand_bool():
        grid[1, :] = -grid[1, :]

    # Flip y axis
    if rand_bool():
        grid[2, :] = -grid[2, :]

    return grid


def trilinear_interpolation(input_array, indices):
    """
    Linear interpolation
    code taken from
    http://stackoverflow.com/questions/6427276/3d-interpolation-of-numpy-arrays-without-scipy

    Args:
        input_array: 3D data array
        indices: 3D grid coordinates

    Returns:
        interpolated input array
    """

    x_indices, y_indices, z_indices = indices[0:3]

    n0, n1, n2 = input_array.shape

    x0 = x_indices.astype(np.integer)
    y0 = y_indices.astype(np.integer)
    z0 = z_indices.astype(np.integer)
    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1

    # put all samples outside datacube to 0
    inds_out_of_range = (
        (x0 < 0)
        | (x1 < 0)
        | (y0 < 0)
        | (y1 < 0)
        | (z0 < 0)
        | (z1 < 0)
        | (x0 >= n0)
        | (x1 >= n0)
        | (y0 >= n1)
        | (y1 >= n1)
        | (z0 >= n2)
        | (z1 >= n2)
    )

    x0[inds_out_of_range] = 0
    y0[inds_out_of_range] = 0
    z0[inds_out_of_range] = 0
    x1[inds_out_of_range] = 0
    y1[inds_out_of_range] = 0
    z1[inds_out_of_range] = 0

    x = x_indices - x0
    y = y_indices - y0
    z = z_indices - z0
    output = (
        input_array[x0, y0, z0] * (1 - x) * (1 - y) * (1 - z)
        + input_array[x1, y0, z0] * x * (1 - y) * (1 - z)
        + input_array[x0, y1, z0] * (1 - x) * y * (1 - z)
        + input_array[x0, y0, z1] * (1 - x) * (1 - y) * z
        + input_array[x1, y0, z1] * x * (1 - y) * z
        + input_array[x0, y1, z1] * (1 - x) * y * z
        + input_array[x1, y1, z0] * x * y * (1 - z)
        + input_array[x1, y1, z1] * x * y * z
    )

    output[inds_out_of_range] = 0
    return output


def rand_float(low, high):
    """
    Generate random floating point number between two limits

    Args:
        low: low limit
        high: high limit

    Returns:
        single random floating point number
    """
    return (high - low) * np.random.random_sample() + low


def rand_int(low, high):
    """
    Generate random integer between two limits

    Args:
        low: low limit
        high: high limit

    Returns:
        random integer between two limits
    """
    return np.random.randint(low, high)


def rand_bool():
    """
    Generate random boolean.

    Returns:
        Random boolean
    """
    return bool(np.random.randint(0, 2))


def trilinear_interpolation(input_array, indices):
    """
    Linear interpolation
    code taken from
    http://stackoverflow.com/questions/6427276/3d-interpolation-of-numpy-arrays-without-scipy

    Args:
        input_array: 3D data array
        indices: 3D grid coordinates

    Returns:
        interpolated input array
    """

    x_indices, y_indices, z_indices = indices[0:3]

    n0, n1, n2 = input_array.shape

    x0 = x_indices.astype(np.integer)
    y0 = y_indices.astype(np.integer)
    z0 = z_indices.astype(np.integer)
    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1

    # put all samples outside datacube to 0
    inds_out_of_range = (
        (x0 < 0)
        | (x1 < 0)
        | (y0 < 0)
        | (y1 < 0)
        | (z0 < 0)
        | (z1 < 0)
        | (x0 >= n0)
        | (x1 >= n0)
        | (y0 >= n1)
        | (y1 >= n1)
        | (z0 >= n2)
        | (z1 >= n2)
    )

    x0[inds_out_of_range] = 0
    y0[inds_out_of_range] = 0
    z0[inds_out_of_range] = 0
    x1[inds_out_of_range] = 0
    y1[inds_out_of_range] = 0
    z1[inds_out_of_range] = 0

    x = x_indices - x0
    y = y_indices - y0
    z = z_indices - z0
    output = (
        input_array[x0, y0, z0] * (1 - x) * (1 - y) * (1 - z)
        + input_array[x1, y0, z0] * x * (1 - y) * (1 - z)
        + input_array[x0, y1, z0] * (1 - x) * y * (1 - z)
        + input_array[x0, y0, z1] * (1 - x) * (1 - y) * z
        + input_array[x1, y0, z1] * x * (1 - y) * z
        + input_array[x0, y1, z1] * (1 - x) * y * z
        + input_array[x1, y1, z0] * x * y * (1 - z)
        + input_array[x1, y1, z1] * x * y * z
    )

    output[inds_out_of_range] = 0
    return output


def rand_float(low, high):
    """
    Generate random floating point number between two limits

    Args:
        low: low limit
        high: high limit

    Returns:
        single random floating point number
    """
    return (high - low) * np.random.random_sample() + low


def rand_int(low, high):
    """
    Generate random integer between two limits

    Args:
        low: low limit
        high: high limit

    Returns:
        random integer between two limits
    """
    return np.random.randint(low, high)


def rand_bool():
    """
    Generate random boolean.

    Returns:
        Random boolean
    """
    return bool(np.random.randint(0, 2))
