import numpy as np

def get_random_batch(
        data_cube,
        label_coordinates,
        im_size,
        num_batch_size,
        random_flip=False,
        random_stretch=None,
        random_rot_xy=None,
        random_rot_z=None,
):
    """
    Returns a batch of augmented samples with center pixels randomly drawn from label_coordinates

    Args:
        data_cube: 3D numpy array with floating point velocity values
        label_coordinates: 3D coordinates of the labeled training slice
        im_size: size of the 3D voxel which we're cutting out around each label_coordinate
        num_batch_size: size of the batch
        random_flip: bool to perform random voxel flip
        random_stretch: bool to enable random stretch
        random_rot_xy: bool to enable random rotation of the voxel around dim-0 and dim-1
        random_rot_z: bool to enable random rotation around dim-2

    Returns:
        a tuple of batch numpy array array of data with dimension
        (batch, 1, data_cube.shape[0], data_cube.shape[1], data_cube.shape[2]) and the associated labels as an array
        of size (batch).
    """

    # Make 3 im_size elements
    if isinstance(im_size, int):
        im_size = [im_size, im_size, im_size]

    # Output arrays
    batch = np.zeros([num_batch_size, 1, im_size[0], im_size[1], im_size[2]])
    ret_labels = np.zeros([num_batch_size])

    class_keys = list(label_coordinates)
    n_classes = len(class_keys)

    # Loop through batch
    n_for_class = 0
    class_ind = 0
    for i in range(num_batch_size):

        # Start by getting a grid centered around (0,0,0)
        grid = get_grid(im_size)

        # Apply random flip
        if random_flip:
            grid = augment_flip(grid)

        # Apply random rotations
        if random_rot_xy:
            grid = augment_rot_xy(grid, random_rot_xy)
        if random_rot_z:
            grid = augment_rot_z(grid, random_rot_z)

        # Apply random stretch
        if random_stretch:
            grid = augment_stretch(grid, random_stretch)

        # Pick random location from the label_coordinates for this class:
        coords_for_class = label_coordinates[class_keys[class_ind]]
        random_index = rand_int(0, coords_for_class.shape[1])
        coord = coords_for_class[:, random_index: random_index + 1]

        # Move grid to be centered around this location
        grid += coord

        # Interpolate samples at grid from the data:
        sample = trilinear_interpolation(data_cube, grid)

        # Insert in output arrays
        ret_labels[i] = class_ind
        batch[i, 0, :, :, :] = np.reshape(
            sample, (im_size[0], im_size[1], im_size[2])
        )

        # We seek to have a balanced batch with equally many samples from each class.
        n_for_class += 1
        if n_for_class + 1 > int(0.5 + num_batch_size / float(n_classes)):
            if class_ind < n_classes - 1:
                class_ind += 1
                n_for_class = 0

    return batch, ret_labels


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


def augment_stretch(grid, stretch_factor):
    """
    Random stretch/scale

    Args:
        grid: 3D coordinate grid of the voxel
        stretch_factor: this is actually a boolean which triggers stretching
        TODO: change this to just call the function and not do -1,1 in rand_float

    Returns:
        stretched grid coordinates
    """
    stretch = rand_float(-stretch_factor, stretch_factor)
    grid *= 1 + stretch
    return grid


def augment_rot_xy(grid, random_rot_xy):
    """
    Random rotation

    Args:
        grid: coordinate grid list  of 3D points
        random_rot_xy: this is actually a boolean which triggers rotation
        TODO: change this to just call the function and not do -1,1 in rand_float

    Returns:
        randomly rotated grid
    """
    theta = np.deg2rad(rand_float(-random_rot_xy, random_rot_xy))
    x = grid[2, :] * np.cos(theta) - grid[1, :] * np.sin(theta)
    y = grid[2, :] * np.sin(theta) + grid[1, :] * np.cos(theta)
    grid[1, :] = x
    grid[2, :] = y
    return grid


def augment_rot_z(grid, random_rot_z):
    """
    Random tilt around z-axis (dim-2)

    Args:
        grid: coordinate grid list of 3D points
        random_rot_z: this is actually a boolean which triggers rotation
        TODO: change this to just call the function and not do -1,1 in rand_float

    Returns:
        randomly tilted coordinate grid
    """
    theta = np.deg2rad(rand_float(-random_rot_z, random_rot_z))
    z = grid[0, :] * np.cos(theta) - grid[1, :] * np.sin(theta)
    x = grid[0, :] * np.sin(theta) + grid[1, :] * np.cos(theta)
    grid[0, :] = z
    grid[1, :] = x
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


"""
TODO: the following is not needed and should be added as tests later.

# Test the batch-functions
if __name__ == "__main__":
    from data import readSEGY, readLabels, get_slice
    import tb_logger
    import numpy as np
    import os

    data, data_info = readSEGY(os.path.join("F3", "data.segy"))

    train_coordinates = {"1": np.expand_dims(np.array([50, 50, 50]), 1)}

    logger = tb_logger.TBLogger("log", "batch test")

    [batch, labels] = get_random_batch(data, train_coordinates, 65, 32)
    logger.log_images("normal", batch)

    [batch, labels] = get_random_batch(
        data, train_coordinates, 65, 32, random_flip=True
    )
    logger.log_images("flipping", batch)

    [batch, labels] = get_random_batch(
        data, train_coordinates, 65, 32, random_stretch=0.50
    )
    logger.log_images("stretching", batch)

    [batch, labels] = get_random_batch(
        data, train_coordinates, 65, 32, random_rot_xy=180
    )
    logger.log_images("rot", batch)

    [batch, labels] = get_random_batch(
        data, train_coordinates, 65, 32, random_rot_z=15
    )
    logger.log_images("dip", batch)

    train_cls_imgs, train_coordinates = readLabels(
        os.path.join("F3", "train"), data_info
    )
    [batch, labels] = get_random_batch(data, train_coordinates, 65, 32)
    logger.log_images("salt", batch[:16, :, :, :, :])
    logger.log_images("not salt", batch[16:, :, :, :, :])

    logger.log_images("data", data[:, :, 50])
"""