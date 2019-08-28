import numpy as np

""" Returns a batch of augmented samples with center pixels randomly drawn from label_coordinates"""


def get_random_batch(
    data_cube,
    label_coordinates,
    im_size,
    batch_size,
    random_flip=False,
    random_stretch=None,
    random_rot_xy=None,
    random_rot_z=None,
):

    # Make 3 im_size elements
    if type(im_size) == type(1):
        im_size = [im_size, im_size, im_size]

    # Output arrays
    batch = np.zeros([batch_size, 1, im_size[0], im_size[1], im_size[2]])
    labels = np.zeros([batch_size])

    class_keys = list(label_coordinates)
    n_classes = len(class_keys)

    # Loop through batch
    n_for_class = 0
    class_ind = 0
    for i in range(batch_size):

        # Start by getting a grid centered around (0,0,0)
        grid = getGrid(im_size)

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
        coord = coords_for_class[:, random_index : random_index + 1]

        # Move grid to be centered around this location
        grid += coord

        # Interpolate samples at grid from the data:
        sample = trilinear_interpolation(data_cube, grid)

        # Insert in output arrays
        labels[i] = class_ind
        batch[i, 0, :, :, :] = np.reshape(
            sample, (im_size[0], im_size[1], im_size[2])
        )

        # We seek to have a balanced batch with equally many samples from each class.
        n_for_class += 1
        if n_for_class + 1 > int(0.5 + batch_size / float(n_classes)):
            if class_ind < n_classes - 1:
                class_ind += 1
                n_for_class = 0

    return batch, labels


""" Get x,y,z grid for sample """


def getGrid(im_size):
    """
    getGrid returns z,x,y coordinates centered around (0,0,0) 
    :param im_size: size of window
    :return: numpy int array with size: 3 x im_size**3
    """
    win0 = np.linspace(-im_size[0] // 2, im_size[0] // 2, im_size[0])
    win1 = np.linspace(-im_size[1] // 2, im_size[1] // 2, im_size[1])
    win2 = np.linspace(-im_size[2] // 2, im_size[2] // 2, im_size[2])

    x0, x1, x2 = np.meshgrid(win0, win1, win2, indexing="ij")
    x0 = np.expand_dims(x0.ravel(), 0)
    x1 = np.expand_dims(x1.ravel(), 0)
    x2 = np.expand_dims(x2.ravel(), 0)
    grid = np.concatenate((x0, x1, x2), axis=0)

    return grid


""" Random flip of non-depth axes """


def augment_flip(grid):
    # Flip x axis
    if rand_bool():
        grid[1, :] = -grid[1, :]

    # Flip y axis
    if rand_bool():
        grid[2, :] = -grid[2, :]

    return grid


""" Random stretch/scale """


def augment_stretch(grid, stretch_factor):
    stretch = rand_float(-stretch_factor, stretch_factor)
    grid *= 1 + stretch
    return grid


""" Random rotation """


def augment_rot_xy(grid, random_rot_xy):
    theta = np.deg2rad(rand_float(-random_rot_xy, random_rot_xy))
    x = grid[2, :] * np.cos(theta) - grid[1, :] * np.sin(theta)
    y = grid[2, :] * np.sin(theta) + grid[1, :] * np.cos(theta)
    grid[1, :] = x
    grid[2, :] = y
    return grid


""" Random tilt """


def augment_rot_z(grid, random_rot_z):
    theta = np.deg2rad(rand_float(-random_rot_z, random_rot_z))
    z = grid[0, :] * np.cos(theta) - grid[1, :] * np.sin(theta)
    x = grid[0, :] * np.sin(theta) + grid[1, :] * np.cos(theta)
    grid[0, :] = z
    grid[1, :] = x
    return grid


""" Linear interpolation """


def trilinear_interpolation(input_array, indices):

    # http://stackoverflow.com/questions/6427276/3d-interpolation-of-numpy-arrays-without-scipy
    output = np.empty(indices[0].shape)
    x_indices = indices[0]
    y_indices = indices[1]
    z_indices = indices[2]

    N0, N1, N2 = input_array.shape

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
        | (x0 >= N0)
        | (x1 >= N0)
        | (y0 >= N1)
        | (y1 >= N1)
        | (z0 >= N2)
        | (z1 >= N2)
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


""" Functions to get random variables: """


def rand_float(low, high):
    return (high - low) * np.random.random_sample() + low


def rand_int(low, high):
    return np.random.randint(low, high)


def rand_bool():
    return bool(np.random.randint(0, 2))


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
