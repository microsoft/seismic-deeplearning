# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license.

# code modified from https://github.com/waldeland/CNN-for-ASI

from __future__ import print_function
from os.path import join
import torch
from torch import nn
from data import read_segy, read_labels, get_slice
from batch import get_random_batch
from torch.autograd import Variable
from texture_net import TextureNet
import tb_logger
import utils

# Parameters
ROOT_PATH = "/home/maxkaz/data/dutchf3"
INPUT_VOXEL = "data.segy"
TRAIN_MASK = "inline_339.png"
VAL_MASK = "inline_405.png"
IM_SIZE = 65
# If you have a GPU with little memory, try reducing this to 16 (may degrade results)
BATCH_SIZE = 32
# Switch to toggle the use of GPU or not
USE_GPU = True
# Log progress on tensor board
LOG_TENSORBOARD = True

# the rest of the code
if LOG_TENSORBOARD:
    logger = tb_logger.TBLogger("log", "Train")

# This is the network definition proposed in the paper
network = TextureNet(n_classes=2)

# Loss function - Softmax function is included
cross_entropy = nn.CrossEntropyLoss()

# Optimizer to control step size in gradient descent
optimizer = torch.optim.Adam(network.parameters())

# Transfer model to gpu
if USE_GPU and torch.cuda.is_available():
    network = network.cuda()

# Load the data cube and labels
data, data_info = read_segy(join(ROOT_PATH, INPUT_VOXEL))
train_class_imgs, train_coordinates = read_labels(join(ROOT_PATH, TRAIN_MASK), data_info)
val_class_imgs, _ = read_labels(join(ROOT_PATH, VAL_MASK), data_info)

# Plot training/validation data with labels
if LOG_TENSORBOARD:
    for class_img in train_class_imgs + val_class_imgs:
        logger.log_images(
            class_img[1] + "_" + str(class_img[2]), get_slice(data, data_info, class_img[1], class_img[2]), cm="gray",
        )
        logger.log_images(
            class_img[1] + "_" + str(class_img[2]) + "_true_class", class_img[0],
        )

# Training loop
for i in range(5000):

    # Get random training batch with augmentation
    # This is the bottle-neck for training and could be done more efficient on the GPU...
    [batch, labels] = get_random_batch(
        data,
        train_coordinates,
        IM_SIZE,
        BATCH_SIZE,
        random_flip=True,
        random_stretch=0.2,
        random_rot_xy=180,
        random_rot_z=15,
    )

    # Format data to torch-variable
    batch = Variable(torch.Tensor(batch).float())
    labels = Variable(torch.Tensor(labels).long())

    # Transfer data to gpu
    if USE_GPU and torch.cuda.is_available():
        batch = batch.cuda()
        labels = labels.cuda()

    # Set network to training phase
    network.train()

    # Run the samples through the network
    output = network(batch)

    # Compute loss
    loss = cross_entropy(torch.squeeze(output), labels)

    # Do back-propagation to get gradients of weights w.r.t. loss
    loss.backward()

    # Ask the optimizer to adjust the parameters in the direction of lower loss
    optimizer.step()

    # Every 10th iteration - print training loss
    if i % 10 == 0:
        network.eval()

        # Log to training loss/acc
        print("Iteration:", i, "Training loss:", utils.var_to_np(loss))
        if LOG_TENSORBOARD:
            logger.log_scalar("training_loss", utils.var_to_np(loss), i)
        for k, v in utils.compute_accuracy(torch.argmax(output, 1), labels).items():
            if LOG_TENSORBOARD:
                logger.log_scalar("training_" + k, v, i)
            print(" -", k, v, "%")

    # every 100th iteration
    if i % 100 == 0 and LOG_TENSORBOARD:
        network.eval()

        # Output predicted train/validation class/probability images
        for class_img in train_class_imgs + val_class_imgs:

            slice = class_img[1]
            slice_no = class_img[2]

            class_img = utils.interpret(
                network.classify, data, data_info, slice, slice_no, IM_SIZE, 16, return_full_size=True, use_gpu=USE_GPU,
            )
            logger.log_images(slice + "_" + str(slice_no) + "_pred_class", class_img[0], step=i)

            class_img = utils.interpret(
                network, data, data_info, slice, slice_no, IM_SIZE, 16, return_full_size=True, use_gpu=USE_GPU,
            )
            logger.log_images(slice + "_" + str(slice_no) + "_pred_prob", class_img[0], i)

        # Store trained network
        torch.save(network.state_dict(), join(ROOT_PATH, "saved_model.pt"))
