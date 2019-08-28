# Compatability Imports
from __future__ import print_function

import os

# set default number of GPUs which are discoverable
N_GPU = 8
DEVICE_IDS = list(range(N_GPU))
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(x) for x in DEVICE_IDS])

# static parameters
RESOLUTION = 1
# these match how the model is trained
N_CLASSES = 2
IM_SIZE = 65

import random
import argparse
import json

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist

if torch.cuda.is_available():
    device_str = os.environ["CUDA_VISIBLE_DEVICES"]
    device = torch.device("cuda:" + device_str)
else:
    raise Exception("No GPU detected for parallel scoring!")

# ability to perform multiprocessing
import multiprocessing

from os.path import join
from data import readSEGY, get_slice
from texture_net import TextureNet
import itertools
import numpy as np
import tb_logger
from data import writeSEGY

# graphical progress bar
from tqdm import tqdm


class ModelWrapper(nn.Module):
    """
    Wrap TextureNet for (Distributed)DataParallel to invoke classify method
    """

    def __init__(self, texture_model):
        super(ModelWrapper, self).__init__()
        self.texture_model = texture_model

    def forward(self, input):
        return self.texture_model.classify(input)


class MyDataset(Dataset):
    def __init__(self, data, window, coord_list):

        # main array
        self.data = data
        self.coord_list = coord_list
        self.window = window
        self.len = len(coord_list)

    def __getitem__(self, index):

        # TODO: can we specify a pixel mathematically by index?
        pixel = self.coord_list[index]
        x, y, z = pixel
        # TODO: current bottleneck - can we slice out voxels any faster
        small_cube = self.data[
            x - self.window : x + self.window + 1,
            y - self.window : y + self.window + 1,
            z - self.window : z + self.window + 1,
        ]

        return small_cube[np.newaxis, :, :, :], pixel

    def __len__(self):
        return self.len


def main_worker(gpu, ngpus_per_node, args):
    """
    Main worker function, given the gpu parameter and how many GPUs there are per node
    it can figure out its rank

    :param gpu: rank of the process if gpu >= ngpus_per_node, otherwise just gpu ID which worker will run on.
    :param ngpus_per_node: total number of GPU available on this node.
    :param args: various arguments for the code in the worker.
    :return: nothing
   """

    print("I got GPU", gpu)

    args.rank = gpu

    # loop around in round-robin fashion if we want to run multiple processes per GPU
    args.gpu = gpu % ngpus_per_node

    # initialize the distributed process and join the group
    print(
        "setting rank",
        args.rank,
        "world size",
        args.world_size,
        args.dist_backend,
        args.dist_url,
    )
    dist.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )

    # set default GPU device for this worker
    torch.cuda.set_device(args.gpu)
    # set up device for the rest of the code
    device = torch.device("cuda:" + str(args.gpu))

    # Load trained model (run train.py to create trained
    network = TextureNet(n_classes=N_CLASSES)
    model_state_dict = torch.load(
        join(args.data, "saved_model.pt"), map_location=device
    )
    network.load_state_dict(model_state_dict)
    network.eval()
    network.cuda(args.gpu)

    # set the scoring wrapper also to eval mode
    model = ModelWrapper(network)
    model.eval()
    model.cuda(args.gpu)

    # When using a single GPU per process and per
    # DistributedDataParallel, we need to divide the batch size
    # ourselves based on the total number of GPUs we have.
    # Min batch size is 1
    args.batch_size = max(int(args.batch_size / ngpus_per_node), 1)
    # obsolete: number of data loading workers - this is only used when reading from disk, which we're not
    # args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)

    # wrap the model for distributed use - for scoring this is not needed
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    # set to benchmark mode because we're running the same workload multiple times
    cudnn.benchmark = True

    # Read 3D cube
    # NOTE: we cannot pass this data manually as serialization of data into each python process is costly,
    # so each worker has to load the data on its own.
    data, data_info = readSEGY(join(args.data, "data.segy"))

    # Get half window size
    window = IM_SIZE // 2

    # reduce data size for debugging
    if args.debug:
        data = data[0 : 3 * window]

    # generate full list of coordinates
    # memory footprint of this isn't large yet, so not need to wrap as a generator
    nx, ny, nz = data.shape
    x_list = range(window, nx - window)
    y_list = range(window, ny - window)
    z_list = range(window, nz - window)

    print("-- generating coord list --")
    # TODO: is there any way to use a generator with pyTorch data loader?
    coord_list = list(itertools.product(x_list, y_list, z_list))

    # we need to map the data manually to each rank - DistributedDataParallel doesn't do this at score time
    print("take a subset of coord_list by chunk")
    coord_list = list(
        np.array_split(np.array(coord_list), args.world_size)[args.rank]
    )
    coord_list = [tuple(x) for x in coord_list]

    # we only score first batch in debug mode
    if args.debug:
        coord_list = coord_list[0 : args.batch_size]

    # prepare the data
    print("setup dataset")
    # TODO: RuntimeError: cannot pin 'torch.cuda.FloatTensor' only dense CPU tensors can be pinned
    data_torch = torch.cuda.FloatTensor(data).cuda(args.gpu, non_blocking=True)
    dataset = MyDataset(data_torch, window, coord_list)

    # not sampling like in training
    # datasampler = DistributedSampler(dataset)
    # just set some default epoch
    # datasampler.set_epoch(1)

    # we use 0 workers because we're reading from memory
    print("setting up loader")
    my_loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        sampler=None
        # sampler=datasampler
    )

    print("running loop")

    pixels_x = []
    pixels_y = []
    pixels_z = []
    predictions = []

    # Loop through center pixels in output cube
    with torch.no_grad():
        print("no grad")
        for (chunk, pixel) in tqdm(my_loader):
            input = chunk.cuda(args.gpu, non_blocking=True)
            output = model(input)
            # save and deal with it later on CPU
            # we want to make sure order is preserved
            pixels_x += pixel[0].tolist()
            pixels_y += pixel[1].tolist()
            pixels_z += pixel[2].tolist()
            predictions += output.tolist()
            # just score a single batch in debug mode
            if args.debug:
                break

    # TODO: legacy Queue Manager code from multiprocessing which we left here for illustration purposes
    # result_queue.append([deepcopy(coord_list), deepcopy(predictions)])
    # result_queue.append([coord_list, predictions])
    # transform pixels into x, y, z list format
    with open("results_{}.json".format(args.rank), "w") as f:
        json.dump(
            {
                "pixels_x": pixels_x,
                "pixels_y": pixels_y,
                "pixels_z": pixels_z,
                "preds": [int(x[0][0][0][0]) for x in predictions],
            },
            f,
        )

    # TODO: we cannot use pickle to dump from multiprocess - processes lock up
    # with open("result_predictions_{}.pkl".format(args.rank), "wb") as f:
    #    print ("dumping predictions pickle file")
    #    pickle.dump(predictions, f)


parser = argparse.ArgumentParser(description="Seismic Distributed Scoring")
parser.add_argument(
    "-d", "--data", default="F3", type=str, help="default dataset folder name"
)
parser.add_argument(
    "-s",
    "--slice",
    default="inline",
    type=str,
    choices=["inline", "crossline", "timeslice", "full"],
    help="slice type which we want to score on",
)
parser.add_argument(
    "-n",
    "--slice-num",
    default=339,
    type=int,
    help="slice number which we want to score",
)
parser.add_argument(
    "-b",
    "--batch-size",
    default=2 ** 15,
    type=int,
    help="batch size which we use for scoring",
)
parser.add_argument(
    "-p",
    "--n-proc-per-gpu",
    default=1,
    type=int,
    help="number of multiple processes to run per each GPU",
)
parser.add_argument(
    "--dist-url",
    default="tcp://127.0.0.1:12345",
    type=str,
    help="url used to set up distributed training",
)
parser.add_argument(
    "--dist-backend", default="nccl", type=str, help="distributed backend"
)
parser.add_argument(
    "--seed", default=0, type=int, help="default random number seed"
)
parser.add_argument(
    "--debug",
    action="store_true",
    help="debug flag - if on we will only process one batch",
)


def main():

    # use distributed scoring+
    if RESOLUTION != 1:
        raise Exception("Currently we only support pixel-level scoring")

    args = parser.parse_args()

    args.gpu = None
    args.rank = 0

    # world size is the total number of processes we want to run across all nodes and GPUs
    args.world_size = N_GPU * args.n_proc_per_gpu

    if args.debug:
        args.batch_size = 4

    # fix away any kind of randomness - although for scoring it should not matter
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.deterministic = True

    print("RESOLUTION {}".format(RESOLUTION))

    ##########################################################################
    print("-- scoring on GPU --")

    ngpus_per_node = torch.cuda.device_count()
    print("nGPUs per node", ngpus_per_node)

    """
    First, read this: https://thelaziestprogrammer.com/python/a-multiprocessing-pool-pickle
    
    OK, so there are a few ways in which we can spawn a running process with pyTorch:
    1) Default mp.spawn should work just fine but won't let us access internals
    2) So we copied out the code from mp.spawn below to control how processes get created
    3) One could spawn their own processes but that would not be thread-safe with CUDA, line
    "mp = multiprocessing.get_context('spawn')" guarantees we use the proper pyTorch context
    
    Input data serialization is too costly, in general so is output data serialization as noted here:
    https://docs.python.org/3/library/multiprocessing.html
    
    Feeding data into each process is too costly, so each process loads its own data.
    
    For deserialization we could try and fail using:
    1) Multiprocessing queue manager
    manager = Manager()
    return_dict = manager.dict()
    OR    
    result_queue = multiprocessing.Queue()
    CALLING
    with Manager() as manager:
        results_list = manager.list()
        mp.spawn(main_worker, nprocs=args.world_size, args=(ngpus_per_node, results_list/dict/queue, args))
        results = deepcopy(results_list)
    2) pickling results to disc.
    
    Turns out that for the reasons mentioned in the first article both approaches are too costly.
    
    The only reasonable way to deserialize data from a Python process is to write it to text, in which case
    writing to JSON is a saner approach: https://www.datacamp.com/community/tutorials/pickle-python-tutorial
    """

    # invoke processes manually suppressing error queue
    mp = multiprocessing.get_context("spawn")
    # error_queues = []
    processes = []
    for i in range(args.world_size):
        # error_queue = mp.SimpleQueue()
        process = mp.Process(
            target=main_worker, args=(i, ngpus_per_node, args), daemon=False
        )
        process.start()
        # error_queues.append(error_queue)
        processes.append(process)

    # block on wait
    for process in processes:
        process.join()

    print("-- aggregating results --")

    # Read 3D cube
    data, data_info = readSEGY(join(args.data, "data.segy"))

    # Log to tensorboard - input slice
    logger = tb_logger.TBLogger("log", "Test")
    logger.log_images(
        args.slice + "_" + str(args.slice_num),
        get_slice(data, data_info, args.slice, args.slice_num),
        cm="gray",
    )

    x_coords = []
    y_coords = []
    z_coords = []
    predictions = []
    for i in range(args.world_size):
        with open("results_{}.json".format(i), "r") as f:
            dict = json.load(f)

        x_coords += dict["pixels_x"]
        y_coords += dict["pixels_y"]
        z_coords += dict["pixels_z"]
        predictions += dict["preds"]

    """
    So because of Python's GIL having multiple workers write to the same array is not efficient - basically
    the only way we can have shared memory is with threading but thanks to GIL only one thread can execute at a time, 
    so we end up with the overhead of managing multiple threads when writes happen sequentially.
    
    A much faster alternative is to just invoke underlying compiled code (C) through the use of array indexing.
    
    So basically instead of the following:
    
    NUM_CORES = multiprocessing.cpu_count()
    print("Post-processing will run on {} CPU cores on your machine.".format(NUM_CORES))
    
    def worker(classified_cube, coord):
        x, y, z = coord
        ind = new_coord_list.index(coord)
        # print (coord, ind)
        pred_class = predictions[ind]
        classified_cube[x, y, z] = pred_class

    # launch workers in parallel with memory sharing ("threading" backend)
    _ = Parallel(n_jobs=4*NUM_CORES, backend="threading")(
        delayed(worker)(classified_cube, coord) for coord in tqdm(pixels)
    )
    
    We do this:    
    """

    # placeholder for results
    classified_cube = np.zeros(data.shape)
    # store final results
    classified_cube[x_coords, y_coords, z_coords] = predictions

    print("-- writing segy --")
    in_file = join(args.data, "data.segy".format(RESOLUTION))
    out_file = join(args.data, "salt_{}.segy".format(RESOLUTION))
    writeSEGY(out_file, in_file, classified_cube)

    print("-- logging prediction --")
    # log prediction to tensorboard
    logger = tb_logger.TBLogger("log", "Test_scored")
    logger.log_images(
        args.slice + "_" + str(args.slice_num),
        get_slice(classified_cube, data_info, args.slice, args.slice_num),
        cm="binary",
    )


if __name__ == "__main__":
    main()
