# Voxel to Pixel approach to Seismic Interpretation 

The code which is used in this approach is greatly described in the paper
<br />
**Convolutional Neural Networks for Automated Seismic Interpretation**,<br />
A. U. Waldeland, A. C. Jensen, L. Gelius and A. H. S. Solberg <br />
[*The Leading Edge, July 2018*](https://library.seg.org/doi/abs/10.1190/tle37070529.1)

There is also an 
EAGE E-lecture which you can watch: [*Seismic interpretation with deep learning*](https://www.youtube.com/watch?v=lm85Ap4OstM) (YouTube)

### Setup to get started
- make sure you follow `README.md` file in root of repo to install all the proper dependencies.
- downgrade TensorFlow and pyTorch's CUDA:
    - downgrade TensorFlow by running `pip install tensorflow-gpu==1.14` 
    - make sure pyTorch uses downgraded CUDA `pip install torch==1.3.1+cu92 torchvision==0.4.2+cu92 -f https://download.pytorch.org/whl/torch_stable.html`
- download the data by running `contrib/scrips/get_F3_voxel.sh` from the `contrib` folder of this repo.
This will download the training and validation labels/masks.
- to get the main input dataset which is the [Dutch F3 dataset](https://terranubis.com/datainfo/Netherlands-Offshore-F3-Block-Complete), 
navigate to [MalenoV](https://github.com/bolgebrygg/MalenoV) project website and follow the links (which will lead to 
[this](https://drive.google.com/drive/folders/0B7brcf-eGK8CbGhBdmZoUnhiTWs) download). Save this file as 
`interpretation/voxel2pixel/F3/data.segy`

If you want to revert downgraded packages, just run `conda env update -f environment/anaconda/local/environment.yml` from the root folder of the repo.

### Monitoring progress with TensorBoard
- from the `voxel2pixel` directory, run `tensorboard --logdir='log'` (all runtime logging information is
written to the `log` folder <br />
- open a web-browser and go to localhost:6006<br />
More information can be found [here](https://www.tensorflow.org/get_started/summaries_and_tensorboard#launching_tensorboard).
  
### Usage
- `python train.py` will train the CNN and produce a model after a few hours on a decent gaming GPU
with at least 6GB of onboard memory<br />
- `python test_parallel.py` - Example of how the trained CNN can be applied to predict salt in a slice or 
the full cube in distributed fashion on a single multi-GPU machine (single GPU mode is also supported). 
In addition it shows how learned attributes can be extracted.<br />

### Files
In addition, it may be useful to have a look on these files<br/>
- texture_net.py - this is where the network is defined <br/>
- batch.py - provides functionality to generate training batches with random augmentation <br/>
- data.py - load/save data sets with segy-format and labeled slices as images <br/>
- tb_logger.py - connects to the tensorboard functionality <br/>
- utils.py - some help functions <br/>
- test_parallel.py - multi-GPU prediction script for scoring<br />

### Using a different data set and custom training labels
If you want to use a different data set, do the following:
- Make a new folder where you place the segy-file
- Make a folder for the training labels
- Save images of the slices you want to train on as 'SLICETYPE_SLICENO.png' (or jpg), where SLICETYPE is either 'inline', 'crossline', or 'timeslice' and SLICENO is the slice number.
- Draw the classes on top of the seismic data, using a simple image editing program with the class colors. Currently up to six classes are supported, indicated by the colors: red, blue, green, cyan, magenta and yellow.

