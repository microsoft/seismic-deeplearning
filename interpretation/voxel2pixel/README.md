# CNN for ASI
Code for the paper:<br />
**Convolutional Neural Networks for Automated Seismic Interpretation**,<br />
A. U. Waldeland, A. C. Jensen, L. Gelius and A. H. S. Solberg <br />
[*The Leading Edge, July 2018*](https://library.seg.org/doi/abs/10.1190/tle37070529.1)

EAGE E-lecture: [*Seismic interpretation with deep learning*](https://www.youtube.com/watch?v=lm85Ap4OstM) (YouTube)

This repository contains python/pytorch code for applying Convolutional Neural Networks (CNN) on seismic data. The input is a segy-file containing post-stack seismic amplitude data. The training labels are given as images of the training slices, colored to indicate the classes. 

### Setup to get started
CNNs requires time-demanding computations, consider using a computer with a fast NVIDIA GPU with at least 6GB ram.
- Make sure you have all the required python packages installed: <br/> 
-- pytorch <br/> 
-- tensorflow (GPU enabled version is not needed - we only use this package for plotting) <br />
-- tensorflow-tensorboard <br />
-- numpy<br />
-- scipy<br />
-- segyio<br />
-- matplotlib<br />
- Clone this repository<br />
- Download the demo data set [here](https://drive.google.com/drive/folders/0B7brcf-eGK8CbGhBdmZoUnhiTWs). This is the F3 Netherlands dataset - originally made available by OpenDTect - and available via the [MalenoV Project](https://github.com/bolgebrygg/MalenoV).
- Locate the '.segy'-file, rename it to 'data.segy' and put it in the 'F3'-folder.
 
### How to use tensorboard (for visualization)
- Open a terminal<br />
- cd to the code-folder<br />
- run: tensorboard --logdir='log'<br />
- Open a web-browser and go to localhost:6006<br />
More information can be found [here](https://www.tensorflow.org/get_started/summaries_and_tensorboard#launching_tensorboard).
  
### Usage
- train.py - train the CNN<br />
- test.py - Example of how the trained CNN can be applied to predict salt in a slice or the full cube. In addition it shows how learned attributes can be extracted.<br />

### Files
In addition, it may be useful to have a look on these files<br/>
- texture_net.py - this is where the network is defined <br/>
- batch.py - provide functionality to generate training batches with random augmentation <br/>
- data.py - load/save data sets with segy-format and labeled slices as images <br/>
- tb_logger.py - connects to the tensorboard functionality <br/>
- utils.py - some help functions <br/>
- test_parallel.py - An implemenation of test.py supporting multi-gpu prediction (thanks to Max Kaznady).<br />

### Using a different data set and custom training labels
If you want to use a different data set, do the following:
- Make a new folder where you place the segy-file
- Make a folder for the training labels
- Save images of the slices you want to train on as 'SLICETYPE_SLICENO.png' (or jpg), where SLICETYPE is either 'inline', 'crossline', or 'timeslice' and SLICENO is the slice number.
- Draw the classes on top of the seismic data, using a simple image editing program with the class colors. Currently up to six classes are supported, indicated by the colors: red, blue, green, cyan, magenta and yellow.

# Contact
Email: anders.u.waldeland@gmail.com
