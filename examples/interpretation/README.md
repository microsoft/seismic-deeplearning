The folder contains notebook examples illustrating the use of segmentation algorithms on openly available datasets. Make sure you have followed the [set up instructions](../README.md) before running these examples. We provide the following notebook examples 

* [Dutch F3 dataset](notebooks/F3_block_training_and_evaluation_local.ipynb): This notebook illustrates section and patch based segmentation approaches on the [Dutch F3](https://terranubis.com/datainfo/Netherlands-Offshore-F3-Block-Complete) open dataset. This notebook uses denconvolution based segmentation algorithm on 2D patches. The notebook will guide you through visualization of the input volume, setting up model training and evaluation. 


* [Penobscot dataset](notebooks/HRNet_Penobscot_demo_notebook.ipynb): 
In this notebook, we demonstrate how to train an [HRNet](https://github.com/HRNet/HRNet-Semantic-Segmentation) model for facies prediction using [Penobscot](https://terranubis.com/datainfo/Penobscot) dataset. The Penobscot 3D seismic dataset was acquired in the Scotian shelf, offshore Nova Scotia, Canada. This notebook illustrates the use of HRNet based segmentation algorithm on the dataset. Details of HRNet based model can be found [here](https://arxiv.org/abs/1904.04514)

