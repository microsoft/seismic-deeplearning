This Docker image allows the user to run the notebooks in this repository on any Unix based operating system without having to setup the environment or install anything other than the Docker engine. We recommend using [Azure Data Science Virtual Machine (DSVM) for Linux (Ubuntu)](https://docs.microsoft.com/en-us/azure/machine-learning/data-science-virtual-machine/dsvm-ubuntu-intro) as outlined [here](../README.md#compute-environment). For instructions on how to install the Docker engine, click [here](https://www.docker.com/get-started). 

# Build the Docker image:

In the `docker` directory, run the following command to build the Docker image and tag it as `seismic-deeplearning`: 

```bash
sudo docker build -t seismic-deeplearning . 
```
This process will take a few minutes to complete. 

# Run the Docker image:
Once the Docker image is built, you can run it anytime using the following command:
```bash
sudo docker run --rm -it -p 9000:9000 -p 9001:9001 --gpus=all --shm-size 11G seismic-deeplearning
```
The command above will run a [JupyterLab](https://jupyterlab.readthedocs.io/en/stable/) instance that you can access by clicking on the link in your terminal. You can then navigate to the notebook or script that you would like to run.

We recommend using [Google Chrome](https://www.google.com/chrome/) web browser for any visualizations shown in the notebook.

You can alternatively use [Jupyter](https://jupyter.org/) notebook instead of Jupyter Lab by changing the last line in the Dockerfile from
```bash
jupyter lab --allow-root --ip 0.0.0.0 --port 9000
```  
to
```bash
jupyter notebook --allow-root --ip 0.0.0.0 --port 9000
```
and rebuilding the Docker image.

# Run TensorBoard:
To run Tensorboard to visualize the logged metrics and results, open a terminal in Jupyter Lab, navigate to the parent of the `output` directory of your model, and run the following command: 
```bash 
tensorboard --logdir output/ --port 9001 --bind_all
```
Make sure your VM has the port 9001 allowed in the networking rules, and then you can open TensorBoard by navigating to `http://<vm_ip_address>:9001/` on your browser where `<vm_ip_address>` is your public VM IP address (or private VM IP address if you are using a VPN).

# Experimental

We also offer the ability so use a semantic segmentation [HRNet](https://github.com/HRNet/HRNet-Semantic-Segmentation) model with the repository from 
[Microsoft Research](https://www.microsoft.com/en-us/research/). Its use is currently experimental. 

## Download the HRNet model: 

To run the [`Dutch_F3_patch_model_training_and_evaluation.ipynb`](https://github.com/microsoft/seismic-deeplearning/blob/master/examples/interpretation/notebooks/Dutch_F3_patch_model_training_and_evaluation.ipynb), you will need to manually download the [HRNet-W48-C](https://1drv.ms/u/s!Aus8VCZ_C_33dKvqI6pBZlifgJk) pretrained model. You can follow the instructions [here](../README.md#pretrained-models). 

If you are using an Azure Virtual Machine to run this code, you can download the HRNet model to your local machine, and then copy it to your Azure VM through the command below. Please make sure you update the `<azureuser>` and `<azurehost>` feilds.
```bash
scp hrnetv2_w48_imagenet_pretrained.pth <azureuser>@<azurehost>:/home/<azureuser>/seismic-deeplearning/docker/hrnetv2_w48_imagenet_pretrained.pth
```

## Run the Docker image:

Once you have the model downloaded (ideally under the `docker` directory), you can proceed to build the Docker image: go to the [Build the Docker image](#build-the-docker-image) section above to do so.

Once the Docker image is built, you can run it anytime using the following command:
```bash
sudo docker run --rm -it -p 9000:9000 -p 9001:9001 --gpus=all --shm-size 11G --mount type=bind,source=$PWD/hrnetv2_w48_imagenet_pretrained.pth,target=/home/username/seismic-deeplearning/docker/hrnetv2_w48_imagenet_pretrained.pth seismic-deeplearning
```

If you have saved the pretrained model in a different directory, make sure you replace `$PWD/hrnetv2_w48_imagenet_pretrained.pth` with the **absolute** path to the pretrained HRNet model. 
The command above will run a Jupyter Lab instance that you can access by clicking on the link in your terminal. You can then navigate to the notebook or script that you would like to run.

