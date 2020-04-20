This Docker image allows the user to run the notebooks in this repository on any operating system without having to setup the environment or install anything other than the Docker engine. For instructions on how to install the Docker engine, click [here](https://www.docker.com/get-started). 

# Download the HRNet model: 

To run the [`HRNet_Penobscot_demo_notebook.ipynb`](https://github.com/microsoft/seismic-deeplearning/blob/master/examples/interpretation/notebooks/HRNet_Penobscot_demo_notebook.ipynb), you will need to manually download the [HRNet-W48-C](https://1drv.ms/u/s!Aus8VCZ_C_33dKvqI6pBZlifgJk) pretrained model. You can follow the instructions [here.](https://github.com/microsoft/seismic-deeplearning#hrnet). 

If you are using an Azure Virtual Machine to run this code, you can download the model to your local machine, and then copy it to your Azure VM through the command below. Please make sure you update the `<azureuser>` and `<azurehost>` feilds.
```bash
scp hrnetv2_w48_imagenet_pretrained.pth <azureuser>@<azurehost>:/home/<azureuser>/seismic-deeplearning/docker/hrnetv2_w48_imagenet_pretrained.pth
```
Once you have the model downloaded (ideally under the `docker` directory), you can process to build the Docker image. 

# Build the Docker image:

In the `docker` directory, run the following command to build the Docker image and tag it as `seismic-deeplearning`: 

```bash
sudo docker build -t seismic-deeplearning . 
```
This process will take a few minutes to complete. 

# Run the Docker image:
Once the Docker image is built, you can run it anytime using the following command:
```bash
sudo docker run --rm -it -p 9000:9000 -p 9001:9001 --gpus=all --shm-size 11G --mount type=bind,source=$PWD/hrnetv2_w48_imagenet_pretrained.pth,target=/home/username/models/hrnetv2_w48_imagenet_pretrained.pth seismic-deeplearning
```
If you have saved the pretrained model in a different directory, make sure you replace `$PWD/hrnetv2_w48_imagenet_pretrained.pth` with the **absolute** path to the pretrained HRNet model. The command above will run a jupyter notebook server that you can access by clicking on the link in your terminal. You can then navigate to the notebook that you would like to run.  

