### Contrib folder  

Code in this folder has not been tested, and are meant for exploratory work only.

We encourage submissions to the contrib folder, and once they are well-tested, do submit a pull request and work with the repository owners to graduate it to the main DeepSeismic repository.

Thank you.

#### Azure Machine Learning
If you would like to leverage Azure Machine Learning to create a Training Pipeline with this dataset we have guidance on how do so [here](interpretation/deepseismic_interpretation/azureml_pipelines/README.md)

### HRNet model guidance (experimental for now)

#### HRNet ImageNet weights model

To enable training from scratch on seismic data and to achieve the same results as the benchmarks quoted below you will need to download the HRNet model [pretrained](https://github.com/HRNet/HRNet-Image-Classification) on ImageNet. We are specifically using the [HRNet-W48-C](https://1drv.ms/u/s!Aus8VCZ_C_33dKvqI6pBZlifgJk) pre-trained model; other  HRNet variants are also available [here](https://github.com/HRNet/HRNet-Image-Classification) - you can navigate to those from the [main HRNet landing page](https://github.com/HRNet/HRNet-Object-Detection) for object detection.

Unfortunately, the OneDrive location which is used to host the model is using a temporary authentication token, so there is no way for us to script up model download. There are two ways to upload and use the pre-trained HRNet model on DS VM:
- download the model to your local drive using a web browser of your choice and then upload the model to the DS VM using something like `scp`; navigate to Portal and copy DS VM's public IP from the Overview panel of your DS VM (you can search your DS VM by name in the search bar of the Portal) then use `scp local_model_location username@DS_VM_public_IP:./model/save/path` to upload
- alternatively, you can use the same public IP to open remote desktop over SSH to your Linux VM using [X2Go](https://wiki.x2go.org/doku.php/download:start): you can basically open the web browser on your VM this way and download the model to VM's disk
