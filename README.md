# DeepSeismic

## Interpretation

### Setting up Environment
Run
```bash
conda env create -f DeepSeismic/environment/anaconda/local/environment.yml
```
This will create the appropriate environment to run experiments

Then you will need to install the common packages for interpretation
```bash
conda activate seismic-interpretation
pip install -e DeepSeismic/deepseismic_interpretation
```

Then you will also need to pull computer vision contrib
```bash
git clone https://aicat-ongip@dev.azure.com/aicat-ongip/AI%20CAT%20OnG%20IP/_git/ComputerVision_fork
pip install -e ComputerVision_fork/contrib
```

Both repos are installed in developer mode with the -e flag. This means that to update simply go to the folder and pull the appropriate commit or branch

### Data
The scripts expect the data to be contained in /mnt/alaudah

## Contributing

This project welcomes contributions and suggestions. Most contributions require you to agree to a Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
