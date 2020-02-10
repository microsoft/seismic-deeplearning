# SEG-Y Data Utilities

SEG-Y files can have a lot of variability which makes it difficult to inferer the geometry when converting to npy. The segyio module attempts to do so but fails if there are missing traces in the file (which happens regularly). This utility reads traces using segyio with the inferencing turned off to avoid data loading errors and it uses its own logic to place traces into a numpy array. If traces are missing, the values of the npy array in that location are set to zero

## convert_segy.py script

THe `convert_segy.py` script can work with SEG-Y files and output data on  local disk. This script will process segy files regardless of their structure and output npy files for use in training/scoring. In addition to the npy files, it will write a json file that includes the standard deviation and mean of the original data. The script can additionally use that to normalize and clip that data if indicated in the command line parameters

The resulting npy files will use the following naming convention:

```<prefix>_<inline id>_<xline id>_<depth>.npy```

These inline and xline ids are the upper left location of the data contained in the file and can be later used to identify where the npy file is located in the segy data.

To use this script, follow these examples:

1) Convert a SEG-Y file to a single npy file of the same dimensions:

    ```
    python ./convert_segy.py --prefix {PREFIX} --input_file {SEGYFILE} --output_dir .
    ```

2) Convert a SEG-Y file to a single npy file of the same dimensions, clip and normalize the results:

    ```
    python ./convert_segy.py --prefix {PREFIX} --input_file {SEGYFILE} --output_dir . --normalize
    ```

2) Convert a SEG-Y file to a single npy file of the same dimensions, clip but do not normalize the results:

    ```
    python ./convert_segy.py --prefix {PREFIX} --input_file {SEGYFILE} --output_dir . --clip
    ```

2) Split a single SEG-Y file into a set of npy files, each npy array with dimension (100,100,100)

    ```
    python ./convert_segy.py --prefix {PREFIX} --input_file {SEGYFILE} --output_dir . --cube_size 100
    ```

There are several additional command line arguments that may be needed to load specific segy files (i.e. the byte locations for data headers may be different). Run --help to review the additional commands if needed.

## Testing

Run pytest from the segyconverter directory to run the local unit tests