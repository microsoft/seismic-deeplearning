#!/bin/bash
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# autoformats all files in the repo to black

# example of using regex -regex ".*\.\(py\|ipynb\|md\|txt\)"
find . -type f -regex ".*\.py" -exec black {} +
