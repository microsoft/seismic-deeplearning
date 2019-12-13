#!/bin/bash
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#
# Example:
# download_hrnet.sh /data/models hrnet.pth
#

echo Using "$1" as the download directory

if [ ! -d "$1" ]
then
    echo "Directory does not exist - creating..."
    mkdir -p "$1"
fi

full_path=$1/$2

echo "Downloading to ${full_path}"

wget --header 'Host: optgaw.dm.files.1drv.com' \
      --user-agent 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:70.0) Gecko/20100101 Firefox/70.0' \
      --header 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8' \
      --header 'Accept-Language: en-GB,en;q=0.5' \
      --referer 'https://onedrive.live.com/' \
      --header 'Upgrade-Insecure-Requests: 1' 'https://optgaw.dm.files.1drv.com/y4m14W1OEuoniQMCT4m64UV8CSQT-dFe2ZRhU0LAZSal80V4phgVIlTYxI2tUi6BPVOy7l5rK8MKpZNywVvtz-NKL2ZWq-UYRL6MAjbLgdFA6zyW8RRrKBe_FcqcWr4YTXeJ18xfVqco6CdGZHFfORBE6EtFxEIrHWNjM032dWZLdqZ0eXd7RZTrHs1KKYa92zcs0Rj91CAyIK4hIaOomzEWA/hrnetv2_w48_imagenet_pretrained.pth?download&psid=1' \
      --output-document ${full_path}