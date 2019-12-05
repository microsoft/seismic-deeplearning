#!/bin/bash
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

echo Using "$1" as the download directory

if [ ! -d "$1" ]
then
    echo "Directory does not exist - creating..."
    mkdir -p "$1"
fi

full_path=$1/$2

echo "Downloading to ${full_path}"

wget \
'https://optgaw.dm.files.1drv.com/y4mYRZ1J12ATHBWxSwZJTycIFbgT88SoWegSN3NXEOJFTJWBLDZ3nyCbj-sDvmGobL2CAGzhPobs5gMt3466nKbATNs9toc5N569Z5xNicNUABQm0MVucO7Vi7cjP__n2MFL5qDZyL4cOx6VgoNjpb9lglVRqoTVfVHdJ3sM7qO-9sAODNzgmKrCrU7uHvB54YtsKLr51Qi6BlDn94DalmEJQ/hrnetv2_w48_imagenet_pretrained.pth?download&psid=1' \
--output-document "${full_path}"
