#!/bin/bash

# Azure VMs lose mounts after restart - this symlinks the data folder from user's
# home directory after VM restart

user=$(whoami)
sudo chown -R $user /mnt
sudo chgrp -R $user /mnt
ln -s ~/dutchf3 /mnt
