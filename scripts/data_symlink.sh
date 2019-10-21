#!/bin/bash

# Azure VMs lose mounts after restart - this symlinks the data folder from user's
# home directory after VM restart

sudo chown -R maxkaz /mnt
sudo chgrp -R maxkaz /mnt
ln -s ~/dutchf3 /mnt
