First, make sure that `/mnt/dutch_f3` folder exists and you have write access.

Next, to get the main input dataset which is the [Dutch F3 dataset](https://terranubis.com/datainfo/Netherlands-Offshore-F3-Block-Complete), 
navigate to [MalenoV](https://github.com/bolgebrygg/MalenoV) project website and follow the links (which will lead to 
[this](https://drive.google.com/drive/folders/0B7brcf-eGK8CbGhBdmZoUnhiTWs) download). Save this file as 
`/mnt/dutch_f3/data.segy`

To download the train and validation masks, from the root of the repo, run
```bash
./scripts/get_F3_voxel.sh /mnt/dutch_f3
```

This will also download train and validation masks to the same location as data.segy.

That's it!
