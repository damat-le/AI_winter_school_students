# Data transfer

To trasfer data from local machine to Leonardo do not use login node.

Use datamovers or GridFTP to transfer data.

## Data movers

* no interactive access
* only limited set of commands available: scp, rsync, sftp, wget, curl, rclone, s3 and aws s3
* no cpu time limit
* alias: `data.leonardo.cineca.it/`
* On datamovers, aliases like $HOME, $WORK etc are not available. You need to use the full path.
* you cannot use your local .ssh/config file to connect

### FROM/TO local TO/FROM cluster
* rsync (see slide 42)
* scp (see slide 43)

### BETWEEN 2 clusters
* see 

### DOWNLOAD from internet
* login node
* use the `lrd_all_serial` partition (see slide 35)
* srun job that downloads the data using wget or curl
* 4h time limit


## GridFTP

* ...

