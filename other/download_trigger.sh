# This script illustrates how it works to download a dataset (IRIS in this case) on the Leonardo cluster using the correct service for data transfer (see https://docs.hpc.cineca.it/hpc/hpc_data_storage.html#data-transfer).


# 1. Download a dataset directly on the SCRATCH dir on the cluster using curl

ssh a08trc0e@data.leonardo.cineca.it \
  curl "https://archive.ics.uci.edu/static/public/53/iris.zip" \
  --output /leonardo_scratch/large/usertrain/a08trc0e/iris.zip


# 2. Upload a dataset from your LOCAL machine to the cluster using scp
# This assumes you have the file iris.zip in your current local directory.

scp iris.zip a08trc0e@data.leonardo.cineca.it:/leonardo_scratch/large/usertrain/a08trc0e/