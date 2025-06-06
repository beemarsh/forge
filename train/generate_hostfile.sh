#!/bin/bash
GPUS_PER_NODE=8

current_directory=$(pwd)
host_path="${current_directory}/hostfiles"


if [ -d "$host_path" ]; then
    echo "hostfiles folder already exists at ${new_folder_path}. Skipping creation."
else
    mkdir "$host_path"
fi

# need to add the current slurm jobid to hostfile name so that we don't add to previous hostfile
hostfile="${host_path}/hosts_$SLURM_JOBID"
# be extra sure we aren't appending to a previous hostfile
rm $hostfile &> /dev/null
# loop over the node names
for i in `scontrol show hostnames $SLURM_NODELIST`
do
    # add a line to the hostfile
    echo $i slots=$GPUS_PER_NODE >>$hostfile
done
