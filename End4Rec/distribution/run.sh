#!/bin/bash

export DEVICE_NUM=4
export RANK_SIZE=$DEVICE_NUM

echo "Launching distributed training on $DEVICE_NUM devices."

echo "Initializing distributed cluster..."
mpirun -n $DEVICE_NUM python launch_cluster.py --device_target GPU --device_num $DEVICE_NUM --config_file config.json
if [ $? -ne 0 ]; then
    echo "Error: Distributed cluster launch failed."
    exit 1
fi

sleep 5

echo "Starting distributed training..."
mpirun -n $DEVICE_NUM python main.py config.json
if [ $? -ne 0 ]; then
    echo "Error: Training failed."
    exit 1
fi

sleep 5

echo "Merging model checkpoints..."
python merge_checkpoints.py --ckpt_dir ./ckpt/ --output_ckpt ./ckpt/merged_ckpt.ckpt
if [ $? -ne 0 ]; then
    echo "Error: Checkpoint merge failed."
    exit 1
fi

echo "Distributed training completed successfully."
