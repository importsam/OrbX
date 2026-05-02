#!/bin/bash

# Log start time
echo "Starting orbit update at $(date)" >> /home/spaceprotocol/orbit_updates.log

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate iscore

# Change to project directory and run first script
cd /home/spaceprotocol/drive/repos/unique_orbits/update_cesium_assets
python main.py

# Log completion
echo "Completed orbit update at $(date)" >> /home/spaceprotocol/orbit_updates.log