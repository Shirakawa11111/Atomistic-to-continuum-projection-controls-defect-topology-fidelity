#!/bin/bash
# Sync package to HPC (gated — run manually after approval)
set -e
rsync -avz --progress '/Users/bojingkai/Desktop/Route_A_protocol_robustness/experiments/hpc_package/' hpc:~/route_A_protocol_robustness/
echo "synced to hpc:~/route_A_protocol_robustness/"
