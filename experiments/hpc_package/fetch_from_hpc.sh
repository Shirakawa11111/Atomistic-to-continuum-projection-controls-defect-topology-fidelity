#!/bin/bash
set -e
rsync -avz 'hpc:~/route_A_protocol_robustness/relaxed_*.dump' '/Users/bojingkai/Desktop/Route_A_protocol_robustness/experiments/hpc_package/'
rsync -avz 'hpc:~/route_A_protocol_robustness/relaxed_*.data' '/Users/bojingkai/Desktop/Route_A_protocol_robustness/experiments/hpc_package/'
rsync -avz 'hpc:~/route_A_protocol_robustness/md_*_*.dump' '/Users/bojingkai/Desktop/Route_A_protocol_robustness/experiments/hpc_package/' || true
