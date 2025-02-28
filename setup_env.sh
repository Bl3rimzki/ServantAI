#!/bin/bash
set -e

if [[ "$1" == "workstation" ]]; then
    docker build -f docker/Dockerfile.workstation -t servantai:workstation .
elif [[ "$1" == "jetson" ]]; then
    docker build -f docker/Dockerfile.jetson -t servantai:jetson .
else
    echo "Usage: $0 {workstation|jetson}"
    exit 1
fi