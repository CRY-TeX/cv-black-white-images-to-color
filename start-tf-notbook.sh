#!/usr/bin/bash

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

image_name="tf-notebook"

docker build -t $image_name $SCRIPT_DIR
docker run --gpus all -d -p 8888:8888 --name $image_name $image_name
docker logs -f $image_name
