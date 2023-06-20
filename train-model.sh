#!/usr/bin/bash

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

image_name="tf-train"

docker build -t $image_name $SCRIPT_DIR
# docker run --rm --gpus all -d -v "$SCRIPT_DIR:/app" --name $image_name $image_name python /app/$1
docker run --rm --gpus all -d -v "$SCRIPT_DIR:/app" --name $image_name $image_name python -u /app/$@
docker logs -f $image_name
