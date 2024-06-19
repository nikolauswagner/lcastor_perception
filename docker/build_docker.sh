#!/usr/bin/env bash

image_name=lcas.lincoln.ac.uk/lcastor/lcastor_perception

weights=${1:-ycb}

docker build --build-arg WEIGHTS="coco"\
             -t ${image_name} $(dirname "$0")/
