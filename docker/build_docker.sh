#!/usr/bin/env bash

image_name=lcas.lincoln.ac.uk/lcastor/lcastor_perception

arch="$(dpkg --print-architecture)"

docker build --build-arg UID=$(id -u) --build-arg GID=$(id -g) --build-arg=$arch -t ${image_name} $(dirname "$0")/
