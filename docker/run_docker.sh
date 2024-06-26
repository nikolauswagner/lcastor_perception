#!/bin/bash

image_name=lcas.lincoln.ac.uk/lcastor/lcastor_perception

# Start docker container
echo "Starting docker container..."
docker run --network host \
           --privileged \
           -v $(pwd)/../models:/home/lcastor/ros_ws/src/lcastor_perception/models \
           --gpus all \
           --runtime nvidia \
           -e IMG_NAME=${image_name} \
           -e ROS_MASTER_URI=${ROS_MASTER_URI} \
           -e ROS_IP=${ROS_IP} \
           -e NVIDIA_DRIVER_CAPABILITIES=all \
           --name "${image_name//\//-}" \
           --rm \
           -it ${image_name}