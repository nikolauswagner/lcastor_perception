#!/bin/bash

set -e
source "/opt/ros/noetic/setup.bash"

echo " "
echo "###"
echo "### This container is part of LCASTOR!"
echo "### Report any issues to https://github.com/LCAS/LCASTOR/issues"
echo "###"
echo " "

{
  echo "Try starting object detection..."
  source "/home/lcastor/ros_ws/devel/setup.bash"
  roslaunch lcastor_perception object_detector.launch detector_type:="mask_rcnn_coco"

} || {

  echo "Object detection failed."
  exec "$@"

}