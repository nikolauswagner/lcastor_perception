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
  /home/lcastor/ros_ws/src/lcastor_perception/scripts/get_model.sh mask_rcnn_coco
  source "/home/lcastor/ros_ws/devel/setup.bash"
  roslaunch lcastor_perception object_detector.launch detector_type:="mask_rcnn_coco"

} || {

  echo "Object detection failed."
  exec "$@"

}