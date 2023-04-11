#!usr/bin/env python
# -*- coding: utf-8 -*-

# Nikolaus Wagner (C) 2023
# nwagner@lincoln.ac.uk

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from pathlib import Path

from lcastor_perception.srv import DetectObjects

if __name__ == '__main__':
  rospy.init_node("test_detection_services")

  bridge = CvBridge()

  img_path = "./test_imgs/"

  labels = np.genfromtxt("../src/labels.txt", dtype=str)
  print(labels)

  for child in Path(img_path).iterdir():
    if child.is_file():
      print(img_path + child.name)
      img = cv2.imread(img_path + child.name)
      img_msg = bridge.cv2_to_imgmsg(img, encoding="passthrough")

      detect_objects = rospy.ServiceProxy("detect_objects", DetectObjects)
      results = detect_objects(img_msg)
      for detection in results.detections.detections:
        if detection.results[0].score > 0.5:
          print(labels[detection.results[0].id - 1])

      cv2.imshow("img", img)
      cv2.waitKey()
      print()
