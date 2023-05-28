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

  labels = np.genfromtxt("../src/lcastor_perception/labels_coco.txt", dtype=str)
  #labels = np.genfromtxt("../src/lcastor_perception/labels_imagenet.txt", dtype=str)

  for child in Path(img_path).iterdir():
    if child.is_file():
      print(img_path + child.name)
      img = cv2.imread(img_path + child.name)
      img_msg = bridge.cv2_to_imgmsg(img, encoding="passthrough")

      detect_objects = rospy.ServiceProxy("detect_objects", DetectObjects)
      results = detect_objects(img_msg)
      for i, detection in enumerate(results.detections.detections):
        if detection.results[0].score > 0.8:
          print(detection.results[0].score, labels[detection.results[0].id - 1])
          x_0 = int(detection.bbox.center.x - detection.bbox.size_x / 2)
          x_1 = int(detection.bbox.center.x + detection.bbox.size_x / 2)
          y_0 = int(detection.bbox.center.y - detection.bbox.size_y / 2)
          y_1 = int(detection.bbox.center.y + detection.bbox.size_y / 2)

          img = cv2.rectangle(img, (x_0, y_0), (x_1, y_1), color=(255, 0, 0))
          img = cv2.putText(img, labels[detection.results[0].id - 1], (x_0, y_0),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0))

      segmask = bridge.imgmsg_to_cv2(results.detections.detections[0].source_img, desired_encoding="passthrough")
      cv2.imshow("segmask", segmask * 255)
      cv2.imshow("img", img)
      cv2.waitKey()
      print()
