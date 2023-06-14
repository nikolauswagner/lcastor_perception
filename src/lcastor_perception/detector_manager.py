#!usr/bin/env python
# -*- coding: utf-8 -*-

# Nikolaus Wagner (C) 2023
# nwagner@lincoln.ac.uk

import rospy
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

from lcastor_perception.srv import DetectObjects

class DetectorManager():
  def __init__(self, image_ns):
    rospy.Subscriber("{:s}/image_raw".format(image_ns), Image, self.callback_img)

    self.img = None

    self.bridge = CvBridge()
    self.labels = np.genfromtxt("./labels_coco.txt", dtype=str)

  def callback_img(self, img_msg):
    detect_objects = rospy.ServiceProxy("detect_objects", DetectObjects)
    results = detect_objects(img_msg)

    img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="passthrough")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    for i, detection in enumerate(results.detections.detections):
      if detection.results[0].score > 0.8:
        print(detection.results[0].score, self.labels[detection.results[0].id - 1])
        x_0 = int(detection.bbox.center.x - detection.bbox.size_x / 2)
        x_1 = int(detection.bbox.center.x + detection.bbox.size_x / 2)
        y_0 = int(detection.bbox.center.y - detection.bbox.size_y / 2)
        y_1 = int(detection.bbox.center.y + detection.bbox.size_y / 2)

        img = cv2.rectangle(img, (x_0, y_0), (x_1, y_1), color=(255, 0, 0))
        img = cv2.putText(img, self.labels[detection.results[0].id - 1], (x_0, y_0),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0))

    cv2.imshow("img", img)
    cv2.waitKey()


if __name__ == '__main__':
  rospy.init_node("detector_manager")
  image_ns = rospy.get_param("image_ns", "/xtion/rgb")
  depth_ns = rospy.get_param("image_ns", "/xtion/depth")

  print("Initiating detector manager...")
  dm = DetectorManager(image_ns)

  print("Detector manager running!")
  rospy.spin()
