#!usr/bin/env python
# -*- coding: utf-8 -*-

# Nikolaus Wagner (C) 2023
# nwagner@lincoln.ac.uk

import rospy
import numpy as np
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose, BoundingBox2D

from lcastor_perception.srv import DetectObjects

class DetectorManager():
  def __init__(self, image_ns, depth_ns=""):
    #rospy.Subscriber("{:s}/image_raw".format(image_ns), Image, self.cb_img, queue_size=1, buff_size=1)
    rospy.Subscriber("{:s}".format(image_ns), Image, self.cb_img, queue_size=1)
    if depth_ns:
      #rospy.Subscriber("{:s}/image_raw".format(depth_ns), Image, self.cb_depth)
      rospy.Subscriber("{:s}".format(depth_ns), Image, self.cb_depth, queue_size=1)

    self.pub_img = rospy.Publisher("{:s}/detections".format(image_ns), Image, queue_size=1)

    rospy.Timer(rospy.Duration(1.0/10.0), self.invoke_detection_service)

    self.img_msg = None
    self.img = None
    self.depth_img = None

    self.bridge = CvBridge()
    self.labels = np.genfromtxt("./labels_ycb.txt", dtype=str)

    self.detect_objects = rospy.ServiceProxy("/object_detector/detect_objects", DetectObjects)

  def invoke_detection_service(self, event):

    if self.img is None:
      rospy.logwarn("No images received yet!")
      return

    img = self.img.copy()
    img_msg = self.img_msg
    depth_img = self.depth_img.copy()

    try:
      results = self.detect_objects(img_msg)
    except:
      rospy.logwarn("Cannot connect to detection service!")
      return

    if len(results.detections.detections) > 0: 
      segmask = results.detections.detections[0].source_img
      segmask = self.bridge.imgmsg_to_cv2(segmask, desired_encoding="passthrough").astype(np.uint8)

      for i, detection in enumerate(results.detections.detections):
        if detection.results[0].score > 0.1:
          #if depth_img is not None:
          #  avg_dist = np.median(depth_img[segmask == i])
          #  print(avg_dist)
            #print(depth_img[segmask == i])
            #print(np.median(depth_img[segmask == i]))
         #print(detection.results[0].score, self.labels[detection.results[0].id - 1])
          x_0 = int(detection.bbox.center.x - detection.bbox.size_x / 2)
          x_1 = int(detection.bbox.center.x + detection.bbox.size_x / 2)
          y_0 = int(detection.bbox.center.y - detection.bbox.size_y / 2)
          y_1 = int(detection.bbox.center.y + detection.bbox.size_y / 2)

          img = cv2.rectangle(img, (x_0, y_0), (x_1, y_1), color=(255, 0, 0))
          img = cv2.putText(img, self.labels[detection.results[0].id - 1], (x_0, y_0),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0))

    img_msg = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
    self.pub_img.publish(img_msg)

    #cv2.imshow("img", img)
    #cv2.imshow("segmask", 20*segmask)
    #cv2.imshow("depth", 100* depth_img)
    #cv2.waitKey()

  def cb_img(self, img_msg):
    self.img_msg = img_msg
    img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="passthrough")
    self.img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

  def cb_depth(self, img_msg):
    self.depth_img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="passthrough")


if __name__ == '__main__':
  rospy.init_node("detector_manager")
  image_ns = rospy.get_param("image_ns", "/xtion/rgb/")
  depth_ns = rospy.get_param("depth_ns", "")
  image_ns = "/device_0/sensor_0/Color_0/image/data"
  depth_ns = "/device_0/sensor_0/Depth_0/image/data"

  print("Initiating detector manager...")
  dm = DetectorManager(image_ns, depth_ns)

  print("Detector manager running!")
  rospy.spin()
