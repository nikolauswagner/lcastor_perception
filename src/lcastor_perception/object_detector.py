#!usr/bin/env python
# -*- coding: utf-8 -*-

# Nikolaus Wagner (C) 2023
# nwagner@lincoln.ac.uk

import numpy as np
import rospy
import tensorflow as tf
import tensorflow_hub as hub
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose2D, PoseWithCovariance
from std_msgs.msg import Header
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose, BoundingBox2D

from lcastor_perception.srv import DetectObjects, DetectObjectsResponse


class ObjectDetector():

  def __init__(self, model="faster_rcnn"):
    self.model = hub.load("https://tfhub.dev/tensorflow/faster_rcnn/inception_resnet_v2_640x640/1")

    self.detection_service = rospy.Service("detect_objects", DetectObjects, self.handle_detection_request)

    self.bridge = CvBridge()

  def handle_detection_request(self, req):
    rospy.loginfo("Running detector...")

    img = self.bridge.imgmsg_to_cv2(req.image, desired_encoding="passthrough")
    img_width = img.shape[1]
    img_height = img.shape[0]

    img_tensor = tf.image.convert_image_dtype(img, tf.uint8)[tf.newaxis, ...]

    results = self.model(img_tensor)
    detection_classes = results["detection_classes"].numpy()[0].astype(np.uint32)
    detection_scores = results["detection_scores"].numpy()[0]
    detection_boxes = results["detection_boxes"].numpy()[0]

    header = Header(stamp=rospy.get_rostime())
    detections = []

    for i in range(len(detection_classes)):

      results = [ObjectHypothesisWithPose(id=detection_classes[i],
                                          score=detection_scores[i],
                                          pose=PoseWithCovariance())]
      center = Pose2D(x=img_width * (detection_boxes[i][3] + detection_boxes[i][1]) / 2,
                      y=img_height * (detection_boxes[i][2] + detection_boxes[i][0]) / 2)
      size_x = img_width * (detection_boxes[i][3] - detection_boxes[i][1])
      size_y = img_height * (detection_boxes[i][2] - detection_boxes[i][0])
      bbox = BoundingBox2D(center=center,
                           size_x=size_x,
                           size_y=size_y) 
      detection = Detection2D(header=header,
                              results=results,
                              bbox=bbox,
                              source_img=self.bridge.cv2_to_imgmsg(img, "passthrough"))

      detections.append(detection)

    rospy.loginfo("Detecting done!")

    detection_msg = Detection2DArray(header=header, detections=detections)
    return DetectObjectsResponse(detections=detection_msg)


if __name__ == '__main__':
  rospy.init_node("object_detector")

  rospy.loginfo("Initialising object detector...")
  detector = ObjectDetector()

  rospy.loginfo("Object detector is up!")
  rospy.spin()
