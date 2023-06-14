#!usr/bin/env python
# -*- coding: utf-8 -*-

# Nikolaus Wagner (C) 2023
# nwagner@lincoln.ac.uk

import numpy as np
import rospy
import os
import cv2
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose2D, PoseWithCovariance
from std_msgs.msg import Header
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose, BoundingBox2D

from lcastor_perception.srv import DetectObjects, DetectObjectsResponse


class ObjectDetector():

  def __init__(self, model_name="faster_rcnn_openimages"):
    rospy.loginfo("Using model: " + model_name)
    self.model_name = model_name
    if self.model_name == "faster_rcnn_coco":
      import tensorflow as tf
      import tensorflow_hub as hub
      self.model = hub.load("https://tfhub.dev/tensorflow/faster_rcnn/inception_resnet_v2_640x640/1")
    elif self.model_name == "faster_rcnn_openimages":
      import tensorflow as tf
      import tensorflow_hub as hub
      self.model = hub.load("https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1")
#    elif self.model_name == "hrnet_coco":
#      import tensorflow as tf
#      import tensorflow_hub as hub
#      self.model = hub.load("https://tfhub.dev/google/HRNet/coco-hrnetv2-w48/1")
    elif self.model_name == "mask_rcnn_coco":
      import torch
      import torchvision
      self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      path = os.path.dirname(os.path.realpath(__file__))
      checkpoint = torch.load(path + "/../../models/mask_rcnn_coco.pth")
      self.model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights=None)
      self.model.load_state_dict(checkpoint)
      self.model.eval()
      self.model.to(self.device)
    else:
      rospy.logerr("Unknown model!")

    self.detection_service = rospy.Service("detect_objects", DetectObjects, self.handle_detection_request)

    self.bridge = CvBridge()
    self.torch_preprocess = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
        #torchvision.transforms.Normalize([0.485, 0.456, 0.406], 
        #                                 [0.229, 0.224, 0.225])
    ])

  def handle_detection_request(self, req):
    rospy.loginfo("Running detector...")

    img = self.bridge.imgmsg_to_cv2(req.image, desired_encoding="8UC3").copy()
    return_img = self.bridge.cv2_to_imgmsg(img, "passthrough")

    img_width = img.shape[1]
    img_height = img.shape[0]

    detection_classes = []
    detection_scores = []
    detection_boxes = []

    if self.model_name == "faster_rcnn_coco":
      img_tensor = tf.image.convert_image_dtype(img, tf.uint8)[tf.newaxis, ...]
      results = self.model(img_tensor)
      detection_classes = results["detection_classes"].numpy()[0].astype(np.uint32)
      detection_scores = results["detection_scores"].numpy()[0]
      detection_boxes = results["detection_boxes"].numpy()[0]
      detection_boxes[..., 0] *= img_height
      detection_boxes[..., 1] *= img_width
      detection_boxes[..., 2] *= img_height
      detection_boxes[..., 3] *= img_width

    elif self.model_name == "faster_rcnn_openimages":
      img_tensor = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
      results = self.model.signatures["default"](img_tensor)
      detection_classes = results["detection_class_labels"].numpy().astype(np.uint32)
      detection_scores = results["detection_scores"].numpy()
      detection_boxes = results["detection_boxes"].numpy()
      detection_boxes[..., 0] *= img_height
      detection_boxes[..., 1] *= img_width
      detection_boxes[..., 2] *= img_height
      detection_boxes[..., 3] *= img_width
#
#    elif self.model_name == "hrnet_coco":
#      img_tensor = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
#      results = self.model.predict(img_tensor)
#      img_seg = np.argmax(results.numpy()[0], axis=2).astype(int)
#      ids = np.unique(img_seg)
#      for i in ids:
#        seg_id = (img_seg == i)
#        rows = np.any(seg_id, axis=1)
#        cols = np.any(seg_id, axis=0)
#        r_min, r_max = np.where(rows)[0][[0, -1]]
#        c_min, c_max = np.where(cols)[0][[0, -1]]
#
#        detection_boxes.append([r_min + ((r_max - r_min) / 2),
#                                r_max - r_min,
#                                c_min + ((c_max - c_min) / 2),
#                                c_max - c_min])
#
#      print(detection_boxes)
#
#      cv2.imshow("results", np.argmax(results.numpy()[0], axis=2).astype(np.uint8))
#      cv2.waitKey()
#      #detection_classes = results["detection_class_labels"].numpy()[0].astype(np.uint32)
#      #detection_scores = results["detection_scores"].numpy()[0]
#      #detection_boxes = results"[detection_boxes"].numpy()[0]

    elif self.model_name == "mask_rcnn_coco":
      img_tensor = self.torch_preprocess(img).unsqueeze(0).to(self.device)
      results = self.model(img_tensor)
      num_detections = results[0]["labels"].squeeze().detach().cpu().numpy().size
      
      if num_detections == 0:
        segmask = np.zeros_like(img[..., 0])
      elif num_detections == 1:
        segmask = results[0]["masks"].squeeze().detach().cpu().numpy().astype(np.uint16)
        detection_classes = [results[0]["labels"].squeeze().detach().cpu().numpy()]
        detection_scores = [results[0]["scores"].squeeze().detach().cpu().numpy()]
        detection_boxes = results[0]["boxes"].squeeze().detach().cpu().numpy()
        detection_boxes = [detection_boxes.take((1, 0, 3, 2), axis=0)]
      else:
        segmask = np.argmax(results[0]["masks"].squeeze().detach().cpu().numpy(), axis=0).astype(np.uint16)
        detection_classes = results[0]["labels"].squeeze().detach().cpu().numpy()
        detection_scores = results[0]["scores"].squeeze().detach().cpu().numpy()
        detection_boxes = results[0]["boxes"].squeeze().detach().cpu().numpy()
        detection_boxes = detection_boxes.take((1, 0, 3, 2), axis=1)
      
      return_img = self.bridge.cv2_to_imgmsg(segmask, "passthrough")

    header = Header(stamp=rospy.get_rostime())
    detections = []

    for i in range(num_detections):
      results = [ObjectHypothesisWithPose(id=detection_classes[i],
                                          score=detection_scores[i],
                                          pose=PoseWithCovariance())]
      center = Pose2D(x=(detection_boxes[i][3] + detection_boxes[i][1]) / 2,
                      y=(detection_boxes[i][2] + detection_boxes[i][0]) / 2)
      size_x = (detection_boxes[i][3] - detection_boxes[i][1])
      size_y = (detection_boxes[i][2] - detection_boxes[i][0])
      bbox = BoundingBox2D(center=center,
                           size_x=size_x,
                           size_y=size_y)

      detection = Detection2D(header=header,
                              results=results,
                              bbox=bbox,
                              source_img=return_img)

      detections.append(detection)

    rospy.loginfo("Detecting done!")

    detection_msg = Detection2DArray(header=header, detections=detections)
    return DetectObjectsResponse(detections=detection_msg)


if __name__ == '__main__':
  rospy.init_node("object_detector")

  rospy.loginfo("Initialising object detector...")
  detector = ObjectDetector("mask_rcnn_coco")

  rospy.loginfo("Object detector is up!")
  rospy.spin()
