#!usr/bin/env python
# -*- coding: utf-8 -*-

# Nikolaus Wagner (C) 2023
# nwagner@lincoln.ac.uk

import numpy as np
import rospy
import os
import cv2
import gc
import torch
import torchvision
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose2D, PoseWithCovariance
from std_msgs.msg import Header, Bool
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose, BoundingBox2D

from lcastor_perception.srv import DetectObjects, DetectObjectsResponse, LoadModel, LoadModelResponse, UnloadModel, UnloadModelResponse


class ObjectDetector():

  def __init__(self, model_name="faster_rcnn_openimages"):
    rospy.loginfo("Using model: " + model_name)
    self.model_name = model_name
    self.model_is_loaded = False
    self.load_model()

    self.serv_detect_objects = rospy.Service("/object_detector/detect_objects", DetectObjects, self.handle_detection_request)
    self.serv_load_model = rospy.Service("/object_detector/load_model", LoadModel, self.load_model)
    self.serv_unload_model = rospy.Service("/object_detector/unload_model", UnloadModel, self.unload_model)

    self.bridge = CvBridge()
    self.torch_preprocess = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.ConvertImageDtype(torch.float)
        #torchvision.transforms.Normalize([0.485, 0.456, 0.406], 
        #                                 [0.229, 0.224, 0.225])
    ])

  def load_model(self, req=None):
    if req:
      self.model_name = req.model_name.data

    if self.model_name == "faster_rcnn_coco":
      import tensorflow as tf
      import tensorflow_hub as hub

      self.model = hub.load("https://tfhub.dev/tensorflow/faster_rcnn/inception_resnet_v2_640x640/1")

    elif self.model_name == "faster_rcnn_openimages":
      import tensorflow as tf
      import tensorflow_hub as hub

      self.model = hub.load("https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1")

    elif self.model_name == "mask_rcnn_coco":
      self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      path = os.path.dirname(os.path.realpath(__file__))

      if os.path.isfile(path + "/../../models/mask_rcnn_coco.pth"):
        checkpoint = torch.load(path + "/../../models/mask_rcnn_coco.pth")
      else:
        rospy.logerr("Model not found! Make sure model is present locally!")
        return LoadModelResponse(success=Bool(False))

      self.model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights=None)
      self.model.load_state_dict(checkpoint)
      self.model.eval()
      self.model.to(self.device)

    elif self.model_name == "mask_rcnn_ycb":
      num_classes = 34
      from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
      from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

      self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      path = os.path.dirname(os.path.realpath(__file__))
      if os.path.isfile(path + "../../models/maskrcnn_newest.pth"):
        checkpoint = torch.load(path + "../../models/maskrcnn_newest.pth")
      else:
        rospy.logerr("Model not found! Make sure model is present locally!")
        return LoadModelResponse(success=Bool(False))

      self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
      in_features = self.model.roi_heads.box_predictor.cls_score.in_features
      self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
      in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
      self.model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                              256,
                                                              num_classes)
      self.model.load_state_dict(checkpoint)
      self.model.eval()
      self.model.to(self.device)

    else:
      rospy.logerr("Unknown model!")
      return LoadModelResponse(success=Bool(False))

    self.model_is_loaded = True
    return LoadModelResponse(success=Bool(True))

  def unload_model(self, req=None):
    del self.model
    gc.collect()
    torch.cuda.empty_cache()
    self.model_is_loaded = False
    return UnloadModelResponse(success=Bool(True))

  def handle_detection_request(self, req):
    if not self.model_is_loaded:
      rospy.logerr("No model loaded!")
      return DetectObjectsResponse()

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

    elif self.model_name == "mask_rcnn_coco" or self.model_name == "mask_rcnn_ycb":
      img_tensor = self.torch_preprocess(img).unsqueeze(0).to(self.device)
      results = self.model(img_tensor)
      print(results)
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

      # Only attach image to first detection in array
      if i > 0:
        return_img = None

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
  detector = ObjectDetector("mask_rcnn_ycb")

  rospy.loginfo("Object detector is up!")
  rospy.spin()
