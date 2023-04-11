import numpy as np
import rospy
import tensorflow as tf
import tensorflow_hub as hub

from std_msgs.msg import Header
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose, BoundingBox2D
from geometry_msgs.msg import Pose, Pose2D, PoseWithCovariance
from lcastor_perception.srv import DetectObjects, DetectObjectsResponse

from cv_bridge import CvBridge


class Detector():

  def __init__(self, model="faster_rcnn"):
    self.model = hub.load("https://tfhub.dev/tensorflow/faster_rcnn/inception_resnet_v2_640x640/1")

    self.detection_service = rospy.Service("detect_objects", DetectObjects, self.handle_detection_request)

    self.bridge = CvBridge()

  def handle_detection_request(self, req):
    print("Detecting in img...")

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

    detection_msg = Detection2DArray(header=header, detections=detections)
    return DetectObjectsResponse(detections=detection_msg)


if __name__ == '__main__':
  rospy.init_node("detector")

  detector = Detector()

  print("Detector running")
  rospy.spin()