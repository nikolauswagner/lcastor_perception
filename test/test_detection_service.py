import rospy
import os
from pathlib import Path
import cv2
from cv_bridge import CvBridge

from sensor_msgs.msg import Image

from lcastor_perception.srv import DetectObjects



if __name__ == '__main__':
  rospy.init_node("test_detection_services")

  bridge = CvBridge()

  img_path = "./test_imgs/"

  for child in Path(img_path).iterdir():
    if child.is_file():
      print(img_path + child.name)
      img = cv2.imread(img_path + child.name)
      img_msg = bridge.cv2_to_imgmsg(img, encoding="passthrough")
      print(img_msg)

      detect_objects = rospy.ServiceProxy("detect_objects", DetectObjects)
      detections = detect_objects(img_msg)
      cv2.imshow("img", img)
      cv2.waitKey()