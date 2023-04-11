# rasberry_perception

Object detection and vision utilities for the [LCASTOR](https://github.com/LCAS/LCASTOR/) robocup team.

To bring up the detection service, run:

```bash
rosrun lcastor_perception object_detector.py
```

This will provide a rosservice, accepting [sensor_msgs/Image](https://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/Image.html) messages and returning [vision_msgs/Detection2DArray](https://docs.ros.org/en/noetic/api/vision_msgs/html/msg/Detection2DArray.html).

The service passes the images on to a [Faster R-CNN with Resnet V2 Object detection model](https://tfhub.dev/tensorflow/faster_rcnn/inception_resnet_v2_640x640/1), trained on the [COCO 2017 dataset](https://cocodataset.org/#overview). The labels corresponding to the detection IDs are provided in [labels.txt](src/lcastor_perception/labels.txt).

For a demo, see [test_detection_service.py](test/test_detection_service.py).