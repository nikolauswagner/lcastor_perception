#!usr/bin/env python
# -*- coding: utf-8 -*-

# Nikolaus Wagner (C) 2023
# nwagner@lincoln.ac.uk

import tensorflow_hub as hub

def get_model(model="faster_rcnn"):
  model = hub.load("https://tfhub.dev/tensorflow/faster_rcnn/inception_resnet_v2_640x640/1")