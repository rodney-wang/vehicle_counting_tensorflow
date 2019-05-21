#!/usr/bin/python
# -*- coding: utf-8 -*-
# ----------------------------------------------
# --- Author         : Fei Wang
# --- Mail           : rodney.wang@gmail.com
# --- Date           : 27th January 2018
# ----------------------------------------------

# Imports
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tensorflow as tf
import cv2
import numpy as np
import argparse
import glob

# Object detection imports
from utils import label_map_util
from utils import filter_detection_box_crop
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

if tf.__version__ < '1.4.0':
    raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!'
                      )

# Variables
total_passed_vehicle = 0  # using it to count vehicles

# By default I use an "SSD with Mobilenet" model here. See the detection model zoo (https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.
# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = \
    'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

# Download Model
# uncomment if you have not download the model yet
# Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# Loading label map
# Label maps map indices to category names, so that when our convolution network predicts 5, we know that this corresponds to airplane. Here I use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map,
        max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# Helper code
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width,
            3)).astype(np.uint8)


# Detection
def object_detection_function(frame, img_basename, crop_folder):
    total_passed_vehicle = 0

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:

            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')


            input_frame = frame

            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(input_frame, axis=0)

            # Actual detection.
            (boxes, scores, classes, num) = \
                sess.run([detection_boxes, detection_scores,
                         detection_classes, num_detections],
                         feed_dict={image_tensor: image_np_expanded})


            # Visualization of the results of a detection.

            boxes_filtered=filter_detection_box_crop.crop_boxes_from_image(
                                img_basename,
                                crop_folder,
                                input_frame,
                                np.squeeze(boxes),
                                np.squeeze(classes).astype(np.int32),
                                np.squeeze(scores),
                                category_index,
                                use_normalized_coordinates=True,
                                line_thickness=4,
                                )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-l', '--data_folder', default='/ssd/wfei/data/hongkong_raw/batch3')
    parser.add_argument('-c', '--crop_folder', default='/ssd/wfei/data/plate_for_label/hk_double/car_crop_20190517', help='cropped car image folder')
    args = parser.parse_args()

    if not os.path.exists(args.crop_folder):
          os.makedirs(args.crop_folder)
    img_all = glob.glob(os.path.join(args.data_folder, "*.jpg"))
    img_all.sort()

    for idx, fname in enumerate(img_all):
        print idx, fname
        #if idx < 240:
        #    continue
        bname = os.path.splitext(os.path.basename(fname))[0]
        frame = cv2.imread(fname)
        object_detection_function(frame, bname, args.crop_folder)
