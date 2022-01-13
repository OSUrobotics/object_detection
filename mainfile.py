import tensorflow as tf
import numpy as np

import cv2
cap = cv2.VideoCapture(0)
if not(cap.isOpened()):
    print("Can't find webcam, shutting down...")
    quit()
# print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640.0)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480.0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960.0)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960.0)

import os
import sys
sys.path.append("..")
from utils import label_map_util
from utils import visualization_utils as viz_utils
import custom_utils

print("\nLoading Saved Model...")
PATH_TO_MODEL = "new_model" # point to folder containing 'model' folder
detect_fn = tf.saved_model.load(os.path.join(PATH_TO_MODEL, "saved_model"))

print("\nLoading Category Index...")
category_index = label_map_util.create_category_index_from_labelmap(os.path.join(PATH_TO_MODEL, "mscoco_label_map.pbtxt"), use_display_name=True)

print("\nBegin Live Image Predicting")
while True:
    ret, image_np = cap.read() 
    if ret == False:
        print("Error Reading Frame, skipping...")
        continue
    
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]

    detections = detect_fn(input_tensor)
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
    detections['num_detections'] = num_detections
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    image_np_with_detections = image_np.copy()

    custom_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'],
        detections['detection_classes'],
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=0.65,
        agnostic_mode=False,
        line_thickness=8)

    cv2.imshow('object detection', cv2.resize(image_np_with_detections, (800,600)))
    if cv2.waitKey(25) == ord('q'):
        cv2.destroyAllWindows()
        break