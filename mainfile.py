import tensorflow as tf
import numpy as np

import cv2
cap = cv2.VideoCapture(0)

import sys
sys.path.append("..")
from utils import label_map_util
from utils import visualization_utils as viz_utils

print("\nLoading Saved Model...")
PATH_TO_MODEL = "new_model" # point to folder containing 'model' folder
detect_fn = tf.saved_model.load(PATH_TO_MODEL + "/saved_model")

print("\nLoading Category Index...")
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_MODEL + "/mscoco_label_map.pbtxt", use_display_name=True)

print("\nBegin Live Image Predicting")
while True:
    ret, image_np = cap.read()
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]
    if input_tensor == None:
        continue

    detections = detect_fn(input_tensor)
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
    detections['num_detections'] = num_detections
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'],
        detections['detection_classes'],
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=0.85,
        agnostic_mode=False)

    cv2.imshow('object detection', cv2.resize(image_np_with_detections, (800,600)))
    if cv2.waitKey(25) == ord('q'):
        cv2.destroyAllWindows()
        break