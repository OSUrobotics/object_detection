# import tensorflow libraries
import tensorflow as tf
import numpy as np

# import opencv and find webcam
import cv2
cap = cv2.VideoCapture(0)
if not(cap.isOpened()):
    print("Can't find webcam, shutting down...")
    quit()

# set resolution of camera capture
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960.0)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960.0)

# import system libraries
import os
import sys
sys.path.append("..")
# import tf and custom tf libraries
from utils import label_map_util
import custom_utils

# load prediction model
print("\nLoading Saved Model...")
PATH_TO_MODEL = "new_model" # point to folder containing 'model' folder
detect_fn = tf.saved_model.load(os.path.join(PATH_TO_MODEL, "saved_model"))

# load category index for prediction model
print("\nLoading Category Index...")
category_index = label_map_util.create_category_index_from_labelmap(os.path.join(PATH_TO_MODEL, "mscoco_label_map.pbtxt"), use_display_name=True)

# begin main loop
print("\nBegin Live Image Predicting")
while True:
    # capture image from webcam
    ret, image_np = cap.read() 
    if ret == False:
        print("Error Reading Frame, skipping...")
        continue
    
    # convert image to tensor
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]

    # perform prediction/detection on image tensor
    detections = detect_fn(input_tensor)
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
    detections['num_detections'] = num_detections
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    # draw detection boxes on image using modified visualization function
    custom_utils.visualize_boxes_and_labels_on_image_array(
        image_np,
        detections['detection_boxes'],
        detections['detection_classes'],
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=0.65,
        agnostic_mode=False,
        line_thickness=8)

    # display captured image with detection boxes
    cv2.imshow('object detection', cv2.resize(image_np, (800,600)))
	
    # exit program when 'q' key is pressed
    if cv2.waitKey(25) == ord('q'):
        cv2.destroyAllWindows()
        break