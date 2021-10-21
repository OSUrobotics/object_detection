import tensorflow as tf
import numpy as np
import argparse
from PIL import Image
import os

import cv2
cap = cv2.VideoCapture(0)

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

parser = argparse.ArgumentParser()
parser.add_argument('sample_folder', type=str)
parser.add_argument('results_folder', type=str)
parser.add_argument('--thresh', type=float, required=False, default=0.75)
args = parser.parse_args()

print("\nLoading Saved Model...")
PATH_TO_MODEL = "../model_library/model_10_20_2021" # point to folder containing 'model' folder
detect_fn = tf.saved_model.load(PATH_TO_MODEL + "/model/saved_model")

print("\nLoading Category Index...")
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_MODEL + "/mscoco_label_map.pbtxt", use_display_name=True)

imgs = []
recorded_imgs = []
valid_images = [".jpg",".gif",".png",".tga"]

print("\nLoad All Images in [sample_folder]")
for f in os.listdir(args.sample_folder):
    ext = os.path.splitext(f)[1]
    if ext.lower() not in valid_images:
        continue
    imgs.append(Image.open(os.path.join(args.sample_folder, f)))
    
print("\nRun Prediction on All Images in [sample_folder]")
for img in imgs:
    image_np = np.array(img)
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
        min_score_thresh=args.thresh,
        agnostic_mode=False)
        
    recorded_imgs.append(Image.fromarray(image_np_with_detections))
    
print("\nSaving Predicted Images to [results_folder]")
for i in range(0, len(recorded_imgs)):
    recorded_imgs[i].save(args.results_folder + "/image_" + str(i) + ".jpg")
    
    
    
    
    
    
    
    
    
    
    
    
    
    