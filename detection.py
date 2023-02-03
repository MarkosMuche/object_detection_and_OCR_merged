

import os
import cv2
import tensorflow as tf
import numpy as np
from PIL import Image
from object_detection.utils import label_map_util,visualization_utils as v_utils,config_util
from matplotlib import pyplot as plt


def single_model_inference(image,MODEL_PATH,LABELS_PATH ):
    
    detection_model = tf.saved_model.load(MODEL_PATH)

    # list of strings to add correct label for each box
    category_index = label_map_util.create_category_index_from_labelmap(LABELS_PATH,use_display_name=True)
    # conver image to numpy array
    image = np.asarray(image)
    # the model takes a tensor of image
    input_tensor = tf.convert_to_tensor(image)
    # model expects batch of images so tf.newaxis
    input_tensor = input_tensor[tf.newaxis,...]
    # running inference
    detect_fn = detection_model.signatures['serving_default']
    output_dict = detect_fn(input_tensor)
    
    # convert all batch tensor output to numpy array
    # taking index [0] to remove the batch dimension
    # only interested in first number of detection
    num_detection = int(output_dict.pop('num_detections'))
    
    output_dict = {key:value[0, :num_detection].numpy() for key,value in output_dict.items()}
    output_dict['num_detections'] = num_detection
    # detection classes must be an integer
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
    # visualize the detected objects
    final_img = v_utils.visualize_boxes_and_labels_on_image_array(image, 
                                                                  output_dict['detection_boxes'],
                                                                  output_dict['detection_classes'],
                                                                  output_dict['detection_scores'], 
                                                                  category_index,
                                                                  use_normalized_coordinates=True,
                                                                  line_thickness=8)
    return final_img


if __name__ == "__main__":

    img_path = ['TensorFlow/workspace/training_demo/images/train/Screenshot-2022-01-13-at-12.12.08.png',] # add all your image path here Example : "E:\Train\IMG22.jpg"
    MODEL_PATH = "/content/ID-card-detection-project/TensorFlow/workspace/training_demo/exported-models/my_model/saved_model" # add "model_graph" path from the drive
    LABELS_PATH = "TensorFlow/workspace/training_demo/annotations/label_map.pbtxt" # add the "Annotations" folder path 
         
    i = 0
    for p in img_path:
        i += 1
        img = cv2.imread(p)
        # img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        MODEL_PATH = '/media/mark/New Volume/projects/Ben/ID-card-detection-project/Trained_model/my_model/saved_model'
        LABELS_PATH = '/media/mark/New Volume/projects/Ben/ID-card-detection-project/TensorFlow/workspace/training_demo/annotations'
        final_Img = single_model_inference(img, MODEL_PATH, LABELS_PATH)
        final_Img = cv2.cvtColor(final_Img,cv2.COLOR_RGB2BGR)
        image_name = 'Saved_img_{}.jpg'.format(str(i));
        cv2.imwrite(image_name,final_Img)
        print(image_name,' saved sucessfully');



    


    
    
