"""
Preprocessing utilities for lpr-elangai-trt application. 

It processes raw input into preprocessed data.
"""


from elangai import *
import cv2
import numpy as np

# Please replace 'your_model_name' to the respected model name
@elangai_callback("preprocessing")
def vehicle_preprocessing(ELANGAI_ENV):
    """Preprocessing function.

    :param ELANGAI_ENV: elangai environment variable to interact with
    :return: valid dictionary of preprocessing output
    """
    frame = ELANGAI_ENV['frame']
    preprocessed_input = {'input_0': None}
    img = cv2.resize(frame, (300, 300)) 
    img_mean = np.array([127, 127, 127])
    img = (img - img_mean) / 128
    img = img.astype(np.float32)
    img = img.transpose((2, 0, 1))
    # print('[INFO] preprocess -> img.shape:', img.shape)
    preprocessed_input['input_0'] = img
    return {'preprocessed_input': preprocessed_input}
    
@elangai_callback("preprocessing")
def plate_preprocessing(ELANGAI_ENV):
    """Preprocessing function.

    :param ELANGAI_ENV: elangai environment variable to interact with
    :return: valid dictionary of preprocessing output
    """
    img = ELANGAI_ENV['frame']
    # get vehicle box
    box = ELANGAI_ENV['box']
    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
    cropped_vehicle_img = img.copy()
    cropped_vehicle_img = cropped_vehicle_img[y1:y2,x1:x2]
    preprocessed_input = {'input_0': None}
    cropped_vehicle_img = cv2.resize(cropped_vehicle_img, (300, 300)) 
    img_mean = np.array([127, 127, 127])
    cropped_vehicle_img = (cropped_vehicle_img - img_mean) / 128
    cropped_vehicle_img = cropped_vehicle_img.astype(np.float32)
    cropped_vehicle_img = cropped_vehicle_img.transpose((2, 0, 1))
    cropped_vehicle_img = np.expand_dims(cropped_vehicle_img, axis=0)
    # print('[INFO] preprocess -> cropped_vehicle_img.shape:', cropped_vehicle_img.shape)
    preprocessed_input['input_0'] = cropped_vehicle_img
    return {'preprocessed_input': preprocessed_input,
            'cropped_vehicle_img': cropped_vehicle_img}