"""
Preprocessing utilities for try-face-rekog application. 

It processes raw input into preprocessed data.
"""


from elangai import *
import cv2
import numpy as np

# Please replace 'your_model_name' to the respected model name
@elangai_callback("preprocessing")
def face_640_preprocessing(ELANGAI_ENV):
    """Preprocessing function.

    :param ELANGAI_ENV: elangai environment variable to interact with
    :return: valid dictionary of preprocessing output
    """
    frame = ELANGAI_ENV['frame']
    preprocessed_input = {'input': None}
    img = cv2.resize(frame, (640, 480)) 
    img_mean = np.array([127, 127, 127])
    img = (img - img_mean) / 128
    img = img.astype(np.float32)
    img = img.transpose((2, 0, 1))
    preprocessed_input['input'] = img
    return {'preprocessed_input': preprocessed_input}
    # return {'preprocessed_input': preprocessed_input, }

# Please replace 'your_model_name' to the respected model name
@elangai_callback("preprocessing")
def mobilefacenet_preprocessing(ELANGAI_ENV):
    """Preprocessing function.

    :param ELANGAI_ENV: elangai environment variable to interact with
    :return: valid dictionary of preprocessing output
    """
    img = ELANGAI_ENV['frame']
    box = ELANGAI_ENV['box']
    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
    cropped_face_image = img.copy()
    cropped_face_image = cropped_face_image[y1:y2,x1:x2]
    cropped_face_image = cv2.resize(cropped_face_image, (112, 112)) 
    # mobilefacenet
    image_mean = np.array([127, 127, 127])
    cropped_face_image = (cropped_face_image - image_mean) / 128
    # finish
    cropped_face_image = np.transpose(cropped_face_image, [2, 0, 1])
    cropped_face_image = np.expand_dims(cropped_face_image, axis=0)
    cropped_face_image = cropped_face_image.astype(np.float32)
    # preprocessed_input = {'data': cropped_face_image}
    # mobilefacenet
    preprocessed_input = {'input': cropped_face_image}
    return {'preprocessed_input':  preprocessed_input, }

