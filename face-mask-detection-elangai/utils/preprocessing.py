"""
Preprocessing utilities for face-mask-detection application. 

It processes raw input into preprocessed data.
"""


from elangai import *
import cv2 
import numpy as np

# Please replace 'your_model_name' to the respected model name
@elangai_callback("preprocessing")
def face_mask_preprocessing(ELANGAI_ENV):
    """Preprocessing function.

    :param ELANGAI_ENV: elangai environment variable to interact with
    :return: valid dictionary of preprocessing output
    """
    input_frame = ELANGAI_ENV['frame']
    preprocessed_input = {'input_0': None}
    input_frame = cv2.resize(input_frame, (300,300), interpolation = cv2.INTER_AREA)
    input_frame = np.array(input_frame, dtype='float32', order='C')
    input_frame -= np.array([127, 127, 127])
    input_frame /= 128
    input_frame = input_frame.transpose((2, 0, 1))
    
    preprocessed_input['input_0'] = input_frame
    # preprocessed_input = dict()
    return {'preprocessed_input': preprocessed_input, }
