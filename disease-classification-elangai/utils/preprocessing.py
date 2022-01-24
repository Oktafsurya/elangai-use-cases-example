"""
Preprocessing utilities for disease-classification-elangai application. 

It processes raw input into preprocessed data.
"""


from elangai import *
import cv2
import numpy as np

# Please replace 'your_model_name' to the respected model name
@elangai_callback("preprocessing")
def disease_preprocessing(ELANGAI_ENV):
    """Preprocessing function.

    :param ELANGAI_ENV: elangai environment variable to interact with
    :return: valid dictionary of preprocessing output
    """
    input_frame = ELANGAI_ENV['frame']
    preprocessed_input = {'x': None}
    input_frame = cv2.resize(input_frame, (224,224), interpolation = cv2.INTER_AREA)
    input_frame = np.array(input_frame, dtype='float32', order='C')
    # input_frame -= np.array([127, 127, 127])
    input_frame /= 255
    # input_frame = input_frame.transpose((2, 0, 1))
    preprocessed_input['x'] = input_frame
    # print('[INFO] preprocess -> ELANGAI_ENV:', ELANGAI_ENV)
    return {'preprocessed_input': preprocessed_input, }
