"""
Postprocessing utilities for disease-classification-elangai application. 

It processes predicted data into intepretable output.
"""


from elangai import *
import numpy as np
import cv2

color = (0, 0, 0)

# Please replace 'your_model_name' to the respected model name
@elangai_callback("postprocessing")
def disease_postprocessing(ELANGAI_ENV):
    """Postprocessing function.

    :param ELANGAI_ENV: elangai environment variable to interact with
    :return: valid dictionary of postprocessing output
    """
    img = ELANGAI_ENV['frame']
    out_classification = ELANGAI_ENV['Identity'].reshape((38))
    label_list = ELANGAI_ENV['disease']['label']

    cls_max_idx = np.argmax(out_classification)
    cls_out = label_list[cls_max_idx]
    confidence = out_classification[cls_max_idx] * 100
    print('[INFO] cls_max_idx:', cls_max_idx)
    print('[INFO] cls_out:', cls_out)
    print('[INFO] confidence :', confidence)

    disease_name = f'Disease Name: {str(cls_out[:10])}'
    disease_conf = f'Confidence: {round(confidence, 2)}%'
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(img, disease_name, (10, 20), font, 0.5, color, 1)
    cv2.putText(img, disease_conf, (10, 40), font, 0.5, color, 1)

    # inference_data = dict()
    return {'inference_data': {'frame':img, 'data':out_classification}}
    # return {'inference_data': inference_data, }
