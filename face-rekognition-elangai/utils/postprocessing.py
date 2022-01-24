"""
Postprocessing utilities for try-face-rekog application. 

It processes predicted data into intepretable output.
"""

from elangai import *
import numpy as np
import sklearn.preprocessing 
import cv2
import os
import sys
from os.path import dirname, join, abspath, isfile 

global face_data_dir 
# face_data_dir = join(dirname(abspath(str(sys.modules['__main__'].__file__))), 'input/face_database')
# mobileface
face_data_dir = join(dirname(abspath(str(sys.modules['__main__'].__file__))), 'input/face_database/mobileface_data_128')
global face_data_list
face_data_list = [f for f in os.listdir(face_data_dir) if isfile(join(face_data_dir, f))]

def area_of(left_top, right_bottom):
    """
    Compute the areas of rectangles given two corners.
    Args:
        left_top (N, 2): left top corner.
        right_bottom (N, 2): right bottom corner.
    Returns:
        area (N): return the area.
    """
    hw = np.clip(right_bottom - left_top, 0.0, None)
    return hw[..., 0] * hw[..., 1]

def iou_of(boxes0, boxes1, eps=1e-5):
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Args:
        boxes0 (N, 4): ground truth boxes.
        boxes1 (N or 1, 4): predicted boxes.
        eps: a small number to avoid 0 as denominator.
    Returns:
        iou (N): IoU values.
    """
    overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)

def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
    """
    Perform hard non-maximum-supression to filter out boxes with iou greater
    than threshold
    Args:
        box_scores (N, 5): boxes in corner-form and probabilities.
        iou_threshold: intersection over union threshold.
        top_k: keep top_k results. If k <= 0, keep all the results.
        candidate_size: only consider the candidates with the highest scores.
    Returns:
        picked: a list of indexes of the kept boxes
    """
    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    picked = []
    indexes = np.argsort(scores)
    indexes = indexes[-candidate_size:]
    print('[INFO] postprocessing -> indexes:', indexes)
    while len(indexes) > 0:
        current = indexes[-1]
        picked.append(current)
        if 0 < top_k == len(picked) or len(indexes) == 1:
            break
        current_box = boxes[current, :]
        indexes = indexes[:-1]
        rest_boxes = boxes[indexes, :]
        iou = iou_of(
            rest_boxes,
            np.expand_dims(current_box, axis=0),
        )
        indexes = indexes[iou <= iou_threshold]

    return box_scores[picked, :]

def predict(width, height, confidences, boxes, prob_threshold, iou_threshold=0.5, top_k=-1):
    """
    Select boxes that contain human faces
    Args:
        width: original image width
        height: original image height
        confidences (N, 2): confidence array
        boxes (N, 4): boxes array in corner-form
        iou_threshold: intersection over union threshold.
        top_k: keep top_k results. If k <= 0, keep all the results.
    Returns:
        boxes (k, 4): an array of boxes kept
        labels (k): an array of labels for each boxes kept
        probs (k): an array of probabilities for each boxes being in corresponding labels
    """
    #boxes = boxes[0]
    #confidences = confidences[0]
    picked_box_probs = []
    picked_labels = []
    for class_index in range(1, confidences.shape[1]):
        probs = confidences[:, class_index]
        mask = probs > prob_threshold
        probs = probs[mask]
        if probs.shape[0] == 0:
            continue
        subset_boxes = boxes[mask, :]
        box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
        box_probs = hard_nms(box_probs,
           iou_threshold=iou_threshold,
           top_k=top_k,
           )
        picked_box_probs.append(box_probs)
        picked_labels.extend([class_index] * box_probs.shape[0])
    if not picked_box_probs:
        return np.array([]), np.array([]), np.array([])
    picked_box_probs = np.concatenate(picked_box_probs)
    picked_box_probs[:, 0] *= height
    picked_box_probs[:, 1] *= width
    picked_box_probs[:, 2] *= height
    picked_box_probs[:, 3] *= width
    return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]

def get_dist_sim(out1, out2):
    dist = np.sum(np.square(out1-out2))
    sim = np.dot(out1, out2.T)
    return dist, sim

# Please replace 'your_model_name' to the respected model name
@elangai_callback("postprocessing")
def face_640_postprocessing(ELANGAI_ENV):
    """Postprocessing function.

    :param ELANGAI_ENV: elangai environment variable to interact with
    :return: valid dictionary of postprocessing output
    """
    img = ELANGAI_ENV['frame']

    out_detect = ELANGAI_ENV['boxes'].reshape((17640,4))
    label_list = ELANGAI_ENV['face_640']['label']
    out_scores = ELANGAI_ENV['scores'].reshape((17640,len(label_list)))

    WIDTH = img.shape[0]
    HEIGH = img.shape[1]

    boxes, labels, probs = predict(WIDTH, HEIGH, out_scores, out_detect, 0.7)
    inference_data = {'frame': img, 'data':''}
    return {'inference_data': inference_data,
            'boxes':boxes, 
            'labels':labels, 
            'probs':probs}

@elangai_callback("postprocessing")
def mobilefacenet_postprocessing(ELANGAI_ENV):
    box = ELANGAI_ENV['box']
    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
    # face_feature = ELANGAI_ENV['fc1'].reshape((1, 512))
    # mobileface
    face_feature = ELANGAI_ENV['output'].reshape((1, 128))
    face_feature = sklearn.preprocessing.normalize(face_feature).flatten()
    ELANGAI_ENV['face_features'].append(face_feature)
    face_ref_list = []
    dist_list = []
    sim_list = []

    for face_data in face_data_list:
        print('[INFO] face_data:', face_data)
        face_ref = np.load(join(face_data_dir, face_data))
        face_feat = face_feature
        #dist = cosine_similarity(face_feat, face_ref)
        # print('[INFO] face_ref.shape:', face_ref.shape)
        # print('[INFO] face_feat.shape:', face_feat.shape)
        dist, sim = get_dist_sim(face_ref, face_feat)
        # print('dist:', dist)
        # print('sim:', sim)
        face_ref_list.append(face_data)
        dist_list.append(dist)
        sim_list.append(sim)
            
    sim_max = max(sim_list)
    sim_max_idx = sim_list.index(sim_max)

    if sim_max > 0.6:
        name = str(face_ref_list[sim_max_idx].split('.')[0])
        text = f'{name} - {round(sim_max*100, 2)}%'
    else:
        name = 'UNKNOWN'
        text = f'{name}'
        
    cv2.rectangle(ELANGAI_ENV['img_result'], (x1, y1), (x2, y2), (80,18,236), 2)
    cv2.rectangle(ELANGAI_ENV['img_result'], (x1, y2 - 20), (x2, y2), (80,18,236), cv2.FILLED)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(ELANGAI_ENV['img_result'], text, (x1, y2 - 10), font, 1, (255, 255, 255), 2)

    return {'inference_data': {'frame':ELANGAI_ENV['img_result'], 'data':''}}
