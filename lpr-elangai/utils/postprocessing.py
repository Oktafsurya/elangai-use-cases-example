"""
Postprocessing utilities for lpr-elangai-trt application. 

It processes predicted data into intepretable output.
"""


from elangai import *
import cv2
import numpy as np
import easyocr

global ocr_reader 
ocr_reader = easyocr.Reader(['en'])

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

def filter_box(width, height, confidences, boxes, prob_threshold, iou_threshold=0.5, top_k=-1):
    """
    Select boxes that contain object
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

# Please replace 'your_model_name' to the respected model name
@elangai_callback("postprocessing")
def vehicle_postprocessing(ELANGAI_ENV):
    """Postprocessing function.

    :param ELANGAI_ENV: elangai environment variable to interact with
    :return: valid dictionary of postprocessing output
    """
    # print('[INFO] ELANGAI_ENV keys:', ELANGAI_ENV.keys())
    img = ELANGAI_ENV['frame']
    # print('[INFO] preprocess_cropped_plate_img.shape:', img.shape)
    out_detect = ELANGAI_ENV['boxes'].reshape((3000,4))
    out_scores = ELANGAI_ENV['scores'].reshape((3000,3))

    # print('[INFO] out_detect.shape:', out_detect.shape)
    # print('[INFO] out_scores.shape:', out_scores.shape)

    WIDTH = img.shape[0]
    HEIGH = img.shape[1]

    boxes, labels, probs = filter_box(WIDTH, HEIGH, out_scores, out_detect, 0.7)

    inference_data = {'frame': img, 'data':''}
    return {'inference_data': inference_data,
            'vehicle_boxes':boxes, 
            'vehicle_labels':labels, 
            'vehicle_probs':probs}

@elangai_callback("postprocessing")
def plate_postprocessing(ELANGAI_ENV):
    """Postprocessing function.

    :param ELANGAI_ENV: elangai environment variable to interact with
    :return: valid dictionary of postprocessing output
    """
    # orig_img = ELANGAI_ENV['img_result']
    # print('[INFO] ELANGAI_ENV:', ELANGAI_ENV)
    vehicle_img = ELANGAI_ENV['cropped_vehicle_img']
    # print('[INFO] vehicle_img.shape:', vehicle_img.shape)
    out_detect = ELANGAI_ENV['boxes'].reshape((3000,4))
    out_scores = ELANGAI_ENV['scores'].reshape((3000,2))

    # vehicle_box = ELANGAI_ENV['box']
    # x1, y1, x2, y2 = vehicle_box[0], vehicle_box[1], vehicle_box[2], vehicle_box[3]

    WIDTH = vehicle_img.shape[0]
    HEIGH = vehicle_img.shape[1]
    boxes, labels, probs = filter_box(WIDTH, HEIGH, out_scores, out_detect, 0.7)
    print('[INFO] num plate detected:', boxes.shape[0])

    # iterate over num of plate detected
    license_num_list = []
    for i in range(boxes.shape[0]):
        plate_box = boxes[i,:]
        x1, y1, x2, y2 = plate_box[0], plate_box[1], plate_box[2], plate_box[3]
        cropped_plate_img = vehicle_img.copy()
        cropped_plate_img = cropped_plate_img[y1:y2,x1:x2]
        # print('[INFO] cropped_plate_img.shape:', cropped_plate_img.shape)
        license_num = ocr_reader.readtext(cropped_plate_img, detail = 0)
        print('[INFO] license num:', license_num)
        license_num_list.append(license_num)

    inference_data = {'frame': cropped_plate_img, 'data': license_num_list}
    return {'inference_data': inference_data,}