"""
Postprocessing utilities for face-mask-detection application. 

It processes predicted data into intepretable output.
"""


from elangai import *
import numpy as np
import cv2

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
def face_mask_postprocessing(ELANGAI_ENV):
    """Postprocessing function.

    :param ELANGAI_ENV: elangai environment variable to interact with
    :return: valid dictionary of postprocessing output
    """
    img = ELANGAI_ENV['frame']
    out_detect = ELANGAI_ENV['boxes'].reshape((3000,4))
    label_list = ELANGAI_ENV['face_mask']['label']
    out_scores = ELANGAI_ENV['scores'].reshape((3000,len(label_list)))

    WIDTH = img.shape[0]
    HEIGH = img.shape[1]
    NMS_THRESH = 0.8
    SCORE_THRES = 0.8

    boxes, labels, probs = predict(WIDTH, HEIGH, out_scores, out_detect, 0.7)
    print('labels:', labels)
    print('labels type:', type(labels))

    label_each_frame = []
    score_each_frame = []

    counter = str(boxes.shape[0])
    # print('[INFO] counter:', counter)

    for i in range(boxes.shape[0]):
        box = boxes[i, :]
        label = labels[0]
        label_info = label_list[label]
        conf_score = round(float(probs[i]), 3) * 100
        # print('label:', label)
        # print('label_list[label]:', label_list[label])

        label_each_frame.append(str(label_list[label]))
        score_each_frame.append(str(probs[i]))

        green = (0, 255, 0)
        red = (255, 0, 0)
        orange = (255,165,0)
        box_color = (80,18,236)

        if label_info == 'with_mask':
            box_color = green

        if label_info == 'without_mask':
            box_color = red
        
        if label_info == 'mask_weared_incorrect':
            box_color = orange

        x1, y1, x2, y2 = box
        cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 2)
        cv2.rectangle(img, (x1, y2 - 20), (x2, y2), box_color, cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        text = f"{label_list[label]}: {conf_score}%"
        cv2.putText(img, text, (x1 + 6, y2 - 6), font, 0.3, (255, 255, 255), 1)
    
    json_array = {"class_str": label_each_frame,
                	"score": score_each_frame,
                    "counter": counter}

    return {'inference_data': {'frame':img, 'data':json_array}}
    # inference_data = dict()
    # return {'inference_data': inference_data, }
