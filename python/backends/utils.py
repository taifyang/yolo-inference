'''
Author: taifyang 
Date: 2024-06-12 22:23:07
LastEditors: taifyang 58515915+taifyang@users.noreply.github.com
LastEditTime: 2024-11-22 22:43:15
Description: utilities functions
'''


import cv2
import numpy as np
from backends.yolo import *


'''
description:                Non-Maximum Suppression
param {*} boxes             detect bounding boxes
param {*} scores            detect scores
param {*} score_threshold   detect score threshold
param {*} nms_threshold     IOU threshold
return {*}                  detect indices
'''
def nms(boxes, scores, score_threshold, nms_threshold):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (y2 - y1 + 1) * (x2 - x1 + 1)
    keep = []
    index = scores.argsort()[::-1] 

    while index.size > 0:
        i = index[0]
        keep.append(i)
        x11 = np.maximum(x1[i], x1[index[1:]]) 
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])
        w = np.maximum(0, x22 - x11 + 1)                              
        h = np.maximum(0, y22 - y11 + 1) 
        overlaps = w * h
        ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)
        idx = np.where(ious <= nms_threshold)[0]
        index = index[idx + 1]
    return keep


'''
description:    convert xywh bounding boxes to x1y1x2y2 bounding boxes
param {*} x     xywh bounding boxes
return {*}      x1y1x2y2 bounding boxes
'''
def xywh2xyxy(x):
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


'''
description:        letterbox image process
param {*} im        input image
param {*} new_shape output shape
param {*} color     filled color
return {*}          output image
'''
def letterbox(im, new_shape=(416, 416), color=(114, 114, 114)):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    
    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))    
    dw, dh = (new_shape[1] - new_unpad[0])/2, (new_shape[0] - new_unpad[1])/2  # wh padding 
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    
    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im


'''
description:            scale boxes
param {*} boxes         bounding boxes
param {*} input_shape   input image shape
param {*} output_shape  output image shape
return {*}              scaled boxes
'''
def scale_boxes(boxes, input_shape, output_shape):
    # Rescale boxes (xyxy) from self.inputs_shape to shape
    gain = min(input_shape[0] / output_shape[0], input_shape[1] / output_shape[1])  # gain  = old / new
    pad = (input_shape[1] - output_shape[1] * gain) / 2, (input_shape[0] - output_shape[0] * gain) / 2  # wh padding
    boxes[..., [0, 2]] -= pad[0]  # x padding
    boxes[..., [1, 3]] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, output_shape[1])  # x1, x2
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, output_shape[0])  # y1, y2
    return boxes


'''
description:    crop mask
param {*} masks input masks
param {*} boxes bounding boxes
return {*}      cropped masks
'''
def crop_mask(masks, boxes):
    n, h, w = masks.shape
    x1, y1, x2, y2 = np.split(boxes[..., :4], 4, axis=1)
    x1, y1, x2, y2 = np.expand_dims(x1, 2), np.expand_dims(y1, 2), np.expand_dims(x2, 2), np.expand_dims(y2, 2)
    r = np.arange(w)[None, None, :]
    c = np.arange(h)[None, :, None]
    cropped_masks = masks * ((r >= x1) & (r < x2) & (c >= y1) & (c < y2))
    return cropped_masks


'''
description:            scale mask
param {*} mask          input masks
param {*} input_shape   input image shape
param {*} output_shape  output image shape
return {*}              scaled masks
'''
def scale_mask(mask, input_shape, output_shape):
    gain = min(input_shape[0] / output_shape[0], input_shape[1] / output_shape[1])  # gain  = old / new
    pad = (input_shape[1] - output_shape[1] * gain) / 2, (input_shape[0] - output_shape[0] * gain) / 2  # wh padding
    mask = mask[int(pad[1]):mask.shape[1]-int(pad[1]), int(pad[0]):mask.shape[0]-int(pad[0])]
    mask = cv2.resize(mask, (output_shape[1], output_shape[0]), cv2.INTER_LINEAR)
    return mask


'''
description:            draw result
param {*} image         input image
param {*} preds         prediction result
param {*} masks         masks
param {*} input_shape   input image shape
return {*}              output image
'''
def draw_result(image, preds, masks=[]):
    image_copy = image.copy()   
    boxes = preds[...,:4].astype(np.int32) 
    scores = preds[...,4]
    classes = preds[...,5].astype(np.int32)
    
    for mask in masks:
        image_copy[mask] = [np.random.randint(0,256), np.random.randint(0,256), np.random.randint(0,256)]
    result = (image*0.5 + image_copy*0.5).astype(np.uint8)
    
    for box, score, cl in zip(boxes, scores, classes):
        top, left, right, bottom = box
        cv2.rectangle(result, (top, left), (right, bottom), (255, 0, 0), 1)
        cv2.putText(result, 'class:{0} score:{1:.2f}'.format(cl, score), (top, left), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
    return result