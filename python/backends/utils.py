'''
Author: taifyang 
Date: 2024-06-12 22:23:07
LastEditors: taifyang 58515915+taifyang@users.noreply.github.com
LastEditTime: 2025-12-14 00:05:32
Description: utilities functions
'''


import cv2
import math
import numpy as np
try:
    import cupy
    from cupyx.scipy import ndimage
except:
    print('cupy import failed!')
from backends.yolo import *


'''
description:                Non-Maximum Suppression
param {*} boxes             detect bounding boxes
param {*} scores            detect scores
param {*} iou_threshold     IOU threshold
return {*}                  detect indices
'''
def nms(boxes, scores, iou_threshold):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (y2 - y1 + 1) * (x2 - x1 + 1)
    keep = []
    index = scores.squeeze().argsort()[::-1] 

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
        idx = np.where(ious <= iou_threshold)[0]
        index = index[idx + 1]
    return keep


'''
description:    convert xywh bounding boxes to x1y1x2y2 bounding boxes
param {*} x     xywh bounding boxes
return {*}      x1y1x2y2 bounding boxes
'''
def xywh2xyxy(x):
    y = np.copy(x) if isinstance(x, np.ndarray) else x.clone()
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


def letterbox_cupy(im, new_shape=(416, 416), color=(114, 114, 114)):
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
        im_cupy = cupy.asarray(im)
        zoom_factors = (new_unpad[1]/im.shape[0], new_unpad[0]/im.shape[1], 1) 
        im_cupy = ndimage.zoom(im_cupy, zoom_factors, order=0)
    im_cupy = cupy.pad(array=im_cupy, pad_width=((top, bottom), (left, right), (0, 0)), mode='constant', constant_values=(color, color))
    return im_cupy


'''
description:            scale boxes
param {*} boxes         bounding boxes
param {*} input_shape   input image shape
param {*} output_shape  output image shape
return {*}              scaled boxes
'''
def scale_boxes(boxes, input_shape, output_shape, xywh=False):
    # Rescale boxes (xyxy) from self.inputs_shape to shape
    gain = min(input_shape[0] / output_shape[0], input_shape[1] / output_shape[1])  # gain  = old / new
    pad = (input_shape[1] - output_shape[1] * gain) / 2, (input_shape[0] - output_shape[0] * gain) / 2  # wh padding
    boxes[..., 0] -= pad[0]  # x padding
    boxes[..., 1] -= pad[1]  # y padding
    if not xywh:
        boxes[..., 2] -= pad[0]  # x padding
        boxes[..., 3] -= pad[1]  # y padding
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
    if isinstance(boxes, np.ndarray):
        x1, y1, x2, y2 = np.split(boxes[..., :4], 4, axis=1)
        x1, y1, x2, y2 = np.expand_dims(x1, 2), np.expand_dims(y1, 2), np.expand_dims(x2, 2), np.expand_dims(y2, 2)
        r = np.arange(w)[None, None, :]
        c = np.arange(h)[None, :, None]
    else:
        import torch
        x1, y1, x2, y2 = torch.chunk(boxes[..., :4][:, :, None], 4, 1)  # x1 shape(n,1,1)
        r = torch.arange(w, device=masks.device, dtype=x1.dtype)[None, None, :]  # rows shape(1,1,w)
        c = torch.arange(h, device=masks.device, dtype=x1.dtype)[None, :, None]  # cols shape(1,h,1)
    return masks * ((r >= x1) & (r < x2) & (c >= y1) & (c < y2))


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
    
    for box, score, cls in zip(boxes, scores, classes):
        top, left, right, bottom = box
        cv2.rectangle(result, (top, left), (right, bottom), (255, 0, 0), 1)
        cv2.putText(result, 'class:{0} score:{1:.2f}'.format(cls, score), (top, left), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
    return result

# def draw_result(image, preds, masks=[]):
#     image_copy = image.copy()   
#     boxes = preds[...,:4].astype(np.int32) 
    
#     for mask in masks:
#         image_copy[mask] = [np.random.randint(0,256), np.random.randint(0,256), np.random.randint(0,256)]
#     result = (image*0.5 + image_copy*0.5).astype(np.uint8)
    
#     for box in boxes:
#         top, left, right, bottom = box
#         cv2.rectangle(result, (top, left), (right, bottom), (255, 0, 0), 1)
#     return result


# import torch
# def nms_rotated(boxes, scores, threshold=0.45):
#     """
#     NMS for oriented bounding boxes using probiou and fast-nms.

#     Args:
#         boxes (torch.Tensor): Rotated bounding boxes, shape (N, 5), format xywhr.
#         scores (torch.Tensor): Confidence scores, shape (N,).
#         threshold (float, optional): IoU threshold. Defaults to 0.45.

#     Returns:
#         (torch.Tensor): Indices of boxes to keep after NMS.
#     """
#     if len(boxes) == 0:
#         return np.empty((0,), dtype=np.int8)
#     #sorted_idx = torch.argsort(scores, descending=True)
#     sorted_idx = np.argsort(-scores)#[::-1]
#     boxes = boxes[sorted_idx]
#     #ious = batch_probiou(boxes, boxes).triu_(diagonal=1).numpy()
#     ious = np.triu(batch_probiou_np(boxes, boxes), k=1)
#     #pick = torch.nonzero(ious.max(dim=0)[0] < threshold).squeeze_(-1)
#     pick = np.nonzero(np.max(ious, axis=0) < threshold)[0]
#     return sorted_idx[pick]


# def batch_probiou(obb1, obb2, eps=1e-7):
#     """
#     Calculate the prob IoU between oriented bounding boxes, https://arxiv.org/pdf/2106.06072v1.pdf.

#     Args:
#         obb1 (torch.Tensor | np.ndarray): A tensor of shape (N, 5) representing ground truth obbs, with xywhr format.
#         obb2 (torch.Tensor | np.ndarray): A tensor of shape (M, 5) representing predicted obbs, with xywhr format.
#         eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

#     Returns:
#         (torch.Tensor): A tensor of shape (N, M) representing obb similarities.
#     """
#     obb1 = torch.from_numpy(obb1) if isinstance(obb1, np.ndarray) else obb1 
#     obb2 = torch.from_numpy(obb2) if isinstance(obb2, np.ndarray) else obb2

#     x1, y1 = obb1[..., :2].split(1, dim=-1)                                
#     x2, y2 = (x.squeeze(-1)[None] for x in obb2[..., :2].split(1, dim=-1))  
#     a1, b1, c1 = _get_covariance_matrix(obb1)                              
#     a2, b2, c2 = (x.squeeze(-1)[None] for x in _get_covariance_matrix(obb2))

#     t1 = (((a1 + a2) * (y1 - y2).pow(2) + (b1 + b2) * (x1 - x2).pow(2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)) * 0.25
#     t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)) * 0.5
#     t3 = (((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2)) / (4 * ((a1 * b1 - c1.pow(2)).clamp_(0) * (a2 * b2 - c2.pow(2)).clamp_(0)).sqrt() + eps) + eps).log() * 0.5
#     bd = (t1 + t2 + t3).clamp(eps, 100.0)   
#     hd = (1.0 - (-bd).exp() + eps).sqrt()
#     return 1 - hd

# def batch_probiou_np(obb1, obb2, eps=1e-7):
#     x1, y1 = obb1[..., 0:1], obb1[..., 1:2]
#     x2, y2 = obb2[..., 0][None, ...], obb2[..., 1][None, ...]
    
#     a1, b1, c1 = _get_covariance_matrix_np(obb1)  
#     a2_raw, b2_raw, c2_raw = _get_covariance_matrix_np(obb2)
#     a2 = np.squeeze(a2_raw, axis=-1)[None, ...]
#     b2 = np.squeeze(b2_raw, axis=-1)[None, ...]
#     c2 = np.squeeze(c2_raw, axis=-1)[None, ...]

#     t1 = (((a1 + a2) * (y1 - y2)**2 + (b1 + b2) * (x1 - x2)**2 ) / ((a1 + a2) * (b1 + b2) - (c1 + c2)**2 + eps)) * 0.25
#     t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2)**2 + eps)) * 0.5  
#     term1 = np.clip(a1 * b1 - c1**2, 0, None) 
#     term2 = np.clip(a2 * b2 - c2**2, 0, None)
#     t3 = np.log(((a1 + a2) * (b1 + b2) - (c1 + c2)**2) / (4 * np.sqrt(term1 * term2) + eps) + eps) * 0.5
#     bd = np.clip(t1 + t2 + t3, eps, 100.0)
#     hd = np.sqrt(1.0 - np.exp(-bd) + eps)    
#     return 1 - hd


# def _get_covariance_matrix(boxes):
#     """
#     Generating covariance matrix from obbs.

#     Args:
#         boxes (torch.Tensor): A tensor of shape (N, 5) representing rotated bounding boxes, with xywhr format.

#     Returns:
#         (torch.Tensor): Covariance matrices corresponding to original rotated bounding boxes.
#     """
#     # Gaussian bounding boxes, ignore the center points (the first two columns) because they are not needed here.
#     gbbs = torch.cat((boxes[:, 2:4].pow(2) / 12, boxes[:, 4:]), dim=-1)
#     a, b, c = gbbs.split(1, dim=-1)
#     cos = c.cos()
#     sin = c.sin()
#     cos2 = cos.pow(2)
#     sin2 = sin.pow(2)
#     return a * cos2 + b * sin2, a * sin2 + b * cos2, (a - b) * cos * sin

# def _get_covariance_matrix_np(boxes):
#     gbbs = np.concatenate([boxes[:, 2:4]**2 / 12.0,  boxes[:, 4:]], axis=-1)
#     a, b, c = np.split(gbbs, 3, axis=-1)
#     cos = np.cos(c)
#     sin = np.sin(c)
#     cos2 = cos **2
#     sin2 = sin** 2
#     return  a * cos2 + b * sin2, a * sin2 + b * cos2, (a - b) * cos * sin


# def regularize_rboxes(rboxes):
#     """
#     Regularize rotated boxes in range [0, pi/2].

#     Args:
#         rboxes (torch.Tensor): Input boxes of shape(N, 5) in xywhr format.

#     Returns:
#         (torch.Tensor): The regularized boxes.
#     """
#     x, y, w, h, t = rboxes.unbind(dim=-1)
#     # Swap edge and angle if h >= w
#     w_ = torch.where(w > h, w, h)
#     h_ = torch.where(w > h, h, w)
#     t = torch.where(w > h, t, t + math.pi / 2) % math.pi
#     return torch.stack([x, y, w_, h_, t], dim=-1)  # regularized boxes

# def regularize_rboxes_np(rboxes_np):
#     x, y, w, h, t = rboxes_np[..., 0], rboxes_np[..., 1], rboxes_np[..., 2], rboxes_np[..., 3], rboxes_np[..., 4]
#     w_ = np.where(w > h, w, h) 
#     h_ = np.where(w > h, h, w) 
#     t = np.where(w > h, t, t + np.pi / 2) % np.pi
#     return np.stack([x, y, w_, h_, t], axis=-1)


# def xywhr2xyxyxyxy(x):
#     """
#     Convert batched Oriented Bounding Boxes (OBB) from [xywh, rotation] to [xy1, xy2, xy3, xy4]. Rotation values should
#     be in radians from 0 to pi/2.

#     Args:
#         x (numpy.ndarray | torch.Tensor): Boxes in [cx, cy, w, h, rotation] format of shape (n, 5) or (b, n, 5).

#     Returns:
#         (numpy.ndarray | torch.Tensor): Converted corner points of shape (n, 4, 2) or (b, n, 4, 2).
#     """
#     cos, sin, cat, stack = (
#         (torch.cos, torch.sin, torch.cat, torch.stack)
#         if isinstance(x, torch.Tensor)
#         else (np.cos, np.sin, np.concatenate, np.stack)
#     )

#     ctr = x[..., :2]
#     w, h, angle = (x[..., i : i + 1] for i in range(2, 5))
#     cos_value, sin_value = cos(angle), sin(angle)
#     vec1 = [w / 2 * cos_value, w / 2 * sin_value]
#     vec2 = [-h / 2 * sin_value, h / 2 * cos_value]
#     vec1 = cat(vec1, -1)
#     vec2 = cat(vec2, -1)
#     pt1 = ctr + vec1 + vec2
#     pt2 = ctr + vec1 - vec2
#     pt3 = ctr - vec1 - vec2
#     pt4 = ctr - vec1 + vec2
#     return stack([pt1, pt2, pt3, pt4], -2)