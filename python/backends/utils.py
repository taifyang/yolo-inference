'''
Author: taifyang
Date: 2024-06-12 22:23:07
LastEditTime: 2026-01-12 20:57:57
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
try:
    import torch
except:
    print('torch import failed!')
from backends.yolo import *


'''
description:            image centercrop process
param {*} image         input image
param {*} inputs_shape  input shape
param {*} use_cupy      use cupy
return {*}              processed image
'''
def centercrop(image, inputs_shape, use_cupy=False):
    crop_size = min(image.shape[0], image.shape[1])
    left = (image.shape[1] - crop_size) // 2
    top = (image.shape[0] - crop_size) // 2
    if use_cupy:
        crop_image = cupy.asarray(image)[top:(top+crop_size), left:(left+crop_size), ...]
        zoom_factors = (inputs_shape[0]/crop_image.shape[0], inputs_shape[1]/crop_image.shape[1], 1) 
        return ndimage.zoom(crop_image, zoom_factors, order=0)
    else:
        crop_image = image[top:(top+crop_size), left:(left+crop_size), ...]
        return cv2.resize(crop_image, inputs_shape)


'''
description:            image normalize process
param {*} image         input image
param {*} algo_type     algorithm type
param {*} use_cupy      use cupy
return {*}              processed image
'''
def normalize(image, algo_type, use_cupy=False):
    if use_cupy:
        image = image.astype(cupy.float32) / 255.0
    else:
        image = image / 255.0
        
    if algo_type in ['YOLOv5']:
        if use_cupy:
            image = image - cupy.asarray([0.406, 0.456, 0.485], dtype=cupy.float32).reshape(1, 1, -1)
            image = image / cupy.asarray([0.225, 0.224, 0.229], dtype=cupy.float32).reshape(1, 1, -1)
        else:
            image = image - np.array([0.406, 0.456, 0.485])
            image = image / np.array([0.225, 0.224, 0.229])
    return image


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
param {*} use_cupy  use cupy
return {*}          output image
'''
def letterbox(im, new_shape=(416, 416), color=(114, 114, 114), use_cupy=False):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    
    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))    
    dw, dh = (new_shape[1] - new_unpad[0])/2, (new_shape[0] - new_unpad[1])/2  # wh padding 
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    
    # resize
    if shape[::-1] != new_unpad:  
        if use_cupy:
            im_cupy = cupy.asarray(im)
            zoom_factors = (new_unpad[1]/im.shape[0], new_unpad[0]/im.shape[1], 1) 
            im_cupy = ndimage.zoom(im_cupy, zoom_factors, order=0)
        else:
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    # add border
    if use_cupy:
        return cupy.pad(array=im_cupy, pad_width=((top, bottom), (left, right), (0, 0)), mode='constant', constant_values=(color, color))
    else:
        return cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color) 


'''
description:            scale boxes
param {*} boxes         bounding boxes
param {*} input_shape   input image shape
param {*} output_shape  output image shape
param {*} xywh          if xywh
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
    if boxes.shape[1] == 57:
        for kid in range(2, 19):
            boxes[:, kid * 3] = (boxes[:, kid * 3] - pad[0]) / gain
            boxes[:, kid * 3 + 1]  = (boxes[:, kid * 3 + 1] -  pad[1]) / gain
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
description:            	plot skeleton keypoints
param {*} im          		input image
param {*} kpt   			keypoints
param {*} score_threshold  	score threshold
return {*}              
'''
def plot_skeleton_kpts(im, kpt, score_threshold=0.5):
    num_kpts = len(kpt) // 3 
    for kid in range(num_kpts):
        x_coord, y_coord, conf = kpt[3 * kid], kpt[3 * kid + 1], kpt[3 * kid + 2]
        if conf > score_threshold:  
            cv2.circle(im, (int(x_coord), int(y_coord)), 5, (255, 0, 0), -1)

    skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
                [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
                [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
    
    for sk_id, sk in enumerate(skeleton):
        pos1 = (int(kpt[(sk[0]-1)*3]), int(kpt[(sk[0]-1)*3+1]))
        pos2 = (int(kpt[(sk[1]-1)*3]), int(kpt[(sk[1]-1)*3+1]))
        conf1 = kpt[(sk[0]-1)*3+2]
        conf2 = kpt[(sk[1]-1)*3+2]
        if conf1 > score_threshold and conf2 > score_threshold:  
            cv2.line(im, pos1, pos2, (255, 0, 0), thickness=2)


'''
description:            NMS for oriented bounding boxes using probiou and fast-nms
param {*} boxes         Rotated bounding boxes, format xywhr
param {*} scores   		Confidence scores, shape (N,)
param {*} threshold  	IoU threshold. Defaults to 0.45
return {*}              Indices of boxes to keep after NMS     
'''
def nms_rotated(boxes, scores, threshold=0.45):
    if isinstance(boxes, np.ndarray):
        sorted_idx = np.argsort(-scores)
        boxes = boxes[sorted_idx]
        ious = np.triu(probiou(boxes, boxes), k=1)
        pick = np.nonzero(np.max(ious, axis=0) < threshold)[0]
    else:
        sorted_idx = torch.argsort(scores, descending=True)  
        boxes = boxes[sorted_idx]
        ious = probiou(boxes, boxes).triu_(diagonal=1)
        pick = torch.nonzero(ious.max(dim=0)[0] < threshold).squeeze_(-1)
    return sorted_idx[pick]


'''
description:    Calculate the prob IoU between oriented bounding boxes, https://arxiv.org/pdf/2106.06072v1.pdf
param {*} obb1  ground truth obbs with xywhr format
param {*} obb2  predicted obbs with xywhr format
param {*} eps   A small value to avoid division by zero. Defaults to 1e-7
return {*}      A tensor of shape (N, M) representing obb similarities   
'''
def probiou(obb1, obb2, eps=1e-7):
    if isinstance(obb1, np.ndarray) :
        x1, y1 = obb1[..., 0:1], obb1[..., 1:2]
        x2, y2 = obb2[..., 0][None, ...], obb2[..., 1][None, ...]
        
        a1, b1, c1 = _get_covariance_matrix(obb1)  
        a2_raw, b2_raw, c2_raw = _get_covariance_matrix(obb2)
        a2 = np.squeeze(a2_raw, axis=-1)[None, ...]
        b2 = np.squeeze(b2_raw, axis=-1)[None, ...]
        c2 = np.squeeze(c2_raw, axis=-1)[None, ...]

        t1 = (((a1 + a2) * (y1 - y2)**2 + (b1 + b2) * (x1 - x2)**2 ) / ((a1 + a2) * (b1 + b2) - (c1 + c2)**2 + eps)) * 0.25
        t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2)**2 + eps)) * 0.5  
        term1 = np.clip(a1 * b1 - c1**2, 0, None) 
        term2 = np.clip(a2 * b2 - c2**2, 0, None)
        t3 = np.log(((a1 + a2) * (b1 + b2) - (c1 + c2)**2) / (4 * np.sqrt(term1 * term2) + eps) + eps) * 0.5
        bd = np.clip(t1 + t2 + t3, eps, 100.0)
        hd = np.sqrt(1.0 - np.exp(-bd) + eps) 
    else: 
        x1, y1 = obb1[..., :2].split(1, dim=-1)                                
        x2, y2 = (x.squeeze(-1)[None] for x in obb2[..., :2].split(1, dim=-1))  
        a1, b1, c1 = _get_covariance_matrix(obb1)                              
        a2, b2, c2 = (x.squeeze(-1)[None] for x in _get_covariance_matrix(obb2))

        t1 = (((a1 + a2) * (y1 - y2).pow(2) + (b1 + b2) * (x1 - x2).pow(2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)) * 0.25
        t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)) * 0.5
        t3 = (((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2)) / (4 * ((a1 * b1 - c1.pow(2)).clamp_(0) * (a2 * b2 - c2.pow(2)).clamp_(0)).sqrt() \
                                                            + eps) + eps).log() * 0.5
        bd = (t1 + t2 + t3).clamp(eps, 100.0)   
        hd = (1.0 - (-bd).exp() + eps).sqrt()
    return 1 - hd


'''
description:    Generating covariance matrix from obbs
param {*} boxes rotated bounding boxes, with xywhr format
return {*}      Covariance matrices corresponding to original rotated bounding boxes
'''
def _get_covariance_matrix(boxes):
    # Gaussian bounding boxes, ignore the center points (the first two columns) because they are not needed here.
    if isinstance(boxes, np.ndarray):                                            
        gbbs = np.concatenate([boxes[:, 2:4]**2 / 12.0,  boxes[:, -1:]], axis=-1)    
        a, b, c = np.split(gbbs, 3, axis=-1)
        cos = np.cos(c)
        sin = np.sin(c)
        cos2 = cos **2
        sin2 = sin** 2
    else:
        gbbs = torch.cat((boxes[:, 2:4].pow(2) / 12, boxes[:, -1:]), dim=-1)
        a, b, c = gbbs.split(1, dim=-1)
        cos = c.cos()
        sin = c.sin()
        cos2 = cos.pow(2)
        sin2 = sin.pow(2)
    return a * cos2 + b * sin2, a * sin2 + b * cos2, (a - b) * cos * sin

'''
description:    Regularize rotated bounding boxes to range [0, pi/2]
param {*} boxes rotated bounding boxes 
return {*}      regularized rotated bounding boxes
'''
def regularize_rboxes(rboxes):
    if isinstance(rboxes, np.ndarray):
        x, y, w, h, score, cls, t = rboxes[..., 0], rboxes[..., 1], rboxes[..., 2], rboxes[..., 3], rboxes[..., 4], rboxes[..., 5], rboxes[..., 6]
        w_ = np.where(w > h, w, h) 
        h_ = np.where(w > h, h, w) 
        t = np.where(w > h, t, t + np.pi / 2) % np.pi
        return np.stack([x, y, w_, h_, score, cls, t], axis=-1)
    else:
        x, y, w, h, score, cls, t  = rboxes.unbind(dim=-1)
        w_ = torch.where(w > h, w, h)
        h_ = torch.where(w > h, h, w)
        t = torch.where(w > h, t, t + math.pi / 2) % math.pi
        return torch.stack([x, y, w_, h_, score, cls, t], dim=-1) 


'''
description:    Convert batched Oriented Bounding Boxes (OBB) from [xywh, rotation] to [xy1, xy2, xy3, xy4]
param {*} x     Boxes in [cx, cy, w, h, rotation] format of shape (n, 5) or (b, n, 5)t
return {*}      Converted corner points of shape (n, 4, 2) or (b, n, 4, 2
'''
def xywhr2xyxyxyxy(x):
    cos, sin, cat, stack = (
        (torch.cos, torch.sin, torch.cat, torch.stack)
        if isinstance(x, torch.Tensor)
        else (np.cos, np.sin, np.concatenate, np.stack)
    )

    ctr = x[..., :2]
    w, h, = (x[..., i : i + 1] for i in range(2, 4))
    angle =  x[..., -1] 
    cos_value, sin_value = cos(angle), sin(angle)
    vec1 = [w / 2 * cos_value, w / 2 * sin_value]
    vec2 = [-h / 2 * sin_value, h / 2 * cos_value]
    vec1 = cat(vec1, -1)
    vec2 = cat(vec2, -1)
    pt1 = ctr + vec1 + vec2
    pt2 = ctr + vec1 - vec2
    pt3 = ctr - vec1 - vec2
    pt4 = ctr - vec1 + vec2
    return stack([pt1, pt2, pt3, pt4], -2)

'''
description:    draw result
param {*} image input image
param {*} preds prediction result
param {*} masks masks
param {*} kpts  keypoints
return {*}      output image
'''
def draw_result(image, preds, masks=[], kpts=None):
    image_copy = image.copy()   
    boxes = preds[..., :4] 
    scores = preds[..., 4]
    classes = preds[..., 5].astype(np.int32)
    
    for mask in masks:
        image_copy[mask] = [np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256)]
    result = (image*0.5 + image_copy*0.5).astype(np.uint8)
    
    if preds.shape[1] == 7:
        boxes = np.concatenate((boxes, preds[..., -1:]), axis=1)   
        for box, score, cls in zip(boxes, scores, classes):
            box = xywhr2xyxyxyxy(box).astype(np.int32)
            cv2.polylines(result, [np.asarray(box)], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.putText(result, 'class:{0} score:{1:.2f}'.format(cls, score), (box[0][0], box[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    elif kpts is not None:
        for box, score, cls, kpt in zip(boxes, scores, classes, kpts):
            box = box.astype(np.int32)
            cv2.rectangle(result, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv2.putText(result, 'class:{0} score:{1:.2f}'.format(cls, score), (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            plot_skeleton_kpts(result, kpt)
    else:
        for box, score, cls in zip(boxes, scores, classes):
            box = box.astype(np.int32)
            cv2.rectangle(result, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv2.putText(result, 'class:{0} score:{1:.2f}'.format(cls, score), (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)       
    return result