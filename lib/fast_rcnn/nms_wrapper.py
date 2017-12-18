# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import numpy as np
from .config import cfg
from ..nms.gpu_nms import gpu_nms
from ..nms.cpu_nms import cpu_nms


def nms(dets, thresh, force_cpu=False):
    """Dispatch to either CPU or GPU NMS implementations."""

    if dets.shape[0] == 0:
        return []
    if cfg.USE_GPU_NMS and not force_cpu:
        return gpu_nms(dets, thresh, device_id=cfg.GPU_ID)
    else:
        return cpu_nms(dets, thresh)


def nms_wrapper(scores, boxes, labels_map, mask_pred, mask_boxes, threshold=0.5, class_sets=None, cls_names=None):
    """
    post-process the results of im_detect
    :param scores: N * (K * 4) numpy
    :param boxes: N * K numpy
    :param class_sets: e.g. CLASSES = ('__background__','person','bike','motorbike','car','bus')
    :return: a list of K-1 dicts, no background, each is {'class': classname, 'dets': None | [[x1,y1,x2,y2,score],...]}
    """
    num_class = scores.shape[1] if class_sets is None else len(class_sets)
    assert num_class * 4 == boxes.shape[1],\
        'Detection scores and boxes dont match'
    class_sets = [('class_' + str(i) if cls_names is None else cls_names[i])
                  for i in range(0, num_class)] if class_sets is None else class_sets

    labels_map_inv = {labels_map[i]:i for i in range(labels_map.size)}

    print(labels_map_inv, mask_pred.shape)

    res = []
    for ind, cls in enumerate(class_sets[1:]):
        ind += 1  # skip background

        # select relevant class and concat
        cls_boxes = boxes[:, 4 * ind: 4 * (ind + 1)]
        cls_scores = scores[:, ind]
        cls_mask = mask_pred[:,:,:,ind]
        dets = np.hstack(
            (cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)

        # do nms
        keep_mask = np.zeros((dets.shape[0]), dtype=np.bool)
        keep = nms(dets, thresh=0.3)
        keep_mask[keep] = True
        keep_mask[np.where(dets[:, 4] <= threshold)[0]] = False
        keep = np.where(keep_mask)[0]

        # filter
        dets = dets[keep, :]

        cls_mask_ = np.zeros((keep.shape[0], 28, 28))
        cls_boxes_mask = np.zeros((keep.shape[0], 4))
        for i, keep_idx in enumerate(keep):
            if keep_idx in labels_map_inv:
                cls_mask_[i] = cls_mask[labels_map_inv[keep_idx]]
                cls_boxes_mask[i] = mask_boxes[labels_map_inv[keep_idx]]
        
        # push to output container
        r = {}
        if dets.shape[0] > 0:
            r['class'], r['dets'], r['mask'], r['mask_rois'] = cls, dets, cls_mask_, cls_boxes_mask
        else:
            r['class'], r['dets'], r['mask'], r['mask_rois'] = cls, None, None, None
        res.append(r)
    return res
