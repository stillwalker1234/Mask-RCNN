import argparse
import numpy as np
import cv2
import cPickle
import heapq
import os
import math
import tensorflow as tf
import matplotlib.pyplot as plt

from .config import cfg, get_output_dir

from ..utils.timer import Timer
from ..utils.cython_nms import nms, nms_new
from ..utils.blob import im_list_to_blob
from ..utils.boxes_grid import get_boxes_grid

from ..fast_rcnn.bbox_transform import clip_boxes, bbox_transform_inv


def _get_image_blob(im):
    """Converts an image into a network input.
    Arguments:
        im (ndarray): a color image in BGR order
    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale_factors (list): list of image scales (relative to im) used
            in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)


def _get_rois_blob(im_rois, im_scale_factors):
    """Converts RoIs into network inputs.
    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        im_scale_factors (list): scale factors as returned by _get_image_blob
    Returns:
        blob (ndarray): R x 5 matrix of RoIs in the image pyramid
    """
    rois, levels = _project_im_rois(im_rois, im_scale_factors)
    rois_blob = np.hstack((levels, rois))
    return rois_blob.astype(np.float32, copy=False)


def _project_im_rois(im_rois, scales):
    """Project image RoIs into the image pyramid built by _get_image_blob.
    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        scales (list): scale factors as returned by _get_image_blob
    Returns:
        rois (ndarray): R x 4 matrix of projected RoI coordinates
        levels (list): image pyramid levels used by each projected RoI
    """
    im_rois = im_rois.astype(np.float, copy=False)
    scales = np.array(scales)

    if len(scales) > 1:
        widths = im_rois[:, 2] - im_rois[:, 0] + 1
        heights = im_rois[:, 3] - im_rois[:, 1] + 1

        areas = widths * heights
        scaled_areas = areas[:, np.newaxis] * (scales[np.newaxis, :] ** 2)
        diff_areas = np.abs(scaled_areas - 224 * 224)
        levels = diff_areas.argmin(axis=1)[:, np.newaxis]
    else:
        levels = np.zeros((im_rois.shape[0], 1), dtype=np.int)

    rois = im_rois * scales[levels]

    return rois, levels


def _get_blobs(im, rois=None):
    """Convert an image and RoIs within that image into network inputs."""
    if cfg.TEST.HAS_RPN:
        blobs = {'data': None, 'rois': None}
        blobs['data'], im_scale_factors = _get_image_blob(im)
    else:
        blobs = {'data': None, 'rois': None}
        blobs['data'], im_scale_factors = _get_image_blob(im)
        if cfg.IS_MULTISCALE:
            if cfg.IS_EXTRAPOLATING:
                blobs['rois'] = _get_rois_blob(rois, cfg.TEST.SCALES)
            else:
                blobs['rois'] = _get_rois_blob(rois, cfg.TEST.SCALES_BASE)
        else:
            blobs['rois'] = _get_rois_blob(rois, cfg.TEST.SCALES_BASE)

    return blobs, im_scale_factors


def _clip_boxes(boxes, im_shape):
    """Clip boxes to image boundaries."""
    # x1 >= 0
    boxes[:, 0::4] = np.maximum(boxes[:, 0::4], 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(boxes[:, 1::4], 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.minimum(boxes[:, 2::4], im_shape[1] - 1)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.minimum(boxes[:, 3::4], im_shape[0] - 1)
    return boxes


def _rescale_boxes(boxes, inds, scales):
    """Rescale boxes according to image rescaling."""

    for i in xrange(boxes.shape[0]):
        boxes[i, :] = boxes[i, :] / scales[int(inds[i])]

    return boxes


def im_detect(sess, net, im, net_outputs=None):
    """Detect object classes in an image given object proposals.
    Arguments:
        net (caffe.Net): Fast R-CNN network to use
        im (ndarray): color image to test (in BGR order)
        boxes (ndarray): R x 4 array of object proposals
    Returns:
        labels (ndarray): R labels
        dets: R x 5 [x1, y1, x2, y2, score]
        masks: R x H x W binary masks
    """

    blobs, im_scales = _get_blobs(im)

    # When mapping from image ROIs to feature map ROIs, there's some aliasing
    # (some distinct image ROIs get mapped to the same feature ROI). Here, we identify duplicate feature ROIs, so we only compute features on the unique subset.
    if cfg.DEDUP_BOXES > 0 and not cfg.TEST.HAS_RPN:
        v = np.array([1, 1e3, 1e6, 1e9, 1e12])
        hashes = np.round(blobs['rois'] * cfg.DEDUP_BOXES).dot(v)
        _, index, inv_index = np.unique(hashes, return_index=True, return_inverse=True)
        blobs['rois'] = blobs['rois'][index, :]
        boxes = boxes[index, :]

    if cfg.TEST.HAS_RPN:
        im_blob = blobs['data']
        blobs['im_info'] = np.array(
            [[im_blob.shape[1], im_blob.shape[2], im_scales[0]]],
            dtype=np.float32)
    # forward pass
    if cfg.TEST.HAS_RPN:
        feed_dict = {
            net.data: blobs['data'], net.im_info: blobs['im_info'], net.keep_prob: 1.0}
    else:
        feed_dict = {net.data: blobs['data'],
                     net.rois: blobs['rois'], net.keep_prob: 1.0}

    labels, dets, mask_pred = sess.run(net_outputs, feed_dict=feed_dict)

    R = labels.shape[0]

    im_info = blobs['im_info'].flatten()
    masks = np.zeros((R, int(im_info[0]), int(im_info[1])), dtype=np.uint8)

    for i in range(R):
        x0, y0, x1, y1 = [int(j) for j in dets[i,:4]]
        h, w = y1 - y0 + 1, x1 - x0 + 1

        mask_resize = cv2.resize(mask_pred[i], (w,h))
        masks[i, y0:y1+1, x0:x1+1] = mask_resize > 0.5

    return labels, dets, masks


def vis_detections(im, class_name, dets, thresh=0.8):
    """Visual debugging of detections."""
    import matplotlib.pyplot as plt
    # im = im[:, :, (2, 1, 0)]
    for i in xrange(np.minimum(10, dets.shape[0])):
        bbox = dets[i, :4]
        score = dets[i, -1]
        if score > thresh:
            # plt.cla()
            # plt.imshow(im)
            plt.gca().add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1], fill=False,
                              edgecolor='g', linewidth=3)
            )
            plt.gca().text(bbox[0], bbox[1] - 2,
                           '{:s} {:.3f}'.format(class_name, score),
                           bbox=dict(facecolor='blue', alpha=0.5),
                           fontsize=14, color='white')

            plt.title('{}  {:.3f}'.format(class_name, score))
    # plt.show()


def apply_nms(all_boxes, thresh):
    """Apply non-maximum suppression to all predicted boxes output by the
    test_net method.
    """
    num_classes = len(all_boxes)
    num_images = len(all_boxes[0])
    nms_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(num_classes)]
    for cls_ind in xrange(num_classes):
        for im_ind in xrange(num_images):
            dets = all_boxes[cls_ind][im_ind]
            if dets == []:
                continue

            x1 = dets[:, 0]
            y1 = dets[:, 1]
            x2 = dets[:, 2]
            y2 = dets[:, 3]
            scores = dets[:, 4]
            inds = np.where((x2 > x1) & (y2 > y1) & (
                scores > cfg.TEST.DET_THRESHOLD))[0]
            dets = dets[inds, :]
            if dets == []:
                continue

            keep = nms(dets, thresh)
            if len(keep) == 0:
                continue
            nms_boxes[cls_ind][im_ind] = dets[keep, :].copy()
    return nms_boxes


def test_net(sess, net, imdb, weights_filename, max_per_image=300, thresh=0.05, vis=False):
    """Test a Fast R-CNN network on an image database."""
    num_images = len(imdb.image_index)
    
    net_outputs = net.get_test_outputs()

    json_doc = []

    for i in xrange(num_images):
        im = cv2.imread(imdb.image_path_at(i))
        labels, dets, mask_pred = im_detect(sess, net, im, net_outputs=net_outputs)

        json_doc += imdb.get_json_str(i, labels, dets, mask_pred)

    output_dir = get_output_dir(imdb, weights_filename)

    imdb.write_res_json(json_doc, output_dir, evaluate=True)
