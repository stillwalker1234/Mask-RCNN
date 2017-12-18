import numpy as np
import cv2
import sys
import os
import matplotlib.pyplot as plt

from .config import cfg, get_output_dir
from ..utils.blob import im_list_to_blob

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


def im_detect(sess, net, im, net_outputs=None, use_mask=True):
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

    if use_mask:
        labels, dets, mask_pred = sess.run(net_outputs, feed_dict=feed_dict)
        dets[:, :4] /= im_scales

        R = labels.shape[0]

        masks = np.zeros((R, int(im.shape[0]), int(im.shape[1])), dtype=np.uint8)

        for i in range(R):
            x0, y0, x1, y1 = [int(j) for j in dets[i,:4]]
            h, w = y1 - y0 + 1, x1 - x0 + 1

            mask_resize = cv2.resize(mask_pred[i], (w,h))
            masks[i, y0:y1+1, x0:x1+1] = mask_resize > 0.5

        return labels, dets, masks
    else:
        labels, dets = sess.run(net_outputs, feed_dict=feed_dict)

        dets[:, :4] /= im_scales

        return labels, dets


def _mask_to_image_layer(mask):
    h, w = mask.shape
    im = np.ones((h, w, 3))
    color_mask = np.random.random((1, 3)).tolist()[0]
    mask = mask.astype('float32')

    for i in range(3):
        im[:, :, i] = color_mask[i]

    return np.dstack((im, mask*0.5))


def save_image(output_dir, img_name, im, dets, masks, labels, thresh=0.75):
    def to_xywh(x1, y1, x2, y2):
        return [x1, y1, x2 - x1 + 1, y2 - y1 + 1]
    
    fig = plt.figure()
    ax = plt.gca()
    ax.imshow(im)

    boxes = dets[:, :4]
    scores = dets[:, 4]

    for i in range(len(labels)):
        if scores[i] > thresh:
            x, y, w, h = to_xywh(*boxes[i])

            ax.add_patch(
                plt.Rectangle((x, y), w, h, fill=False, edgecolor='g', linewidth=2)
            )
            ax.text(
                x, y-2, '{:s} {:.3f}'.format(str(labels[i]), scores[i]), bbox=dict(facecolor='blue', alpha=0.5), fontsize=8, color='white'
            )
            ax.imshow(_mask_to_image_layer(masks[i]))

    plt.axis("off")
    plt.savefig(output_dir + '/' + img_name, bbox_inches='tight')
    plt.close()


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
    
    net_outputs = net.get_test_outputs(get_mask=cfg.TEST.USE_MASK)

    json_doc = []
    
    output_dir = get_output_dir(imdb, weights_filename)
    names = imdb.get_class_names()

    img_dir = output_dir + '/imgs'
    if not os.path.exists(img_dir):
        os.mkdir(img_dir)

    with sess:
        for i in range(num_images):
            sys.stdout.write("\r%i, %s" % (i, imdb.image_path_at(i)))
            sys.stdout.flush()
            
            if "000000035899" in  imdb.image_path_at(i):
                continue

            im = cv2.imread(imdb.image_path_at(i))
            try:
                if cfg.TEST.USE_MASK:
                    labels, dets, mask_pred = im_detect(sess, net, im, net_outputs=net_outputs, use_mask=True)
                else:
                    labels, dets = im_detect(sess, net, im, net_outputs=net_outputs, use_mask=False)
                    mask_pred = None

                if cfg.TEST.DO_IMG_SAVE and cfg.TEST.USE_MASK:
                    save_image(img_dir, "img_%i.png" % (i+1), im, dets, mask_pred, [names[j] for j in labels])

                json_doc += imdb.get_json_str(i, labels, dets, mask_pred)
            except Exception:
                print("image %s, failed" % imdb.image_path_at(i))

    if cfg.TEST.USE_MASK:
        eval_metrics = ['segm', 'bbox']
    else:
        eval_metrics = ['bbox']

    imdb.write_res_json(json_doc, output_dir, evaluate=True, eval_metrics=eval_metrics)
