import tensorflow as tf


def roi_align(images, rois_, pooled_height, pooled_width, stride, name="roi_align"):
    with tf.name_scope(name):
        rois = rois_[:, 1:]
        batch_inds = tf.cast(rois_[:, 0], tf.int32)
        rois = rois * (stride + 0.0)
        rois = normalize_and_swap_xy(rois, tf.shape(images))

        with tf.control_dependencies([images, batch_inds]):
            image_crops = tf.image.crop_and_resize(images, rois, batch_inds, [pooled_height, pooled_width], method='bilinear', name='resize')

        return image_crops, None


def roi_align_max_pooled(images, rois_, pooled_height, pooled_width, stride, name="roi_align"):
    with tf.name_scope(name):
        num_locations = 2

        rois = rois_[:, 1:]

        N = tf.shape(rois)[0]
        # (x_2 - x_1) / num_bins x-axis
        x_deltas = (rois[:, 2] - rois[:, 0]) / pooled_width

        # (y_1 - y_2) / num-bins y-axis
        y_deltas = (rois[:, 3] - rois[:, 1]) / pooled_height

        x_1 = tf.cast(tf.range(pooled_width), tf.float32)
        x_2 = tf.cast(tf.range(pooled_width), tf.float32) + 1

        x_1 = tf.tile(tf.expand_dims(x_1, 0), (N, 1)) * tf.expand_dims(x_deltas, 1) + tf.expand_dims(rois[:, 0], 1)  # (N, #bins-width)
        x_2 = tf.tile(tf.expand_dims(x_2, 0), (N, 1)) * tf.expand_dims(x_deltas, 1) + tf.expand_dims(rois[:, 0], 1)  # (N, #bins-width)

        y_1 = tf.cast(tf.range(pooled_width), tf.float32)
        y_2 = tf.cast(tf.range(pooled_width), tf.float32) + 1

        y_1 = tf.tile(tf.expand_dims(y_1, 0), (N, 1)) * tf.expand_dims(y_deltas, 1) + tf.expand_dims(rois[:, 1], 1)  # (N, #bins-height)
        y_2 = tf.tile(tf.expand_dims(y_2, 0), (N, 1)) * tf.expand_dims(y_deltas, 1) + tf.expand_dims(rois[:, 1], 1)  # (N, #bins-height)

        x_1 = tf.tile(tf.expand_dims(x_1, 1), (1, pooled_height, 1))  # (N, #bins-height, #bins-width)
        x_2 = tf.tile(tf.expand_dims(x_2, 1), (1, pooled_height, 1))  # (N, #bins-height, #bins-width)

        y_1 = tf.tile(tf.expand_dims(y_1, -1), (1, 1, pooled_width))  # (N, #bins-height, #bins-width)
        y_2 = tf.tile(tf.expand_dims(y_2, -1), (1, 1, pooled_width))  # (N, #bins-height, #bins-width)

        rois = tf.stack([x_1, y_1, x_2, y_2], axis=3)
        rois = tf.reshape(rois, (-1, 4))

        batch_inds_ = tf.tile(tf.expand_dims(tf.cast(rois_[:, 0], tf.int32), 0), (pooled_height*pooled_width, 1))
        batch_inds_ = tf.transpose(batch_inds_, [1, 0])
        batch_inds = tf.reshape(batch_inds_, (-1,))
        
        rois = rois * (stride + 0.0)
        
        rois = normalize_and_swap_xy(rois, tf.shape(images))
        
        with tf.control_dependencies([images, batch_inds]):
            image_crops = tf.image.crop_and_resize(
                images,
                rois,
                batch_inds,
                [num_locations, num_locations],
                method='bilinear',
                name='resize'
                )  # (N*pooled_width*pooled_height,2,2,1)

        max_pooled_image_crops = tf.reduce_mean(tf.reshape(image_crops, (-1, 4, tf.shape(images)[-1])), axis=1)
        max_pooled_image_crops = tf.reshape(max_pooled_image_crops, (N, pooled_height, pooled_width, image_crops.shape[-1].value))

        return max_pooled_image_crops, None


def normalize_and_swap_xy(rois, shape):
    rois = tf.reshape(rois, [-1, 2])  # to (x, y)
    xs = rois[:, 0]
    ys = rois[:, 1]
    xs = xs / tf.cast(shape[2]-1, tf.float32)
    ys = ys / tf.cast(shape[1]-1, tf.float32)
    rois = tf.concat([ys[:, tf.newaxis], xs[:, tf.newaxis]], axis=1)
    rois = tf.reshape(rois, [-1, 4])  # to (y1, x1, y2, x2)

    return rois

