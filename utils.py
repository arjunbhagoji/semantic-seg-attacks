import numpy as np
import tensorflow as tf


def gen_mask(input_size, BATCH_SIZE):
    h = input_size[0]
    w = input_size[1]
    mask = np.zeros((h,w))
    mask_rgb = np.zeros((h,w,3))
    row_c, col_c = h/2, w/2
    r = 15
    mask[row_c-r:row_c+r, col_c-r:col_c+r] = 1.0
    for i in range(3):
        mask_rgb[:,:,i] = mask
    # (np.tile(mask,3)).reshape(h, w, 3)
    # mask_rgb = (np.tile(mask_rgb,BATCH_SIZE)).reshape(BATCH_SIZE, h, w, 3)
    mask_rgb = mask_rgb.reshape(BATCH_SIZE, h, w, 3)
    mask_t = tf.convert_to_tensor(mask_rgb, dtype=tf.float32)

    return mask_t