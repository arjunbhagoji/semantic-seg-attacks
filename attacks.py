import tensorflow as tf

CLIP_MIN = 0
CLIP_MAX = 255

from utils import gen_mask


def fgs(x, loss, eps, img_mean_t, input_size, BATCH_SIZE, MASK_FLAG, targeted):
    grads = tf.gradients(loss,x)[0]

    if targeted:
        if MASK_FLAG is not None:
            mask = gen_mask(input_size, BATCH_SIZE)
            x_adv = x - eps * tf.multiply(mask,tf.sign(grads))
        else:
            x_adv = x - eps * tf.sign(grads)
    else:
        if MASK_FLAG is not None:
            mask = gen_mask(input_size, BATCH_SIZE)
            x_adv = x + eps * tf.multiply(mask,tf.sign(grads))
        else:
            x_adv = x + eps * tf.sign(grads)

    x_adv = tf.clip_by_value(x_adv+img_mean_t, CLIP_MIN, CLIP_MAX)
    # r = tf.clip_by_value(x_adv - x, -eps, eps)
    # x_adv = x + r

    x_adv = x_adv - img_mean_t

    return x_adv
