"""Evaluation script for the DeepLab-ResNet network on the validation subset
   of PASCAL VOC dataset.

This script evaluates the model on 1449 validation images.
"""

from __future__ import print_function

import argparse
from datetime import datetime
import os
import sys
import time

import tensorflow as tf
import numpy as np
from PIL import Image

from attacks import fgs
from deeplab_resnet import DeepLabResNetModel, ImageReader, prepare_label, inv_preprocess

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

BATCH_SIZE = 1
DATA_DIRECTORY = '/home/abhagoji/VOCdevkit/VOC2012'
CROPPED_DATA_DIRECTORY = './dataset'
CROPPED_DATA = './dataset/cropped_images/'
CROPPED_LABELS = './dataset/cropped_labels/'
CROPPED_NOISE = './dataset/cropped_noise/'
DATA_LIST_PATH = './dataset/val-cropped.txt'
IGNORE_LABEL = 255
INPUT_SIZE = '321,321'
NUM_CLASSES = 21
NUM_STEPS = 1449 # Number of images in the validation set.
# NUM_STEPS = 1
RESTORE_FROM = './deeplab_resnet.ckpt'
MASK_FLAG = 1
SAVE_FLAG = 1

def image_converter(image_list, step):
    pic = Image.open(image_list[step])
    X = np.array(pic.getdata()).reshape(pic.size[1], pic.size[0], 3)
    X = X[:, :, ::-1]
    X = X - IMG_MEAN
    X= X.reshape(BATCH_SIZE, pic.size[1], pic.size[0], 3)

    return X

def label_converter(label_list, step):
    pic = Image.open(label_list[step])
    pic = pic.convert('L')
    Y = np.array(pic.getdata()).reshape(pic.size[1], pic.size[0], 1)
    Y= Y.reshape(BATCH_SIZE, pic.size[1], pic.size[0], 1)

    return Y


def read_labeled_image_list(data_dir, data_list, ADV_FLAG=None, MASK_FLAG=None, eps=None, attack=None):
    """Reads txt file containing paths to images and ground truth masks.
    
    Args:
      data_dir: path to the directory with images and masks.
      data_list: path to the file with lines of the form '/path/to/image /path/to/mask'.
       
    Returns:
      Two lists with all file names for images and masks, respectively.
    """
    f = open(data_list, 'r')
    images = []
    masks = []
    for line in f:
        try:
            image, mask = line.strip("\n").split(' ')
            if ADV_FLAG is not None:
              image = image.strip('.jpg')
              image = image + '_' + str(eps) + '_' + attack
              if MASK_FLAG is not None:
                image += '_masked.jpg'
              else:
                image += '.jpg'
        except ValueError: # Adhoc for test.
            image = mask = line.strip("\n")
        images.append(data_dir + image)
        masks.append(data_dir + mask)
    return images, masks


def image_saver(X_adv, X, image_name, attack, eps, targeted):
    X = inv_preprocess(X,1,IMG_MEAN)
    X_adv = inv_preprocess(X_adv,1,IMG_MEAN)
    name = os.path.basename(image_name)
    name = name.strip(".jpg")
    name_adv = name + '_' + str(eps) + '_' + attack
    if MASK_FLAG is not None:
        name_adv += '_masked'
    if targeted:
        name_adv += '_T'
    name_r = name_adv + '_noise'
    name_adv += '.jpg'
    name_r += '.jpg'
    name_label = name + '.png'
    name += '.jpg'
    im_adv = Image.fromarray((X_adv[0]).astype(np.uint8))
    r = Image.fromarray((X_adv[0]-X[0]).astype(np.uint8))
    # im = Image.fromarray((X[0]).astype(np.uint8))
    # im_label = Image.fromarray((Y[0,:,:,0]).astype(np.uint8), mode = 'L')
    # im.save(CROPPED_DATA + name)
    im_adv.save(CROPPED_DATA + name_adv)
    r.save(CROPPED_NOISE + name_r)
    # im_label.save(CROPPED_LABELS + name_label)

def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLabLFOV Network")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--data-dir", type=str, default=CROPPED_DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of images in the validation set.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--attack", type=str, default='fgs',
                        help="attack type")
    parser.add_argument("--eps", type=float, default=4.0,
                        help="Perturbation value")
    parser.add_argument("--beta", type=float, default=0.5,
                        help="IFGS Perturbation value")
    parser.add_argument("--iter", type=int, default=10,
                        help="IFGS steps")
    parser.add_argument("--targeted", action='store_true')
    return parser.parse_args()

def load(saver, sess, ckpt_path):
    '''Load trained weights.
    
    Args:
      saver: TensorFlow saver object.
      sess: TensorFlow session.
      ckpt_path: path to checkpoint file with parameters.
    ''' 
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))

def main():
    """Create the model and start the evaluation process."""
    args = get_arguments()

    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)
    
    # Create queue coordinator.
    coord = tf.train.Coordinator()
    
    # Load reader.
    with tf.name_scope("create_inputs"):
        reader = ImageReader(
            args.data_dir,
            args.data_list,
            input_size, # No defined input size.
            False, # No random scale.
            False, # No random mirror.
            args.ignore_label,
            IMG_MEAN,
            coord)
        # image, label = reader.image, reader.label
        image_batch, label_batch = reader.dequeue(args.batch_size)
    # image_batch, label_batch = tf.expand_dims(image, dim=0), tf.expand_dims(label, dim=0) # Add one batch dimension.

    image_name_list = reader.image_list

    x = tf.placeholder(tf.float32,shape=(BATCH_SIZE,h,w,3))
    y = tf.placeholder(tf.float32,shape=(BATCH_SIZE,h,w,1))

    # Create network.
    net = DeepLabResNetModel({'data': x}, is_training=False, num_classes=args.num_classes)

    # Which variables to load.
    restore_var = tf.global_variables()
    
    # Predictions.
    raw_output = net.layers['fc1_voc12']

    raw_prediction = tf.reshape(raw_output, [-1, args.num_classes])
    label_proc = prepare_label(y, tf.stack(raw_output.get_shape()[1:3]), num_classes=args.num_classes, one_hot=False) # [batch_size, h, w]
    raw_gt = tf.reshape(label_proc, [-1,])
    indices_0 = tf.squeeze(tf.where(tf.less_equal(raw_gt, args.num_classes - 1)), 1)
    gt_0 = tf.cast(tf.gather(raw_gt, indices_0), tf.int32)
    prediction = tf.gather(raw_prediction, indices_0)

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=gt_0)
    img_mean_t = tf.convert_to_tensor(IMG_MEAN, dtype=tf.float32)

    if args.attack == 'fgs':
        x_adv = fgs(x, loss, args.eps, img_mean_t, input_size, BATCH_SIZE, MASK_FLAG, args.targeted)
    elif args.attack == 'ifgs':
        x_adv = fgs(x, loss, args.beta, img_mean_t, input_size, BATCH_SIZE, MASK_FLAG, args.targeted)

    raw_output = tf.image.resize_bilinear(raw_output, tf.shape(image_batch)[1:3,])
    raw_output = tf.argmax(raw_output, dimension=3)
    pred = tf.expand_dims(raw_output, dim=3) # Create 4-d tensor.

    # mIoU
    pred = tf.reshape(pred, [-1,])
    gt = tf.reshape(label_batch, [-1,])
    indices = tf.squeeze(tf.where(tf.less_equal(gt, args.num_classes - 1)), 1) ## ignore all labels >= num_classes
    gt = tf.cast(tf.gather(gt, indices), tf.int32)
    pred = tf.gather(pred, indices)
    mIoU, update_op = tf.contrib.metrics.streaming_mean_iou(pred, gt, num_classes=args.num_classes)

    # Set up tf session and initialize variables. 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    
    sess.run(init)
    sess.run(tf.local_variables_initializer())
    
    # Load weights.
    loader = tf.train.Saver(var_list=restore_var)
    if args.restore_from is not None:
        load(loader, sess, args.restore_from)
    
    # Start queue threads.
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    
    # Iterate over evaluation steps.
    image_list, label_list = read_labeled_image_list(args.data_dir,args.data_list)
    for step in range(args.num_steps):
        X = image_converter(image_list, step)
        if args.targeted:
            Y = np.zeros((BATCH_SIZE,321,321,1))
        else:
            Y = label_converter(label_list, step)
        if args.attack == 'fgs':
            preds, _, X_adv = sess.run([pred, update_op, x_adv], feed_dict={x: X,y: Y})
        elif args.attack == 'ifgs':
            X_adv = X
            for i in range(args.iter):
                preds, _, X_adv, loss_v = sess.run([pred, update_op, x_adv, loss], feed_dict={x: X_adv,y: Y})
                r = np.clip(X_adv - X, -args.eps, args.eps)
                X_adv = X + r
        # preds, _ = sess.run([pred, update_op], feed_dict={x: X_adv})
        if SAVE_FLAG is not None:
            image_saver(X_adv, X, image_name_list[step], args.attack, args.eps, args.targeted)
        if step % 100 == 0:
            print('step {:d}'.format(step))
    print('Mean IoU: {:.3f}'.format(mIoU.eval(session=sess)))
    coord.request_stop()
    coord.join(threads)
    
if __name__ == '__main__':
    main()
