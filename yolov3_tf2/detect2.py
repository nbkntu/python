import time
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import numpy as np
import tensorflow as tf
from yolov3_tf2.yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.yolov3_tf2.dataset import transform_images, load_tfrecord_dataset
from yolov3_tf2.yolov3_tf2.utils import draw_outputs

#def initialize_flags(image_id: str, image_path: str):
#    flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
#    flags.DEFINE_string('weights', './checkpoints/yolov3.tf',
#                    'path to weights file')
#    flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
#    flags.DEFINE_integer('size', 416, 'resize images to')
#    flags.DEFINE_string('image', image_path, 'path to input image')
#    flags.DEFINE_string('tfrecord', None, 'tfrecord instead of image')
#    flags.DEFINE_string('output', './output.jpg', 'path to output image')
#    flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')

def detect(i_image_id: str, 
            i_image_path: str,
            i_classes='./data/coco.names',
            i_weights='./checkpoints/yolov3.tf',
            i_tiny=False,
            i_size=416,
            i_tfrecord=None,
            i_output='./output.jpg',
            i_num_classes=80):
    #initialize_flags(image_id, image_path)
    process(i_image_id, 
            i_image_path,
            i_classes,
            i_weights,
            i_tiny,
            i_size,
            i_tfrecord,
            i_output,
            i_num_classes)

def process(i_image_id: str, 
            i_image_path: str,
            i_classes: str,
            i_weights: str,
            i_tiny: bool,
            i_size: int,
            i_tfrecord,
            i_output: str,
            i_num_classes: int):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)

    if i_tiny:
        yolo = YoloV3Tiny(classes=i_num_classes)
    else:
        yolo = YoloV3(classes=i_num_classes)

    yolo.load_weights(i_weights).expect_partial()
    logging.info('weights loaded')

    class_names = [c.strip() for c in open(i_classes).readlines()]
    logging.info('classes loaded')

    if i_tfrecord:
        dataset = load_tfrecord_dataset(
            i_tfrecord, i_classes, i_size)
        dataset = dataset.shuffle(512)
        img_raw, _label = next(iter(dataset.take(1)))
    else:
        img_raw = tf.image.decode_image(
            open(i_image_path, 'rb').read(), channels=3)

    img = tf.expand_dims(img_raw, 0)
    img = transform_images(img, i_size)

    t1 = time.time()
    boxes, scores, classes, nums = yolo(img)
    t2 = time.time()
    logging.info('time: {}'.format(t2 - t1))

    logging.info('detections:')
    for i in range(nums[0]):
        logging.info('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                           np.array(scores[0][i]),
                                           np.array(boxes[0][i])))

    img = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR)
    img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
    cv2.imwrite(i_output, img)
    logging.info('output saved to: {}'.format(i_output))


#if __name__ == '__main__':
#    try:
#        app.run(main)
#    except SystemExit:
#        pass
