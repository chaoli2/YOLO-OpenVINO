import numpy as np
import tensorflow as tf

from yolo_v3 import yolo_v3, load_weights, detections_boxes, non_max_suppression

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('class_names', 'coco.names', 'File with class names')
tf.app.flags.DEFINE_string('weights_file', 'yolov3.weights', 'Binary file with detector weights')

tf.app.flags.DEFINE_integer('size', 608, 'Image size')

tf.app.flags.DEFINE_float('conf_threshold', 0.5, 'Confidence threshold')
tf.app.flags.DEFINE_float('iou_threshold', 0.4, 'IoU threshold')


def load_coco_names(file_name):
    names = {}
    with open(file_name) as f:
        for id, name in enumerate(f):
            names[id] = name
    return names


def main(argv=None):
    classes = load_coco_names(FLAGS.class_names)

    # placeholder for detector inputs
    inputs = tf.placeholder(tf.float32, [None, FLAGS.size, FLAGS.size, 3])

    with tf.variable_scope('detector'):
        detections = yolo_v3(inputs, len(classes), data_format='NHWC')
        load_ops = load_weights(tf.global_variables(scope='detector'), FLAGS.weights_file)

    boxes = detections_boxes(detections)    

    with tf.Session() as sess:
        sess.run(load_ops)
        from tensorflow.python.framework import graph_io
        frozen = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, ['concat_1'])
        graph_io.write_graph(frozen, '../build/YOLOv3/', 'yolo_v3.pb', as_text=False)

if __name__ == '__main__':
    tf.app.run()
