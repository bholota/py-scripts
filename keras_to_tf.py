from pathlib import Path

import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
from keras import backend as K
from keras.models import load_model

INPUT_FILE = "/home/mr3mpty/machine_learning/clothulhu.h5"
OUTPUT_DIR = "/home/mr3mpty/machine_learning"
OUTPUT_FILE = str(Path(INPUT_FILE).name).split('.')[0] + '.pb'


def dice_coef(y_true, y_pred, smooth=0.9):
    y_true_binary = K.flatten(K.greater(y_true, 0))
    y_true_f = K.flatten(K.cast(y_true_binary, 'float32'))
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def main():
    K.set_learning_phase(0)
    K.set_image_data_format('channels_last')

    try:
        model = load_model(INPUT_FILE, custom_objects={
            'dice_coef_loss': dice_coef_loss,
            'dice_coef': dice_coef
        })
    except ValueError as err:
        print("File doesn't contain proper architecture")
        raise err

    num_output = 1
    pred = [None] * num_output
    pred_node_names = [None] * num_output

    for i in range(num_output):
        pred_node_names[i] = "out_" + str(i)
        pred[i] = tf.identity(model.outputs[i], name=pred_node_names[i])
    print('output nodes names are: ', pred_node_names)

    sess = K.get_session()
    constatn_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), pred_node_names)
    graph_io.write_graph(constatn_graph, OUTPUT_DIR, OUTPUT_FILE, as_text=False)


if __name__ == "__main__":
    main()
