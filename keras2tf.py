import argparse
import os
from pathlib import Path

import tensorflow as tf
from keras import backend as K
from keras.models import model_from_json, load_model
from tensorflow.python.framework import graph_io
from tensorflow.python.framework import graph_util

MODEL_JSON_FILE = "./KR/model.json"
WEIGHT_FILE = "./KR/weights.h5"

MODE_JSON = 0
MODE_HDF5 = 1


def convert_and_save(keras_model_path, weight_path, dry_run=None):
    """Tries to open keras model (h5 or json format) and optionally applies weight from file.
       Loaded model is saved as tensorflow compatible pb file."""
    K.set_learning_phase(0)
    K.set_image_data_format('channels_last')
    name = Path(keras_model_path).name
    extension = name.split('.')[-1]
    if extension == "json":
        model = model_from_json(keras_model_path)
    else:
        model = load_model(keras_model_path)

    if weight_path is not None:
        model.load_weights(weight_path)

    num_output = 1
    pred = [None] * num_output
    pred_node_names = [None] * num_output

    for i in range(num_output):
        pred_node_names[i] = "out_" + str(i)
        pred[i] = tf.identity(model.outputs[i], name=pred_node_names[i])
    print('Found output nodes: ', pred_node_names)

    if dry_run is not None:
        return  # just show output nodes

    out_file_name = name + ".pb"

    if os.path.isfile("./" + out_file_name):
        os.remove("./" + out_file_name)

    sess = K.get_session()
    constatn_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), pred_node_names)
    graph_io.write_graph(constatn_graph, ".", out_file_name, as_text=False)


def main():
    """Sample usage: python3 keras.json weights.h5
      \nSample usage: python3 keras.h5
      \nSample usage: python3 keras.h5 weights.h5"""
    parser = argparse.ArgumentParser(description="Convert keras model to tensorflow pb model file")
    parser.add_argument("keras_model", help="Keras model file (json or h5)", type=str)
    parser.add_argument("weight_file", help="Additional weight file for model", type=str, default=None, nargs='?')
    parser.add_argument("--dry", help="Dry run, without saving output", type=bool)

    args = parser.parse_args()

    convert_and_save(args.keras_model, args.weight_file, args.dry)


if __name__ == "__main__":
    main()
