import argparse
from utils import *
import tensorflow as tf
from tensorflow.lite.python.util import run_graph_optimizations, get_grappler_config
from tensorflow.python.framework.convert_to_constants import (
    convert_variables_to_constants_v2,
)


def main():
    parser = argparse.ArgumentParser(description="Used to export TensorFlow model")
    parser.add_argument(
        "config",
        metavar="config",
        help="Path to the configuration file containing all parameters for model training",
    )
    parser.add_argument(
        "--checkpoint_path",
        dest="checkpoint_path",
        metavar="path",
        help="Path to the checkpoint weights that should be exported",
    )
    parser.add_argument(
        "--output_path",
        dest="output_path",
        metavar="path",
        help="Path where the models should be saved",
    )
    parser.add_argument(
        "--save_frozen_graph",
        dest="save_frozen_graph",
        metavar="bool",
        type=bool,
        help="If true, the model will be also saved as frozen graph",
    )

    args = parser.parse_args()
    config_path = args.config
    config = Config(config_path)

    if not args.checkpoint_path:
        ValueError
    else:
        config.train.checkpoint_path = args.checkpoint_path

    model_container = create_model(config)

    tf.saved_model.save(model_container.model, args.output_path)
    graph_def = frozen_keras_graph(model_container.model)
    if args.save_frozen_graph and args.save_frozen_graph == True:
        tf.io.write_graph(graph_def, ".", args.output_path + "/frozen_graph.pb")


def frozen_keras_graph(model):
    real_model = tf.function(model).get_concrete_function(
        tf.TensorSpec(
            model.inputs[0].shape, model.inputs[0].dtype, name=model.inputs[0].name
        )
    )
    frozen_func = convert_variables_to_constants_v2(real_model)
    graph_def = frozen_func.graph.as_graph_def()

    input_tensors = [
        tensor for tensor in frozen_func.inputs if tensor.dtype != tf.resource
    ]
    output_tensors = frozen_func.outputs

    graph_def = run_graph_optimizations(
        graph_def,
        input_tensors,
        output_tensors,
        config=get_grappler_config(["constfold", "function"]),
        graph=frozen_func.graph,
    )

    return graph_def


if __name__ == "__main__":
    main()
