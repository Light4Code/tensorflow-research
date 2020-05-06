import os
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import (
    convert_variables_to_constants_v2,
)
from tensorflow.lite.python.util import get_grappler_config, run_graph_optimizations


def export_model(model, path: str, save_frozen: bool = True, model_name: str = "frozen_graph") -> None:
    if not os.path.exists(path):
        os.makedirs(path)

    tf.saved_model.save(model, path)

    if save_frozen == True:
        graph = _create_frozen_graph(model)
        tf.io.write_graph(graph, ".", path +
                          "/{0}.pb".format(model_name), as_text=False)


def _create_frozen_graph(model):
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
