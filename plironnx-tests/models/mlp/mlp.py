#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.23.2",
#   "numpy",
# ]
# ///

# Generate mlp.onnx: an MNIST-sized multilayer perceptron for
# `Flatten -> Linear -> ReLU -> Linear`.
#
#   x: [3, 1, 28, 28] -> Flatten -> [3, 784] -> Gemm -> [3, 32] -> Relu -> Gemm -> [3, 10]

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OPSET_VERSION = 16
BATCH, IN, HIDDEN, OUT = 3, 28 * 28, 32, 10


def linear_params(rng, name, fan_in, fan_out):
    # The default initialization of a PyTorch `Linear`, with the weight in its `[out, in]` layout.
    bound = 1.0 / np.sqrt(fan_in)
    weight = rng.uniform(-bound, bound, (fan_out, fan_in)).astype(np.float32)
    bias = rng.uniform(-bound, bound, (fan_out,)).astype(np.float32)
    return [
        numpy_helper.from_array(weight, f"{name}.weight"),
        numpy_helper.from_array(bias, f"{name}.bias"),
    ]


def build_model():
    rng = np.random.default_rng(seed=0)
    initializers = linear_params(rng, "fc1", IN, HIDDEN) + linear_params(
        rng, "fc2", HIDDEN, OUT
    )
    nodes = [
        helper.make_node("Flatten", ["x"], ["flat"], axis=1),
        # Without `alpha` and `beta`, burn keeps this node a Gemm.
        helper.make_node(
            "Gemm", ["flat", "fc1.weight", "fc1.bias"], ["hidden"], transB=1
        ),
        helper.make_node("Relu", ["hidden"], ["activated"]),
        # As in a PyTorch export, with `alpha = beta = 1`, burn fuses this node into Linear.
        helper.make_node(
            "Gemm", ["activated", "fc2.weight", "fc2.bias"], ["logits"], alpha=1.0, beta=1.0, transB=1
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "mlp",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [BATCH, 1, 28, 28])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [BATCH, OUT])],
        initializer=initializers,
    )
    return helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)]
    )


if __name__ == "__main__":
    onnx_model = build_model()
    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, "mlp.onnx")
    print("Finished exporting model to mlp.onnx")
