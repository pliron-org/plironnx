#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.23.2",
#   "numpy",
# ]
# ///

# Generate softmax_middle_axis.onnx: `Softmax` along axis 1 of a rank-3 tensor,
# so that the reduced dimension is neither the first nor the last one.
#
#   x: [2, 3, 4] -> Softmax(axis=1) -> [2, 3, 4]

import onnx
from onnx import TensorProto, helper

OPSET_VERSION = 16
SHAPE = [2, 3, 4]


def build_model():
    graph = helper.make_graph(
        [helper.make_node("Softmax", ["x"], ["y"], axis=1)],
        "softmax_middle_axis",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, SHAPE)],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, SHAPE)],
    )
    return helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)]
    )


if __name__ == "__main__":
    onnx_model = build_model()
    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, "softmax_middle_axis.onnx")
    print("Finished exporting model to softmax_middle_axis.onnx")
