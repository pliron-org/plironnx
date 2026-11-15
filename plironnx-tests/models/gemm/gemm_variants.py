#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.23.2",
#   "numpy",
# ]
# ///

# Generate gemm_variants.onnx. It has
# - Gemm nodes in the forms that burn does not fuse into Linear and
# - one node that burn fuses into Linear.

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OPSET_VERSION = 16

def build_model():
    product_weight = np.arange(6).reshape(3, 2)
    initializers = [
        numpy_helper.from_array(np.arange(-6, 6, dtype=np.int32).reshape(4, 3), "w"),
        numpy_helper.from_array(
            np.array([np.nan, np.inf, -np.inf, 1.0], dtype=np.float32), "d"
        ),
        numpy_helper.from_array(product_weight.astype(np.float64), "w_f64"),
        numpy_helper.from_array(product_weight.astype(np.float16), "w_f16"),
        numpy_helper.from_array(product_weight.astype(np.uint32), "w_u32"),
    ]
    nodes = [
        # Runtime operands transposed by `transA` and `transB`, and a row bias.
        helper.make_node(
            "Gemm",
            ["a", "b", "c"],
            ["y_f32"],
            alpha=2.0,
            beta=-1.0,
            transA=1,
            transB=1,
        ),
        # The initializer `w` transposed by `transB`, and a column bias.
        helper.make_node(
            "Gemm", ["p", "w", "q"], ["y_i32"], alpha=3.0, beta=2.0, transB=1
        ),
        # burn fuses this node into Linear, with the runtime weight `b`.
        helper.make_node(
            "Gemm", ["b", "b", "c"], ["y_fused"], alpha=1.0, beta=1.0, transB=1
        ),
        # `beta = 0` with the bias `d`, which holds NaN and infinities.
        helper.make_node(
            "Gemm", ["a", "b", "d"], ["y_beta0"], beta=0.0, transA=1, transB=1
        ),
        # Scaling of f64, f16 and u32 elements.
        helper.make_node("Gemm", ["x_f64", "w_f64"], ["y_f64"], alpha=0.5),
        helper.make_node("Gemm", ["x_f16", "w_f16"], ["y_f16"], alpha=0.5),
        helper.make_node("Gemm", ["x_u32", "w_u32"], ["y_u32"], alpha=3.0),
    ]
    graph = helper.make_graph(
        nodes,
        "gemm_variants",
        [
            helper.make_tensor_value_info("a", TensorProto.FLOAT, [3, 2]),
            helper.make_tensor_value_info("b", TensorProto.FLOAT, [4, 3]),
            helper.make_tensor_value_info("c", TensorProto.FLOAT, [4]),
            helper.make_tensor_value_info("p", TensorProto.INT32, [2, 3]),
            helper.make_tensor_value_info("q", TensorProto.INT32, [2, 1]),
            helper.make_tensor_value_info("x_f64", TensorProto.DOUBLE, [2, 3]),
            helper.make_tensor_value_info("x_f16", TensorProto.FLOAT16, [2, 3]),
            helper.make_tensor_value_info("x_u32", TensorProto.UINT32, [2, 3]),
        ],
        [
            helper.make_tensor_value_info("y_f32", TensorProto.FLOAT, [2, 4]),
            helper.make_tensor_value_info("y_i32", TensorProto.INT32, [2, 4]),
            helper.make_tensor_value_info("y_fused", TensorProto.FLOAT, [4, 4]),
            helper.make_tensor_value_info("y_beta0", TensorProto.FLOAT, [2, 4]),
            helper.make_tensor_value_info("y_f64", TensorProto.DOUBLE, [2, 2]),
            helper.make_tensor_value_info("y_f16", TensorProto.FLOAT16, [2, 2]),
            helper.make_tensor_value_info("y_u32", TensorProto.UINT32, [2, 2]),
        ],
        initializer=initializers,
    )
    return helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)]
    )


if __name__ == "__main__":
    onnx_model = build_model()
    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, "gemm_variants.onnx")
    print("Finished exporting model to gemm_variants.onnx")
