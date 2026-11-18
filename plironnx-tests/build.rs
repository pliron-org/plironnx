// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The plironnx contributors

use burn_onnx::ModelGen;

#[path = "tests/common/mod.rs"]
mod common;

fn main() {
    let models = [
        "add/add.onnx",
        "add/add_broadcast.onnx",
        "sub/sub.onnx",
        "sub/sub_broadcast.onnx",
        "sub/sub_int.onnx",
        "mul/mul.onnx",
        "mul/mul_broadcast.onnx",
        "div/div.onnx",
        "div/div_broadcast.onnx",
        "matmul/matmul.onnx",
        "matmul/matmul_ranks.onnx",
        "linear/linear.onnx",
        "gemm/gemm.onnx",
        "gemm/gemm_no_c.onnx",
        "gemm/gemm_non_unit_alpha_beta.onnx",
        "relu/relu.onnx",
        "constant/constant_f32.onnx",
        "constant/constant_tensor_f32.onnx",
        "cast/cast.onnx",
        "initializer_to_const/initializer_to_const.onnx",
        "identity/identity_only.onnx",
        "reshape/reshape.onnx",
        "flatten/flatten.onnx",
        "flatten/flatten_2d.onnx",
        "flatten/flatten_rank1.onnx",
        "exp/exp.onnx",
        "reduce/reduce_max.onnx",
        "reduce/reduce_sum.onnx",
        "softmax/softmax.onnx",
    ]
    .map(|model| common::onnx_tests_dir().join(model));
    let local_models = [
        "mlp/mlp.onnx",
        "mlp/mlp_softmax.onnx",
        "softmax/softmax_middle_axis.onnx",
    ]
    .map(|model| common::local_models_dir().join(model));

    let mut model_gen = ModelGen::new();
    for model in models.iter().chain(&local_models) {
        println!("cargo:rerun-if-changed={}", model.display());
        model_gen.input(model.to_str().expect("non-utf8 path"));
    }
    model_gen.out_dir("model/").run_from_script();
}
