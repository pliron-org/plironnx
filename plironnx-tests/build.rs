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
        "constant/constant_f32.onnx",
        "constant/constant_tensor_f32.onnx",
        "cast/cast.onnx",
        "initializer_to_const/initializer_to_const.onnx",
        "identity/identity_only.onnx",
        "reshape/reshape.onnx",
    ]
    .map(|model| common::onnx_tests_dir().join(model));

    let mut model_gen = ModelGen::new();
    for model in &models {
        println!("cargo:rerun-if-changed={}", model.display());
        model_gen.input(model.to_str().expect("non-utf8 path"));
    }
    model_gen.out_dir("model/").run_from_script();
}
