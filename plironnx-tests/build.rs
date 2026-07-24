// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The plironnx contributors

use burn_onnx::ModelGen;

#[path = "tests/common/mod.rs"]
mod common;

fn main() {
    let add_onnx = common::onnx_tests_dir().join("add/add.onnx");
    println!("cargo:rerun-if-changed={}", add_onnx.display());

    ModelGen::new()
        .input(add_onnx.to_str().expect("non-utf8 path"))
        .out_dir("model/")
        .run_from_script();
}
