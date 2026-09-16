// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The plironnx contributors

use burn_onnx::ModelGen;

#[path = "tests/common/mod.rs"]
mod common;

fn main() {
    let add_onnx = common::onnx_tests_dir().join("add/add.onnx");
    let add_broadcast_onnx = common::onnx_tests_dir().join("add/add_broadcast.onnx");
    let mul_onnx = common::onnx_tests_dir().join("mul/mul.onnx");
    let mul_broadcast_onnx = common::onnx_tests_dir().join("mul/mul_broadcast.onnx");
    let constant_f32_onnx = common::onnx_tests_dir().join("constant/constant_f32.onnx");
    let constant_tensor_f32_onnx =
        common::onnx_tests_dir().join("constant/constant_tensor_f32.onnx");
    let cast_onnx = common::onnx_tests_dir().join("cast/cast.onnx");
    let initializer_to_const_onnx =
        common::onnx_tests_dir().join("initializer_to_const/initializer_to_const.onnx");
    let identity_only_onnx = common::onnx_tests_dir().join("identity/identity_only.onnx");
    let reshape_onnx = common::onnx_tests_dir().join("reshape/reshape.onnx");
    println!("cargo:rerun-if-changed={}", add_onnx.display());
    println!("cargo:rerun-if-changed={}", add_broadcast_onnx.display());
    println!("cargo:rerun-if-changed={}", mul_onnx.display());
    println!("cargo:rerun-if-changed={}", mul_broadcast_onnx.display());
    println!("cargo:rerun-if-changed={}", constant_f32_onnx.display());
    println!(
        "cargo:rerun-if-changed={}",
        constant_tensor_f32_onnx.display()
    );
    println!("cargo:rerun-if-changed={}", cast_onnx.display());
    println!(
        "cargo:rerun-if-changed={}",
        initializer_to_const_onnx.display()
    );
    println!("cargo:rerun-if-changed={}", identity_only_onnx.display());
    println!("cargo:rerun-if-changed={}", reshape_onnx.display());

    ModelGen::new()
        .input(add_onnx.to_str().expect("non-utf8 path"))
        .input(add_broadcast_onnx.to_str().expect("non-utf8 path"))
        .input(mul_onnx.to_str().expect("non-utf8 path"))
        .input(mul_broadcast_onnx.to_str().expect("non-utf8 path"))
        .input(constant_f32_onnx.to_str().expect("non-utf8 path"))
        .input(constant_tensor_f32_onnx.to_str().expect("non-utf8 path"))
        .input(cast_onnx.to_str().expect("non-utf8 path"))
        .input(initializer_to_const_onnx.to_str().expect("non-utf8 path"))
        .input(identity_only_onnx.to_str().expect("non-utf8 path"))
        .input(reshape_onnx.to_str().expect("non-utf8 path"))
        .out_dir("model/")
        .run_from_script();
}
