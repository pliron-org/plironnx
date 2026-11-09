// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The plironnx contributors

//! Tests from onnx-tests' tests/constant/mod.rs (burn-onnx repo)

#[allow(dead_code)]
mod constant_f32_model {
    include!(concat!(env!("OUT_DIR"), "/model/constant_f32.rs"));
}

use burn::tensor::{Device, Tensor};

mod common;
#[path = "common/jit.rs"]
mod jit;
mod utils;

#[test]
fn constant_f32_ir() {
    utils::assert_ir_snapshots(
        "constant/constant_f32.onnx",
        [
            "constant_f32_onnx.expect.plir",
            "constant_f32_tensor.expect.plir",
            "constant_f32_llvm.expect.plir",
        ],
    );
}

#[test]
fn constant_f32() {
    let device: Device = Default::default();
    let input = Tensor::<3>::zeros([2, 3, 4], &device);
    let expected = constant_f32_model::Model::default().forward(input.clone());
    let actual = jit::run_single_f32("constant/constant_f32.onnx", &input, input.dims());
    actual.to_data().assert_eq(&expected.to_data(), true);
}
