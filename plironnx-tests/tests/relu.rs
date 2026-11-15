// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The plironnx contributors

//! Test from onnx-tests' tests/relu/mod.rs (burn-onnx repo).

#[allow(dead_code)]
mod relu_model {
    include!(concat!(env!("OUT_DIR"), "/model/relu.rs"));
}

use burn::tensor::{Device, Tensor};

mod common;
#[path = "common/jit.rs"]
mod jit;
mod utils;

#[test]
fn relu_ir() {
    utils::assert_ir_snapshots(
        "relu/relu.onnx",
        [
            "relu_onnx.expect.plir",
            "relu_tensor.expect.plir",
            "relu_llvm.expect.plir",
        ],
    );
}

/// Check that positive elements are kept and negative elements become zero, as in Burn.
#[test]
fn relu() {
    let device: Device = Default::default();
    let input = Tensor::<2>::from_floats(
        [
            [0.33669037, 0.128_809_4, 0.23446237],
            [0.23033303, -1.122_856_4, -0.18632829],
        ],
        &device,
    );
    let expected = relu_model::Model::default().forward(input.clone());
    let actual = jit::run_single_f32("relu/relu.onnx", &input, [2, 3]);
    actual.to_data().assert_eq(&expected.to_data(), true);
}
