// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The plironnx contributors

//! Test from onnx-tests' tests/exp/mod.rs (burn-onnx repo).

#[allow(dead_code)]
mod exp_model {
    include!(concat!(env!("OUT_DIR"), "/model/exp.rs"));
}

use burn::tensor::{Device, Tensor, Tolerance};

mod common;
#[path = "common/jit.rs"]
mod jit;
mod utils;

#[test]
fn exp_ir() {
    utils::assert_ir_snapshots(
        "exp/exp.onnx",
        [
            "exp_onnx.expect.plir",
            "exp_tensor.expect.plir",
            "exp_llvm.expect.plir",
        ],
    );
}

/// Check that the exponential of each element matches Burn.
#[test]
fn exp() {
    let device: Device = Default::default();
    let input = Tensor::<4>::from_floats([[[[0.0, std::f32::consts::LN_2]]]], &device);
    let expected = exp_model::Model::default().forward(input.clone());
    let actual = jit::run_single_f32("exp/exp.onnx", &input, [1, 1, 1, 2]);
    actual
        .to_data()
        .assert_approx_eq::<f32>(&expected.to_data(), Tolerance::default());
}
