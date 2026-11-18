// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The plironnx contributors

//! Softmax tests:
//! - the model in onnx-tests' tests/softmax (burn-onnx repo)
//! - the model in models/softmax (see models/softmax/softmax_middle_axis.py).

#[allow(dead_code)]
mod softmax_model {
    include!(concat!(env!("OUT_DIR"), "/model/softmax.rs"));
}

#[allow(dead_code)]
mod softmax_middle_axis_model {
    include!(concat!(env!("OUT_DIR"), "/model/softmax_middle_axis.rs"));
}

use burn::tensor::{Device, Int, Tensor, Tolerance};

mod common;
#[path = "common/jit.rs"]
mod jit;
mod utils;

#[test]
fn softmax_ir() {
    utils::assert_ir_snapshots(
        "softmax/softmax.onnx",
        [
            "softmax_onnx.expect.plir",
            "softmax_tensor.expect.plir",
            "softmax_llvm.expect.plir",
        ],
    );
}

/// Check that the softmax along dimension 1 matches Burn.
#[test]
fn softmax() {
    let device: Device = Default::default();
    let input = Tensor::<2>::from_floats(
        [
            [0.33669037, 0.128_809_4, 0.23446237],
            [0.23033303, -1.122_856_4, -0.18632829],
        ],
        &device,
    );
    let expected = softmax_model::Model::default().forward(input.clone());
    let actual = jit::run_single_f32("softmax/softmax.onnx", &input, [2, 3]);
    actual
        .to_data()
        .assert_approx_eq::<f32>(&expected.to_data(), Tolerance::default());
}

/// Check that the softmax along the middle dimension of a rank-3 tensor matches Burn.
#[test]
fn softmax_middle_axis() {
    let device: Device = Default::default();
    let input = Tensor::<1, Int>::arange(0..24, &device)
        .float()
        .mul_scalar(0.7)
        .sin()
        .mul_scalar(3.0)
        .reshape([2, 3, 4]);
    let expected = softmax_middle_axis_model::Model::default().forward(input.clone());
    let actual = jit::run_single_f32(
        common::local_models_dir().join("softmax/softmax_middle_axis.onnx"),
        &input,
        [2, 3, 4],
    );
    actual
        .to_data()
        .assert_approx_eq::<f32>(&expected.to_data(), Tolerance::default());
}
