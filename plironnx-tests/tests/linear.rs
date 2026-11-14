// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The plironnx contributors

//! Test from onnx-tests' tests/linear/mod.rs (burn-onnx repo).

#[allow(dead_code)]
mod linear_model {
    include!(concat!(env!("OUT_DIR"), "/model/linear.rs"));
}

use burn::tensor::{Device, Int, Tensor, Tolerance};

mod common;
#[path = "common/jit.rs"]
mod jit;
mod utils;

#[test]
fn linear_ir() {
    utils::assert_ir_snapshots(
        "linear/linear.onnx",
        [
            "linear_onnx.expect.plir",
            "linear_tensor.expect.plir",
            "linear_llvm.expect.plir",
        ],
    );
}

/// Check that the three sources of a Burn `Linear` match Burn:
/// - a `Gemm` with a bias, whose weight is in the `[out, in]` layout
/// - a `MatMul` without a bias
/// - a `MatMul` and a bias `Add`, with a 3-D input
#[test]
fn linear() {
    let device: Device = Default::default();
    let ramp = |len: i64| {
        Tensor::<1, Int>::arange(0..len, &device)
            .float()
            .sub_scalar(len as f32 / 2.0)
            .div_scalar(len as f32)
    };
    let x1 = ramp(12).reshape([4, 3]);
    let x2 = ramp(10).reshape([2, 5]);
    let x3 = ramp(42).reshape([3, 2, 7]);
    let (expected1, expected2, expected3) =
        linear_model::Model::default().forward(x1.clone(), x2.clone(), x3.clone());

    let input_data = [x1.to_data(), x2.to_data(), x3.to_data()];
    let input_descriptors = input_data
        .each_ref()
        .map(jit::build_input_descriptor::<f32>);
    let mut outputs = [
        jit::build_empty_output_descriptor(2),
        jit::build_empty_output_descriptor(2),
        jit::build_empty_output_descriptor(3),
    ];
    jit::run_packed(
        "linear/linear.onnx",
        &input_descriptors
            .each_ref()
            .map(|descriptor| descriptor.as_ptr()),
        &outputs.each_mut().map(|descriptor| descriptor.as_mut_ptr()),
    );

    let tolerance = Tolerance::<f32>::default();
    jit::read_f32_output(&outputs[0], [4, 4])
        .to_data()
        .assert_approx_eq(&expected1.to_data(), tolerance);
    jit::read_f32_output(&outputs[1], [2, 6])
        .to_data()
        .assert_approx_eq(&expected2.to_data(), tolerance);
    jit::read_f32_output(&outputs[2], [3, 2, 8])
        .to_data()
        .assert_approx_eq(&expected3.to_data(), tolerance);
}
