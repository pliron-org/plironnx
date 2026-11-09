// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The plironnx contributors

//! Test from onnx-tests' tests/div/mod.rs (burn-onnx repo).

#[allow(dead_code)]
mod div_broadcast_model {
    include!(concat!(env!("OUT_DIR"), "/model/div_broadcast.rs"));
}

use burn::tensor::{Device, Int, Tensor, Tolerance};

mod common;
#[path = "common/jit.rs"]
mod jit;
mod utils;

#[test]
fn div_broadcast_ir() {
    utils::assert_ir_snapshots(
        "div/div_broadcast.onnx",
        [
            "div_broadcast_onnx.expect.plir",
            "div_broadcast_tensor.expect.plir",
            "div_broadcast_llvm.expect.plir",
        ],
    );
}

/// Check that broadcasting a rank-2 operand against a rank-3 operand on either side matches Burn.
#[test]
fn div_broadcast_tensor_ranks() {
    let device: Device = Default::default();
    let x_3d = Tensor::<1, Int>::arange(1..25, &device)
        .float()
        .mul_scalar(10.0)
        .reshape([2, 3, 4]);
    let y_2d = Tensor::<1, Int>::arange(1..13, &device)
        .float()
        .reshape([3, 4]);
    let a_2d = Tensor::<1, Int>::arange(1..13, &device)
        .float()
        .mul_scalar(100.0)
        .reshape([3, 4]);
    let b_3d = Tensor::<1, Int>::arange(1..25, &device)
        .float()
        .mul_scalar(5.0)
        .reshape([2, 3, 4]);
    let expected = div_broadcast_model::Model::default().forward(
        x_3d.clone(),
        y_2d.clone(),
        a_2d.clone(),
        b_3d.clone(),
    );

    let input_data = [
        x_3d.to_data(),
        y_2d.to_data(),
        a_2d.to_data(),
        b_3d.to_data(),
    ];
    let input_descriptors = input_data
        .each_ref()
        .map(jit::build_input_descriptor::<f32>);
    let mut outputs = [
        jit::build_empty_output_descriptor(3),
        jit::build_empty_output_descriptor(3),
    ];
    jit::run_packed(
        "div/div_broadcast.onnx",
        &input_descriptors
            .each_ref()
            .map(|descriptor| descriptor.as_ptr()),
        &outputs.each_mut().map(|descriptor| descriptor.as_mut_ptr()),
    );
    for (output, expected) in outputs.iter().zip([expected.0, expected.1]) {
        jit::read_f32_output(output, [2, 3, 4])
            .to_data()
            .assert_approx_eq::<f32>(&expected.to_data(), Tolerance::default());
    }
}
