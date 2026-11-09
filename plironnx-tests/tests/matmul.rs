// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The plironnx contributors

//! Test from onnx-tests' tests/matmul/mod.rs (burn-onnx repo).

#[allow(dead_code)]
mod matmul_model {
    include!(concat!(env!("OUT_DIR"), "/model/matmul.rs"));
}

use burn::tensor::{Device, Int, Tensor};

mod common;
#[path = "common/jit.rs"]
mod jit;
mod utils;

#[test]
fn matmul_ir() {
    utils::assert_ir_snapshots(
        "matmul/matmul.onnx",
        [
            "matmul_onnx.expect.plir",
            "matmul_tensor.expect.plir",
            "matmul_llvm.expect.plir",
        ],
    );
}

/// Check that a batched matrix product and the products of a batched matrix
/// with a vector on each side match Burn.
#[test]
fn matmul() {
    let device: Device = Default::default();
    let a = Tensor::<1, Int>::arange(0..24, &device)
        .float()
        .reshape([1, 2, 3, 4]);
    let b = Tensor::<1, Int>::arange(0..16, &device)
        .float()
        .reshape([1, 2, 4, 2]);
    let c = Tensor::<1, Int>::arange(0..96, &device)
        .float()
        .reshape([2, 3, 4, 4]);
    let d = Tensor::<1, Int>::arange(0..4, &device).float();
    let (expected_mm, expected_mv, expected_vm) =
        matmul_model::Model::default().forward(a.clone(), b.clone(), c.clone(), d.clone());

    let input_data = [a.to_data(), b.to_data(), c.to_data(), d.to_data()];
    let input_descriptors = input_data
        .each_ref()
        .map(jit::build_input_descriptor::<f32>);
    let mut outputs = [
        jit::build_empty_output_descriptor(4),
        jit::build_empty_output_descriptor(3),
        jit::build_empty_output_descriptor(3),
    ];
    jit::run_packed(
        "matmul/matmul.onnx",
        &input_descriptors
            .each_ref()
            .map(|descriptor| descriptor.as_ptr()),
        &outputs.each_mut().map(|descriptor| descriptor.as_mut_ptr()),
    );

    jit::read_f32_output(&outputs[0], [1, 2, 3, 2])
        .to_data()
        .assert_eq(&expected_mm.to_data(), true);
    jit::read_f32_output(&outputs[1], [2, 3, 4])
        .to_data()
        .assert_eq(&expected_mv.to_data(), true);
    jit::read_f32_output(&outputs[2], [2, 3, 4])
        .to_data()
        .assert_eq(&expected_vm.to_data(), true);
}
