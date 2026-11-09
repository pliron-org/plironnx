// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The plironnx contributors

//! Test from onnx-tests' tests/matmul/mod.rs (burn-onnx repo).

#[allow(dead_code)]
mod matmul_ranks_model {
    include!(concat!(env!("OUT_DIR"), "/model/matmul_ranks.rs"));
}

use burn::tensor::{Device, Int, Tensor};

mod common;
#[path = "common/jit.rs"]
mod jit;
mod utils;

#[test]
fn matmul_ranks_ir() {
    utils::assert_ir_snapshots(
        "matmul/matmul_ranks.onnx",
        [
            "matmul_ranks_onnx.expect.plir",
            "matmul_ranks_tensor.expect.plir",
            "matmul_ranks_llvm.expect.plir",
        ],
    );
}

/// Check that matrix products of rank 1, 2 and 3 operands, in all
/// combinations that the test model contains, match Burn.
#[test]
fn matmul_ranks() {
    let device: Device = Default::default();
    let mat2d = Tensor::<1, Int>::arange(0..12, &device)
        .float()
        .reshape([3, 4]);
    let mat3d = Tensor::<1, Int>::arange(0..24, &device)
        .float()
        .reshape([2, 3, 4]);
    let vec = Tensor::<1, Int>::arange(0..4, &device).float();
    let vec3 = Tensor::<1, Int>::arange(0..3, &device).float();
    let mat2d_square = Tensor::<1, Int>::arange(0..16, &device)
        .float()
        .reshape([4, 4]);
    let mat3d_for_vec = Tensor::<1, Int>::arange(0..24, &device)
        .float()
        .reshape([2, 3, 4]);
    let expected = matmul_ranks_model::Model::default().forward(
        mat2d.clone(),
        mat3d.clone(),
        vec.clone(),
        vec3.clone(),
        mat2d_square.clone(),
        mat3d_for_vec.clone(),
    );

    let input_data = [
        mat2d.to_data(),
        mat3d.to_data(),
        vec.to_data(),
        vec3.to_data(),
        mat2d_square.to_data(),
        mat3d_for_vec.to_data(),
    ];
    let input_descriptors = input_data
        .each_ref()
        .map(jit::build_input_descriptor::<f32>);
    let mut outputs = [1, 1, 2, 2, 2].map(jit::build_empty_output_descriptor);
    jit::run_packed(
        "matmul/matmul_ranks.onnx",
        &input_descriptors
            .each_ref()
            .map(|descriptor| descriptor.as_ptr()),
        &outputs.each_mut().map(|descriptor| descriptor.as_mut_ptr()),
    );

    jit::read_f32_output(&outputs[0], [3])
        .to_data()
        .assert_eq(&expected.0.to_data(), true);
    jit::read_f32_output(&outputs[1], [4])
        .to_data()
        .assert_eq(&expected.1.to_data(), true);
    jit::read_f32_output(&outputs[2], [2, 3])
        .to_data()
        .assert_eq(&expected.2.to_data(), true);
    jit::read_f32_output(&outputs[3], [2, 4])
        .to_data()
        .assert_eq(&expected.3.to_data(), true);
    jit::read_f32_output(&outputs[4], [3, 4])
        .to_data()
        .assert_eq(&expected.4.to_data(), true);
}
