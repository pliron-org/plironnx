// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The plironnx contributors

//! Test from onnx-tests' tests/mul/mod.rs (burn-onnx repo)

#[allow(dead_code)]
mod mul_broadcast_model {
    include!(concat!(env!("OUT_DIR"), "/model/mul_broadcast.rs"));
}

use burn::tensor::{Device, Tensor};

mod common;
#[path = "common/jit.rs"]
mod jit;
mod utils;

fn run_broadcast_plironnx(
    x_3d: &Tensor<3>,
    y_2d: &Tensor<2>,
    a_2d: &Tensor<2>,
    b_3d: &Tensor<3>,
) -> (Tensor<3>, Tensor<3>) {
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
        "mul/mul_broadcast.onnx",
        &input_descriptors
            .each_ref()
            .map(|descriptor| descriptor.as_ptr()),
        &outputs.each_mut().map(|descriptor| descriptor.as_mut_ptr()),
    );
    (
        jit::read_f32_output(&outputs[0], [2, 3, 4]),
        jit::read_f32_output(&outputs[1], [2, 3, 4]),
    )
}

#[test]
fn mul_broadcast_ir() {
    utils::assert_ir_snapshots(
        "mul/mul_broadcast.onnx",
        [
            "mul_broadcast_onnx.expect.plir",
            "mul_broadcast_tensor.expect.plir",
            "mul_broadcast_llvm.expect.plir",
        ],
    );
}

#[test]
fn mul_broadcast_tensor_ranks() {
    let device: Device = Default::default();
    let x_3d = Tensor::<3>::from_floats(
        [
            [[1., 2., 3., 4.], [5., 6., 7., 8.], [9., 10., 11., 12.]],
            [[2., 3., 4., 5.], [6., 7., 8., 9.], [10., 11., 12., 13.]],
        ],
        &device,
    );
    let y_2d = Tensor::<2>::from_floats(
        [[2., 2., 2., 2.], [3., 3., 3., 3.], [4., 4., 4., 4.]],
        &device,
    );
    let a_2d = Tensor::<2>::from_floats(
        [[1., 2., 3., 4.], [5., 6., 7., 8.], [9., 10., 11., 12.]],
        &device,
    );
    let b_3d = Tensor::<3>::from_floats(
        [
            [[2., 2., 2., 2.], [1., 1., 1., 1.], [0.5, 0.5, 0.5, 0.5]],
            [[3., 3., 3., 3.], [2., 2., 2., 2.], [1., 1., 1., 1.]],
        ],
        &device,
    );
    let expected = mul_broadcast_model::Model::default().forward(
        x_3d.clone(),
        y_2d.clone(),
        a_2d.clone(),
        b_3d.clone(),
    );
    let actual = run_broadcast_plironnx(&x_3d, &y_2d, &a_2d, &b_3d);
    actual.0.to_data().assert_eq(&expected.0.to_data(), true);
    actual.1.to_data().assert_eq(&expected.1.to_data(), true);
}
