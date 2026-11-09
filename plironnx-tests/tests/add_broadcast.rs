// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The plironnx contributors

//! Tests from onnx-tests' tests/add/mod.rs (burn-onnx repo)

#[allow(dead_code)]
mod add_broadcast_model {
    include!(concat!(env!("OUT_DIR"), "/model/add_broadcast.rs"));
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
        "add/add_broadcast.onnx",
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
fn add_broadcast_ir() {
    utils::assert_ir_snapshots(
        "add/add_broadcast.onnx",
        [
            "add_broadcast_onnx.expect.plir",
            "add_broadcast_tensor.expect.plir",
            "add_broadcast_llvm.expect.plir",
        ],
    );
}

#[test]
fn add_broadcast_tensor_ranks() {
    let device: Device = Default::default();
    let x_3d = Tensor::<3>::from_floats(
        [
            [[1., 2., 3., 4.], [5., 6., 7., 8.], [9., 10., 11., 12.]],
            [
                [13., 14., 15., 16.],
                [17., 18., 19., 20.],
                [21., 22., 23., 24.],
            ],
        ],
        &device,
    );
    let y_2d = Tensor::<2>::from_floats(
        [[1., 1., 1., 1.], [2., 2., 2., 2.], [3., 3., 3., 3.]],
        &device,
    );
    let a_2d = Tensor::<2>::from_floats(
        [
            [0.5, 0.5, 0.5, 0.5],
            [1.5, 1.5, 1.5, 1.5],
            [2.5, 2.5, 2.5, 2.5],
        ],
        &device,
    );
    let b_3d = x_3d.clone();
    let expected = add_broadcast_model::Model::default().forward(
        x_3d.clone(),
        y_2d.clone(),
        a_2d.clone(),
        b_3d.clone(),
    );
    let actual = run_broadcast_plironnx(&x_3d, &y_2d, &a_2d, &b_3d);
    actual.0.to_data().assert_eq(&expected.0.to_data(), true);
    actual.1.to_data().assert_eq(&expected.1.to_data(), true);
}
