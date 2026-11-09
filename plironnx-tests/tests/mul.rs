// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The plironnx contributors

//! Test from onnx-tests' tests/mul/mod.rs (burn-onnx repo)

#[allow(dead_code)]
mod mul_model {
    include!(concat!(env!("OUT_DIR"), "/model/mul.rs"));
}

use burn::tensor::{Device, Tensor};

mod common;
#[path = "common/jit.rs"]
mod jit;
mod utils;

fn run_plironnx(input: Tensor<4>, scalar: f64) -> Tensor<4> {
    let input_data = input.to_data();
    let input_descriptor = jit::build_input_descriptor::<f32>(&input_data);
    let mut output = jit::build_empty_output_descriptor(4);
    jit::run_packed(
        "mul/mul.onnx",
        &[input_descriptor.as_ptr(), (&raw const scalar).cast()],
        &[output.as_mut_ptr()],
    );
    jit::read_f32_output(&output, [1, 2, 3, 4])
}

#[test]
fn mul_ir() {
    utils::assert_ir_snapshots(
        "mul/mul.onnx",
        [
            "mul_onnx.expect.plir",
            "mul_tensor.expect.plir",
            "mul_llvm.expect.plir",
        ],
    );
}

#[test]
fn mul_scalar_with_tensor_and_tensor_with_tensor() {
    let device: Device = Default::default();
    let input = Tensor::<4>::from_floats(
        [[
            [[1., 2., 3., 4.], [5., 6., 7., 8.], [9., 10., 11., 12.]],
            [
                [13., 14., 15., 16.],
                [17., 18., 19., 20.],
                [21., 22., 23., 24.],
            ],
        ]],
        &device,
    );
    let scalar = 6.0f64;
    let expected = mul_model::Model::default().forward(input.clone(), scalar);
    let actual = run_plironnx(input, scalar);
    actual.to_data().assert_eq(&expected.to_data(), true);
}
