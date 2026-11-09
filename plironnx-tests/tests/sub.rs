// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The plironnx contributors

//! Test from onnx-tests' tests/sub/mod.rs (burn-onnx repo).

#[allow(dead_code)]
mod sub_model {
    include!(concat!(env!("OUT_DIR"), "/model/sub.rs"));
}

use burn::tensor::{Device, Int, Tensor};

mod common;
#[path = "common/jit.rs"]
mod jit;
mod utils;

#[test]
fn sub_ir() {
    utils::assert_ir_snapshots(
        "sub/sub.onnx",
        [
            "sub_onnx.expect.plir",
            "sub_tensor.expect.plir",
            "sub_llvm.expect.plir",
        ],
    );
}

/// Check that a scalar minus a tensor and a tensor minus a tensor match Burn.
#[test]
fn sub_scalar_from_tensor_and_tensor_from_tensor() {
    let device: Device = Default::default();
    // The model declares a [1, 2, 3, 4] input, and compiled shapes are static.
    let input = Tensor::<1, Int>::arange(0..24, &device)
        .float()
        .reshape([1, 2, 3, 4]);
    let scalar = 3.0f64;
    let expected = sub_model::Model::default().forward(input.clone(), scalar);

    let input_data = input.to_data();
    let input_descriptor = jit::build_input_descriptor::<f32>(&input_data);
    let mut output = jit::build_empty_output_descriptor(4);
    jit::run_packed(
        "sub/sub.onnx",
        &[input_descriptor.as_ptr(), (&raw const scalar).cast()],
        &[output.as_mut_ptr()],
    );
    jit::read_f32_output(&output, [1, 2, 3, 4])
        .to_data()
        .assert_eq(&expected.to_data(), true);
}
