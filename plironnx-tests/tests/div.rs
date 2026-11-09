// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The plironnx contributors

//! Test from onnx-tests' tests/div/mod.rs (burn-onnx repo).

#[allow(dead_code)]
mod div_model {
    include!(concat!(env!("OUT_DIR"), "/model/div.rs"));
}

use burn::tensor::{Device, Int, Tensor, Tolerance};

mod common;
#[path = "common/jit.rs"]
mod jit;
mod utils;

#[test]
fn div_ir() {
    utils::assert_ir_snapshots(
        "div/div.onnx",
        [
            "div_onnx.expect.plir",
            "div_tensor.expect.plir",
            "div_llvm.expect.plir",
        ],
    );
}

/// Check that a scalar divided by a scalar and a tensor divided by the result match Burn.
#[test]
fn div_tensor_by_scalar_and_tensor_by_tensor() {
    let device: Device = Default::default();
    // The model declares a [1, 2, 3, 4] input, and compiled shapes are static.
    let input = Tensor::<1, Int>::arange(0..24, &device)
        .float()
        .reshape([1, 2, 3, 4]);
    let (scalar1, scalar2) = (9.0f64, 3.0f64);
    let expected = div_model::Model::new(&device).forward(input.clone(), scalar1, scalar2);

    let input_data = input.to_data();
    let input_descriptor = jit::build_input_descriptor::<f32>(&input_data);
    let mut output = jit::build_empty_output_descriptor(4);
    jit::run_packed(
        "div/div.onnx",
        &[
            input_descriptor.as_ptr(),
            (&raw const scalar1).cast(),
            (&raw const scalar2).cast(),
        ],
        &[output.as_mut_ptr()],
    );
    jit::read_f32_output(&output, [1, 2, 3, 4])
        .to_data()
        .assert_approx_eq::<f32>(&expected.to_data(), Tolerance::default());
}
