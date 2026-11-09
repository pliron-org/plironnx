// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The plironnx contributors

//! Test from onnx-tests' tests/reshape/mod.rs (burn-onnx repo).

#[allow(dead_code)]
mod reshape_model {
    include!(concat!(env!("OUT_DIR"), "/model/reshape.rs"));
}

use burn::tensor::{Device, Tensor};

mod common;
#[path = "common/jit.rs"]
mod jit;
mod utils;

#[test]
fn reshape_ir() {
    utils::assert_ir_snapshots(
        "reshape/reshape.onnx",
        [
            "reshape_onnx.expect.plir",
            "reshape_tensor.expect.plir",
            "reshape_llvm.expect.plir",
        ],
    );
}

#[test]
fn reshape() {
    let device: Device = Default::default();
    let input = Tensor::<1>::from_floats([0., 1., 2., 3.], &device);
    let expected = reshape_model::Model::new(&device).forward(input.clone());
    let actual = jit::run_single_f32("reshape/reshape.onnx", &input, [1, 4]);
    actual.to_data().assert_eq(&expected.to_data(), true);
}
