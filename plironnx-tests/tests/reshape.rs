// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The plironnx contributors

//! Test from onnx-tests' tests/reshape/mod.rs (burn-onnx repo).

#[allow(dead_code)]
mod reshape_model {
    include!(concat!(env!("OUT_DIR"), "/model/reshape.rs"));
}

use burn::tensor::{Device, Tensor};

mod common;
#[path = "common/single_input_f32.rs"]
mod single_input_f32;
mod utils;

type Backend = burn::backend::Flex;

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
    let device: Device<Backend> = Default::default();
    let input = Tensor::<Backend, 1>::from_floats([0., 1., 2., 3.], &device);
    let expected = reshape_model::Model::<Backend>::new(&device).forward(input.clone());
    let actual = single_input_f32::run_plironnx("reshape/reshape.onnx", &input, [1, 4]);
    actual.to_data().assert_eq(&expected.to_data(), true);
}
