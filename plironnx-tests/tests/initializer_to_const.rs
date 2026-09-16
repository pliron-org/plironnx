// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The plironnx contributors

//! Test from onnx-tests' tests/initializer_to_const/mod.rs (burn-onnx repo).

#[allow(dead_code)]
mod initializer_to_const_model {
    include!(concat!(env!("OUT_DIR"), "/model/initializer_to_const.rs"));
}

use burn::tensor::{Device, Tensor};

mod common;
#[path = "common/single_input_f32.rs"]
mod single_input_f32;
mod utils;

type Backend = burn::backend::Flex;

#[test]
fn initializer_to_const_ir() {
    utils::assert_ir_snapshots(
        "initializer_to_const/initializer_to_const.onnx",
        [
            "initializer_to_const_onnx.expect.plir",
            "initializer_to_const_tensor.expect.plir",
            "initializer_to_const_llvm.expect.plir",
        ],
    );
}

#[test]
fn initializer_to_const() {
    let device: Device<Backend> = Default::default();
    let input = Tensor::<Backend, 2>::from_floats([[1., 1., 1.], [1., 1., 1.]], &device);
    let expected = initializer_to_const_model::Model::<Backend>::default().forward(input.clone());
    let actual = single_input_f32::run_plironnx(
        "initializer_to_const/initializer_to_const.onnx",
        &input,
        input.dims(),
    );
    actual.to_data().assert_eq(&expected.to_data(), true);
}
