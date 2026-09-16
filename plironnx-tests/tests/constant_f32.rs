// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The plironnx contributors

//! Tests from onnx-tests' tests/constant/mod.rs (burn-onnx repo)

#[allow(dead_code)]
mod constant_f32_model {
    include!(concat!(env!("OUT_DIR"), "/model/constant_f32.rs"));
}

use burn::tensor::{Device, Tensor};

mod common;
#[path = "common/constant.rs"]
mod constant_utils;
mod utils;

type Backend = burn::backend::Flex;

#[test]
fn constant_f32_ir() {
    utils::assert_ir_snapshots(
        "constant/constant_f32.onnx",
        [
            "constant_f32_onnx.expect.plir",
            "constant_f32_tensor.expect.plir",
            "constant_f32_llvm.expect.plir",
        ],
    );
}

#[test]
fn constant_f32() {
    let device: Device<Backend> = Default::default();
    let input = Tensor::<Backend, 3>::zeros([2, 3, 4], &device);
    let expected = constant_f32_model::Model::<Backend>::default().forward(input.clone());
    let actual = constant_utils::run_plironnx("constant/constant_f32.onnx", &input);
    actual.to_data().assert_eq(&expected.to_data(), true);
}
